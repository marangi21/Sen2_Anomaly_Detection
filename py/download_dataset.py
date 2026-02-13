from pystac_client import Client
import planetary_computer as pc
from pathlib import Path
from contextlib import ExitStack
import rasterio as rio
from tqdm import tqdm
import numpy as np
from rasterio.vrt import WarpedVRT
from argparse import ArgumentParser
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from check_data_integrity import check_dataset_integrity

# Visual references:
    # https://www.researchgate.net/figure/MGRS-encoding-of-Northern-Italy-Figure-2a-shows-the-subdivision-of-GZIs-in-100-km_fig1_317988792
    # https://coordinates-converter.com/en/decimal/37.770715,12.700195?karte=OpenStreetMap&zoom=6

TARGET_REGIONS = [
    # --- ZONA 32 (Nord Ovest, Tirreno, Sardegna) ---
    "32TNS", "32TNT", "32TPS", "32TPT", "32TQS", "32TQT",                   # Alpi / Piemonte / Lombardia Alta
    "32TMR", "32TMS", "32TNR", "32TPR", "32TQR",                            # Pianura Padana Ovest
    "32TLP", "32TLQ", "32TMP", "32TMQ", "32TNP", "32TNQ",                   # Liguria / Toscana Nord
    "32TPP", "32TPQ", "32TQP", "32TQQ", "32TPN", "32TQN", "32TQM",          # Emilia / Toscana / Lazio Nord
    "32TML", "32TNL", "32TMK", "32TNK", "32SMJ", "32SNJ",                   # Sardegna

    # --- ZONA 33 (Nord Est, Adriatico, Centro, Sud, Sicilia) ---
    "33TUL", "33TUM", "33TUN", "33TQR", "33TQS", "33TQT",                   # Trentino / Veneto / Friuli
    "33TTG", "33TPN", "33TPP", "33TQQ",                                     # Emilia Romagna / Marche / Abruzzo / Umbria
    "33TUF", "33TUG", "33TUH", "33TUJ",                                     # Umbria / Lazio / Marche
    "33TVF", "33TVG",                                                       # Molise / Campania Nord
    "33TWG", "33TWF", "33TWE",                                              # Puglia / Basilicata / Campania
    "33TXF", "33TXE", "33TYF", "33TYE",                                     # Puglia
    "33SWD", "33SXD", "33SWC", "33SXC",                                     # Calabria
    "33STB", "33SUB", "33SVB", "33SWB", "33SVA",                            # Sicilia

    # --- ZONA 34 (Puglia Lecce - Otranto) ---
    "34TBK"                                                                 # Salento
]

# Bande da scaricare (https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/bands/)
REQUESTED_BANDS= ["B04", "B03", "B02", "B08", "B11", "B12"]  # RGB + NIR + SWIR1 + SWIR2
PATCH_SIZE = 512
STRIDE = 512
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "esri_lulc_patches"
YEARS = [2023, 2022, 2021, 2020, 2019, 2018, 2017] # Anni candidati per Backfilling Temporale delle patch scartate

def get_sentinel_and_esri_items_multitemporal(region_code, years=YEARS):
    """
    Scarica i metadati STAC per una data regione su più anni.
    Restituisce un dizionario: { 2023: (sentinel_item, esri_item), ... }
    """
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = Client.open(catalog_url)
    items_by_year = {}

    for year in years:
        s_items = []
        for attempt in range(3):
            try:
                # Cerca immagine Sentinel-2 (Estate, Cloud < 10%) con meno nuvole in assoluto nel periodo estivo
                search = client.search(
                    collections=["sentinel-2-l2a"],
                    datetime=f"{year}-06-01/{year}-08-30",
                    query={
                        "eo:cloud_cover": {"lt": 10},
                        "s2:mgrs_tile": {"eq": region_code}
                    },
                    sortby=[{"field": "eo:cloud_cover", "direction": "asc"}], # ordina per cloud cover
                    max_items=1
                )
                s_items = list(search.item_collection())
                break
            except Exception as e:
                print(f"Warning: Timeout/Errore su {region_code} {year} (Tentativo {attempt+1}/3). Ritento tra 5s...")
                time.sleep(5)
        if not s_items:
            continue
        best_sentinel = s_items[0] # La prima è la migliore (meno nuvole)
        
        # Cerca mappa ESRI per quell'anno
        e_items = []
        for attempt in range(3):
            try:
                esri_search = client.search(
                    collections=["io-lulc-9-class"],
                    intersects=best_sentinel.geometry,
                    datetime=f"{year}-01-01/{year}-12-31",
                    max_items=1
                )
                e_items = list(esri_search.item_collection())
                break
            except Exception as e:
                print(f"Warning: Timeout/Errore ESRI {region_code} {year} (Tentativo {attempt+1}/3). Ritento tra 5s...")
                time.sleep(5)
        if not e_items:
             continue
        
        # Firma e ritorna
        items_by_year[year] = (pc.sign(best_sentinel), pc.sign(e_items[0]))
    
    return items_by_year

def process_region_multitemporal(region_code, output_dir=OUTPUT_DIR, patch_size=PATCH_SIZE, stride=STRIDE, years=YEARS, requested_bands=REQUESTED_BANDS):
    yearly_items = get_sentinel_and_esri_items_multitemporal(region_code, years)
    if not yearly_items:
        print(f"Skipping {region_code}: Nessun dato trovato in nessuno degli anni {years}")
        return

    region_dir = output_dir / region_code
    (region_dir / "images").mkdir(parents=True, exist_ok=True)
    (region_dir / "masks").mkdir(parents=True, exist_ok=True)

    # Setup geometrico usando il primo disponibile come riferimento
    ref_year = list(yearly_items.keys())[0]
    ref_sentinel = yearly_items[ref_year][0]
    # Uso B04 (10m GSD) come ancora geometrica
    with rio.open(pc.sign_url(ref_sentinel.assets["B04"].href)) as src:
        ref_profile = src.profile.copy()
        ref_transform = src.transform
        ref_crs = src.crs
        width, height = src.width, src.height
    
    tqdm.write(f"Downloading {region_code}...")
    stats = {"evaluated": 0, "saved": 0, "backfilled": 0}
    
    # Lista classi SCL da scartare
    # 0: NoData, 1: Saturated/Defective, 8: Cloud Medium Prob, 9: Cloud High Prob, 10: Cirrus
    # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    BAD_SCL_CLASSES = [0, 1, 8, 9, 10]

    # Tiling loop
    for y in tqdm(range(0, height - patch_size, stride), desc=f"   └─ {region_code}", leave=False):
        for x in range(0, width - patch_size, stride):
            stats["evaluated"] += 1
            window = rio.windows.Window(x, y, patch_size, patch_size)

            # Loop temporale (2023, se fallisce 2022, se fallisce 2021, ...)
            for year in years:
                if year not in yearly_items: continue
                
                sentinel_item, esri_item = yearly_items[year]

                try:
                    with ExitStack() as stack:
                        # Warp (Nearest Neighbor) SCL da 20m GSD sulla griglia 10m GSD per fixare il problema di troppe patch scartate
                        scl_url = pc.sign_url(sentinel_item.assets["SCL"].href)
                        scl_raw = stack.enter_context(rio.open(scl_url))
                        scl_src = stack.enter_context(WarpedVRT(
                            scl_raw, crs=ref_crs, transform=ref_transform, 
                            width=width, height=height, 
                            resampling=rio.enums.Resampling.nearest
                        ))

                        # Check SCL (NoData/Cloud/Cirrus)
                        patch_scl = scl_src.read(1, window=window)
                        if np.isin(patch_scl, BAD_SCL_CLASSES).sum() > (0.01 * patch_size * patch_size):
                            continue # Anno scartato (nuvole), proviamo l'anno prima


                        # Check Contenuto Immagine (dati mancanti nelle bande, buchi neri)
                        # Warp (Bilinear) bande immagine per allineare bande SWIR da 20m a 10m GSD
                        band_srcs = []
                        for band in requested_bands:
                            band_url = pc.sign_url(sentinel_item.assets[band].href)
                            band_raw = stack.enter_context(rio.open(band_url))
                            if band in ["B11", "B12"]: # SWIR 20m, resampling
                                vrt_band = stack.enter_context(WarpedVRT(
                                    band_raw, crs=ref_crs, transform=ref_transform, 
                                    width=width, height=height, 
                                    resampling=rio.enums.Resampling.bilinear
                                ))
                                band_srcs.append(vrt_band)
                            else : # Bande 10m, no resampling
                                band_srcs.append(band_raw)
                        patch_bands = [src.read(1, window=window) for src in band_srcs]
                        patch_image = np.stack(patch_bands)
                        
                        if np.all(patch_image == 0, axis=0).sum() > (0.01 * patch_size * patch_size):
                            continue # Anno scartato (dati mancanti)

                        # Check Water Glint (riflesso del sole sull'acqua)
                        nir_band = patch_image[3] # B08 NIR
                        water_mask = (patch_scl == 6) # SCL 6 = Water
                        water_count = np.sum(water_mask)
                        # solo se più del 50% dell'immagine è acqua
                        if water_count > (0.5 * patch_size * patch_size):
                            # Se più del 10% dei pixel d'acqua ha NIR > 3000, scarta
                            if np.sum(nir_band[water_mask] > 3000) > (0.1 * water_count):
                                continue # Anno scartato (glint)
                        
                        # Caricamento maschera ESRI per quell'anno
                        esri_url = pc.sign_url(esri_item.assets["data"].href)
                        esri_raw = stack.enter_context(rio.open(esri_url))
                        # Warp anche per ESRI per sicurezza di allineamento geometrico
                        esri_src = stack.enter_context(WarpedVRT(
                            esri_raw, crs=ref_crs, transform=ref_transform, 
                            width=width, height=height, 
                            resampling=rio.enums.Resampling.nearest
                        ))
                        patch_mask = esri_src.read(1, window=window)
                        if np.all(patch_mask == 0):
                            continue # Anno scartato (Maschera vuota)

                        # A questo punto la patch è valida, salva
                        # Calcolo trasformata locale per il GeoTIFF (fondamentale per georeferenziare la patch)
                        patch_transform = rio.windows.transform(window, ref_transform)
                        
                        # Nome file univoco con anno: tile_{x}_{y}_{YEAR}.tif
                        tile_name = f"tile_{x}_{y}_{year}"
                        img_profile = ref_profile.copy()
                        img_profile.update({
                            'driver': 'GTiff', 'height': patch_size, 'width': patch_size,
                            'count': len(REQUESTED_BANDS), 'transform': patch_transform, 'compress': 'lzw'
                            })
                        with rio.open(region_dir / "images" / f"{tile_name}_img.tif", 'w', **img_profile) as dst:
                                dst.write(patch_image)

                        mask_profile = ref_profile.copy()
                        mask_profile.update({
                            'driver': 'GTiff', 'height': patch_size, 'width': patch_size,
                            'count': 1, 'transform': patch_transform, 'dtype': 'uint8', 'compress': 'lzw'
                            })
                        with rio.open(region_dir / "masks" / f"{tile_name}_mask.tif", 'w', **mask_profile) as dst:
                                dst.write(patch_mask.astype(np.uint8), 1)
                        
                        stats["saved"] += 1
                        if year != years[0]:
                            stats["backfilled"] += 1
                        
                        # Trovata l'immagine valida, interruzione loop temporale e passo al prossimo tassello x, y
                        break 

                except Exception as e:
                    print(f"Errore lettura tile {x}_{y} anno {year}: {e}")
                    continue
    
    tqdm.write(f"Fine {region_code}: {stats['saved']} salvate ({stats['backfilled']} da backfilling)")

def download_dataset_multitemporal(regions=TARGET_REGIONS, requested_bands=REQUESTED_BANDS, patch_size=PATCH_SIZE, stride=STRIDE, output_dir=OUTPUT_DIR, years=YEARS):
    tqdm.write(f"Salvataggio immagini in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Determina automaticamente il numero di worker in base alla CPU della macchina
    num_workers = os.cpu_count() or 1
    tqdm.write(f"Avvio download con temporal backfilling: Num workers = {num_workers} | Aree = {len(regions)}")
    
    # ProcessPoolExecutor parallelizza ogni regione su un processo dedicato
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Mappa ogni regione ad un job futuro
        future_to_region = {
            executor.submit(
                process_region_multitemporal,
                region,
                output_dir=output_dir,
                years=years,
                patch_size=patch_size,
                stride=stride,
                requested_bands=requested_bands
            ): region for region in regions
        }
        # Gestione dei risultati e degli errori man mano che i processi finiscono
        for future in tqdm(as_completed(future_to_region), total=len(regions), desc="Avanzamento Totale", unit="regione"):
            region = future_to_region[future]
            try:
                future.result()  # Ottiene il risultato o solleva eccezione
            except Exception as e:
                tqdm.write(f"Errore critico su regione {region}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    
    parser = ArgumentParser(description="Download dataset LULC ESRI con logica multitemporale.")
    parser.add_argument("--regions", nargs="+", default=TARGET_REGIONS, help="Lista di regioni MGRS da processare.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Directory di output per i dati scaricati.")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS, help="Lista di anni da considerare per il backfilling temporale.")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE, help="Dimensione del patch (in pixel).")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Stride per il tiling (in pixel).")
    parser.add_argument("--requested_bands", nargs="+", default=REQUESTED_BANDS, help="Bande Sentinel-2 da scaricare.")
    args = parser.parse_args()

    download_dataset_multitemporal(
        regions=args.regions,
        output_dir=Path(args.output_dir),
        years=args.years,
        patch_size=args.patch_size,
        stride=args.stride,
        requested_bands=args.requested_bands
    )
    check_dataset_integrity(data_dir=Path(args.output_dir))