from pathlib import Path
import os

# Configurazione percorso
DATA_DIR = Path("/shared/marangi/projects/EVOCITY/lulc_esri/data/esri_lulc_patches")

def check_dataset_integrity(data_dir=DATA_DIR):
    print(f"🔍 Analisi integrità cartella: {data_dir}")
    
    if not data_dir.exists():
        print("❌ La cartella data non esiste!")
        return

    regions = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"📂 Trovate {len(regions)} cartelle regione.\n")
    
    empty_regions = []
    mismatched_regions = []
    total_images = 0
    
    print(f"{'REGIONE':<10} | {'IMMAGINI':<10} | {'MASCHERE':<10} | {'STATO':<10}")
    print("-" * 50)
    
    for region in regions:
        img_dir = region / "images"
        mask_dir = region / "masks"
        
        # Conta file .tif (ignora eventuali file nascosti o parziali)
        n_imgs = len(list(img_dir.glob("*.tif"))) if img_dir.exists() else 0
        n_masks = len(list(mask_dir.glob("*.tif"))) if mask_dir.exists() else 0
        
        status = "✅ OK"
        if n_imgs == 0:
            status = "⚠️ VUOTA"
            empty_regions.append(region.name)
        elif n_imgs != n_masks:
            status = "❌ ERROR"
            mismatched_regions.append(region.name)
        
        total_images += n_imgs
        print(f"{region.name:<10} | {n_imgs:<10} | {n_masks:<10} | {status}")

    print("-" * 50)
    print(f"\n📊 RIEPILOGO:")
    print(f"   Totale Patch Salvate: {total_images}")
    print(f"   Regioni Vuote ({len(empty_regions)}): {empty_regions}")
    
    if mismatched_regions:
        print(f"   ❌ Regioni Disallineate (Img != Mask): {mismatched_regions}")
    else:
        print(f"   ✅ Nessun disallineamento trovato (ogni immagine ha la sua maschera).")

if __name__ == "__main__":
    data_dir = DATA_DIR
    check_dataset_integrity(data_dir=data_dir)