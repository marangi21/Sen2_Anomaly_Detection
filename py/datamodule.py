import lightning.pytorch as pl
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio as rio
import json

####################################################################################
#------------------------------DATAMODULE CLASS-------------------------------------
####################################################################################

class LULCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size=16, num_workers=4, split_seed=42, compute_indicies=True, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed
        self.compute_indicies = compute_indicies
        self.test_areas = []
        self.val_areas = []
        if "test_areas" in kwargs:
            self.test_areas = kwargs["test_areas"]
        if "val_areas" in kwargs:
            self.val_areas = kwargs["val_areas"]

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ])

        self.val_test_transforms = A.Compose([
            ToTensorV2()  
        ])

    def setup(self, stage=None):
        # carica la lista di tutte le regioni
        regions = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        if self.test_areas:
            test_regions = self.test_areas
            val_regions = self.val_areas
            train_regions = [r for r in regions if r not in test_regions + val_regions]
        else:
            test_regions = ["33TUH", "32TMS", "32TLQ", "32SMJ"]
            val_regions = ["33TUL", "32TQT", "33TVF"]
            train_regions = [r for r in regions if r not in test_regions and r not in val_regions]
        
        self.train_items_paths = self._collect_items(train_regions)
        self.val_items_paths = self._collect_items(val_regions)
        self.test_items_paths = self._collect_items(test_regions)

        print(f"Dataset Split:")
        print(f"  Train: {len(train_regions)} regions ({len(self.train_items_paths)} patches)")
        print(f"  Val:   {len(val_regions)} regions ({len(self.val_items_paths)} patches)")
        print(f"  Test:  {len(test_regions)} regions ({len(self.test_items_paths)} patches)")

    def _collect_items(self, region_list):
        items_paths = []
        for region in region_list:
            img_dir = self.data_dir / region / "images"
            mask_dir = self.data_dir / region / "masks"
            if not img_dir.exists() or not mask_dir.exists():
                continue
            for img_path in img_dir.glob("*.tif"):
                mask_name = img_path.name.replace("_img.tif", "_mask.tif")
                mask_path = mask_dir / mask_name
                if mask_path.exists():
                    items_paths.append({
                        "img_path": str(img_path),
                        "mask_path": str(mask_path)
                    })
        return items_paths

    def train_dataloader(self):
        return DataLoader(
            dataset=LULCDataset(
                items_paths=self.train_items_paths,
                transforms=self.train_transforms,
                compute_indicies=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=LULCDataset(
                items_paths=self.val_items_paths,
                transforms=self.val_test_transforms,
                compute_indicies=True),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            dataset=LULCDataset(
                items_paths=self.test_items_paths,
                transforms=self.val_test_transforms,
                compute_indicies=True),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


####################################################################################
#--------------------------------DATASET CLASS--------------------------------------
####################################################################################

class LULCDataset(Dataset):
    def __init__(self, items_paths, transforms=None, compute_indicies=True):
        super().__init__()
        self.items_paths = items_paths
        self.transforms = transforms
        self.compute_indicies = compute_indicies
        self.class_map = {
            1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5, 9: 6, 11: 7
        }
        self.ignore_index = 255
        stats_path = Path(__file__).parent.parent / "data" / "band_stats.json"
        with open(stats_path) as f:
            band_stats = json.load(f)
            self.means = np.array(band_stats["raw_means"], dtype=np.float32).reshape(6, 1, 1)
            self.stds = np.array(band_stats["raw_stds"], dtype=np.float32).reshape(6, 1, 1)

    def __len__(self):
        return len(self.items_paths)

    def __getitem__(self, idx):
        item = self.items_paths[idx]
        # caricamento immagine
        with rio.open(item["img_path"]) as src:
            raw_image = src.read().astype(np.float32) # [C, H, W] - [6, 512, 512]         
        
        # Calcolo indici prima della normalizzazione altrimenti perdono significato fisico + esplodono alcuni valori
        if self.compute_indicies:
            indicies = self._compute_indicies(raw_image)

        # normalizzazione solo delle bande raw, gli indici sono già tra -1 e 1 e vanno bene così
        norm_image = (raw_image - self.means) / self.stds

        # concatenazione
        if self.compute_indicies:
            image = np.concatenate([norm_image, indicies[-3:, :, :]], axis=0)  # [C+3, H, W]
        else:
            image = norm_image

        # caricamento maschera
        with rio.open(item["mask_path"]) as src:
            mask = src.read(1)  # [H, W] - [512, 512]
            mask = self._remap_mask(mask)
        
        # applicazione transforms
        image = np.transpose(image, (1, 2, 0))  # [H, W, C], albumentations lavora con questo formato
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.long() # la maschera deve essere long per la loss
    
    def _compute_indicies(self, image):
        """
        Calcola indici spettrali on-the-fly.
        Img shape: (C, H, W) -> Sentinel-2 bands: R, G, B, NIR, SWIR1, SWIR2 (ordine con cui le ho salvate con WarpedVRT in download_dataset.py)
        Indices: https://github.com/awesome-spectral-indices/awesome-spectral-indices/blob/main/output/spectral-indices-table.csv
        - NDVI = (NIR - Red) / (NIR + Red + eps)
        - NDBI = (SWIR1 - NIR) / (SWIR1 + NIR + eps)
        - NDWI = (Green - NIR) / (Green + NIR + eps)
        """
        eps = 1e-8
        red = image[0]
        green = image[1]
        nir = image[3]
        swir1 = image[4]
        ndvi = (nir - red) / (nir + red + eps)
        ndbi = (swir1 - nir) / (swir1 + nir + eps)
        ndwi = (green - nir) / (green + nir + eps)
        # Clip per sicurezza contro divisioni instabili ai bordi neri
        ndvi = np.clip(ndvi, -1, 1)
        ndbi = np.clip(ndbi, -1, 1)
        ndwi = np.clip(ndwi, -1, 1)

        return np.stack([ndvi, ndbi, ndwi], axis=0)  # [3, H, W]
    
    def _remap_mask(self, mask):
        """
        Remappa le classi della maschera secondo self.class_map.
        Le classi non mappate vengono settate a ignore_index.
        """
        remapped_mask = np.full_like(mask, fill_value=self.ignore_index)
        for original_class, new_class in self.class_map.items():
            remapped_mask[mask == original_class] = new_class
        return remapped_mask