# File: datasets/RedEdgeDataset.py

import os
import random
import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from utils.transforms import Transforms

class RedEdgeDataModule(LightningDataModule):
    # ... (constructor and dataloader methods remain the same as the last correct version) ...
    def __init__(self, full_cfg):
        super().__init__()
        self.full_cfg = full_cfg
        data_cfg = self.full_cfg.get("data", {})
        if not data_cfg:
            raise ValueError("'data' section not found in the configuration.")
        self.root_dir = data_cfg.get("root_dir")
        if self.root_dir is None:
            raise KeyError("'root_dir' not found in data configuration.")

        self.train_fields = data_cfg.get("fields", [])
        self.val_fields = data_cfg.get("val_fields", self.train_fields)
        self.test_fields = data_cfg.get("test_fields", self.val_fields)
        
        # image_sources_type can be "composite" (looks for composite_image_filename)
        # or "bands" (looks for R,G,B specified in channels_list)
        self.image_sources_type = data_cfg.get("image_sources_type", "composite") # "composite" or "bands"
        self.composite_image_filename = data_cfg.get("composite_image_filename", "RGB.png") # e.g., "RGB.png"
        self.band_channels_list = data_cfg.get("band_channels_list", ["R", "G", "B"]) # e.g., ["R", "G", "B"]

        self.gt_folder_name = data_cfg.get("gt_folder_name", "groundtruth")
        
        train_params_cfg = self.full_cfg.get("train", {})
        self.batch_size = train_params_cfg.get("batch_size", data_cfg.get("batch_size", 4))
        self.num_workers = train_params_cfg.get("workers", data_cfg.get("workers", 0))
        self.n_gpus = train_params_cfg.get("n_gpus", data_cfg.get("n_gpus", 1))
        
        self.dataset_init_args = {
            "root_dir": self.root_dir,
            "image_sources_type": self.image_sources_type,
            "composite_image_filename": self.composite_image_filename,
            "band_channels_list": self.band_channels_list,
            "gt_folder_name": self.gt_folder_name,
            "target_size": data_cfg.get("target_size", (1024, 1024))
        }
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.train_fields:
                self.train_dataset = RedEdgeDataset(fields=self.train_fields, **self.dataset_init_args)
            if self.val_fields:
                self.val_dataset = RedEdgeDataset(fields=self.val_fields, **self.dataset_init_args)
        if stage == "test" or stage is None:
            if self.test_fields:
                 self.test_dataset = RedEdgeDataset(fields=self.test_fields, **self.dataset_init_args)
    
    def train_dataloader(self):
        if not self.train_dataset: self.setup(stage="fit")
        if not self.train_dataset: print("Warning: Train dataloader - train_dataset not available."); return None
        return DataLoader(self.train_dataset, batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
    def val_dataloader(self):
        if not self.val_dataset: self.setup(stage="fit")
        if not self.val_dataset: print("Warning: Val dataloader - val_dataset not available."); return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    def test_dataloader(self):
        if not self.test_dataset: self.setup(stage="test")
        if not self.test_dataset: print("Warning: Test dataloader - test_dataset not available."); return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)


class RedEdgeDataset(Dataset):
    def __init__(self, root_dir, fields, image_sources_type, 
                 composite_image_filename, band_channels_list, 
                 gt_folder_name, target_size=(1024,1024)):
        super().__init__()
        self.root_dir = root_dir
        self.fields = fields
        self.image_sources_type = image_sources_type
        self.composite_image_filename = composite_image_filename
        self.band_channels_list = band_channels_list
        self.gt_folder_name = gt_folder_name
        self.target_size = target_size
        self.transform = Transforms()

        self.file_index = []
        print(f"--- Initializing RedEdgeDataset ---")
        print(f"Root dir: {os.path.abspath(root_dir)}")
        print(f"Fields: {fields}")
        print(f"Image sources type: {image_sources_type}")
        if image_sources_type == "composite":
            print(f"Composite image filename: {composite_image_filename}")
        else:
            print(f"Band channels list: {band_channels_list}")
        print(f"GT folder name: {gt_folder_name}")

        for field_name in self.fields:
            print(f"Processing field: {field_name}")
            field_gt_path_root = os.path.join(self.root_dir, field_name, self.gt_folder_name)
            print(f"  Attempting GT root path: {field_gt_path_root} (exists: {os.path.isdir(field_gt_path_root)})")

            if not os.path.isdir(field_gt_path_root):
                print(f"  WARNING: GT path for field {field_name} does not exist or is not a directory.")
                continue
            
            # Iterate through ground truth files first
            gt_filenames_in_folder = os.listdir(field_gt_path_root)
            print(f"  Found {len(gt_filenames_in_folder)} files/dirs in GT folder: {gt_filenames_in_folder[:5]}...")

            for gt_filename in gt_filenames_in_folder:
                if not gt_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    continue # Skip non-image files in GT folder

                print(f"    Processing GT file: {gt_filename}")
                current_gt_full_path = os.path.join(field_gt_path_root, gt_filename)
                image_path_to_use = None
                
                if self.image_sources_type == "composite":
                    # Look for the specific composite image filename (e.g., RGB.png)
                    # in the 'composite-png' folder of the current field
                    expected_composite_image_path = os.path.join(self.root_dir, field_name, "composite-png", self.composite_image_filename)
                    print(f"      Checking specific composite image path: {expected_composite_image_path}")
                    if os.path.exists(expected_composite_image_path):
                        image_path_to_use = expected_composite_image_path
                        print(f"      ---> COMPOSITE IMAGE FOUND: {image_path_to_use}")
                    else:
                        print(f"      ---> COMPOSITE IMAGE '{self.composite_image_filename}' NOT FOUND at: {expected_composite_image_path}")
                
                elif self.image_sources_type == "bands":
                    # This part is more complex if band filenames don't match GT filenames.
                    # For now, let's assume band files (R.png, G.png, B.png) exist for each scene/GT.
                    # This implies a 1-to-1 mapping between a GT file and a set of band files for that scene.
                    # The current structure (R.png, G.png) suggests these are fixed names per field, not per GT file.
                    # This logic needs to be robust if there are multiple GTs and one set of R,G,B per field.
                    # For simplicity now, assume one main set of bands per field.
                    band_paths = {}
                    all_bands_found = True
                    for band_name in self.band_channels_list:
                        # Assuming bands are in 'composite-png' or directly in field folder
                        # Let's assume they are in 'composite-png' as per screenshot
                        band_file_path = os.path.join(self.root_dir, field_name, "composite-png", f"{band_name}.png") # e.g., R.png
                        print(f"      Checking band path: {band_file_path}")
                        if os.path.exists(band_file_path):
                            band_paths[band_name] = band_file_path
                            print(f"      ---> BAND '{band_name}' FOUND: {band_file_path}")
                        else:
                            print(f"      ---> BAND '{band_name}' NOT FOUND at: {band_file_path}")
                            all_bands_found = False
                            break
                    if all_bands_found and band_paths:
                        image_path_to_use = band_paths # Store dict of band paths
                    else:
                         print(f"      ---> NOT ALL BANDS FOUND for field {field_name}. Required: {self.band_channels_list}")

                else:
                    print(f"    Unknown image_sources_type: {self.image_sources_type}")


                if image_path_to_use is not None:
                    print(f"    ---> VALID ENTRY for GT '{gt_filename}'. Using image source(s): {image_path_to_use}. Adding to index.")
                    self.file_index.append({
                        "image_sources_location": image_path_to_use, # Can be a single path or a dict of paths
                        "gt": current_gt_full_path,
                        "name_id": f"{field_name}/{gt_filename}" # For identification
                    })
                else:
                    print(f"    ---> IMAGE SOURCE NOT FOUND for GT '{gt_filename}'. Skipping.")
        
        if not self.file_index:
            print(f"FINAL WARNING: No files were added to the index. file_index is empty.")
        else:
            print(f"--- RedEdgeDataset Initialized. Found {len(self.file_index)} valid items. ---")

    def __len__(self):
        return self.real_size if hasattr(self, 'real_size') and self.real_size is not None else len(self.file_index)


    def __getitem__(self, index):
        entry = self.file_index[index]
        sample = {}

        # --- Image Loading ---
        if self.image_sources_type == "composite":
            image_pil = Image.open(entry["image_sources_location"]).convert("RGB")
        elif self.image_sources_type == "bands":
            pil_images = []
            # Load R, G, B in order if they are specified in band_channels_list and found
            for ch_key in ["R", "G", "B"]: 
                if ch_key in entry["image_sources_location"]: # entry["image_sources_location"] is a dict here
                    pil_images.append(Image.open(entry["image_sources_location"][ch_key]).convert("L"))
            
            if len(pil_images) == 3:
                resized_pil_channels = [img.resize(self.target_size, Image.BILINEAR) for img in pil_images]
                image_np_list = [np.array(img) for img in resized_pil_channels]
                image_np_stacked = np.stack(image_np_list, axis=-1)
                image_pil = Image.fromarray(image_np_stacked)
            elif len(pil_images) == 1: 
                 image_pil = pil_images[0].convert("RGB")
            elif pil_images: # Some bands found, but not 3 for RGB. Use first and convert.
                image_pil = pil_images[0].convert("RGB")
            else: # Should not happen if indexing logic is correct
                raise ValueError(f"Could not load bands for {entry['name_id']}. Sources: {entry['image_sources_location']}")
        else:
            raise ValueError(f"Invalid image_sources_type in __getitem__: {self.image_sources_type}")


        image_resized_pil = image_pil.resize(self.target_size, Image.BILINEAR)
        sample["image"] = np.array(image_resized_pil).astype(np.uint8)

        # --- Semantic Label Processing ---
        gt_pil = Image.open(entry["gt"]).convert("RGB")
        gt_resized_pil = gt_pil.resize(self.target_size, Image.NEAREST)
        gt_np_rgb = np.array(gt_resized_pil)
        
        height, width, _ = gt_np_rgb.shape
        semantics_map_hw = np.zeros((height, width), dtype=np.uint8)
        
        crop_mask = (gt_np_rgb[:, :, 1] > 100) & (gt_np_rgb[:, :, 0] < 128) & (gt_np_rgb[:, :, 2] < 128)
        semantics_map_hw[crop_mask] = 1
        
        weed_mask = (gt_np_rgb[:, :, 0] > 100) & (gt_np_rgb[:, :, 1] < 128) & (gt_np_rgb[:, :, 2] < 128)
        semantics_map_hw[weed_mask] = 2

        sample["semantics"] = semantics_map_hw.astype(np.uint8)
        sample["name"] = entry["name_id"]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample