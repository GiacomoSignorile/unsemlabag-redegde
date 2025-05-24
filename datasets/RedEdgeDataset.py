# File: datasets/RedEdgeDataset.py

import os
import random
import numpy as np
from PIL import Image
import tifffile
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from utils.transforms import Transforms

class RedEdgeDataModule(LightningDataModule):
    def __init__(self, full_cfg):
        super().__init__()
        print("===========================================")
        print("--- RedEdgeDataModule __init__ (Loading from Existing Patches) ---")
        self.full_cfg = full_cfg
        data_cfg = self.full_cfg.get("data", {})
        if not data_cfg: raise ValueError("'data' section not found.")
        print(f"  __init__: data_cfg content loaded: {data_cfg}")

        self.root_dir = data_cfg.get("root_dir")
        if self.root_dir is None: raise KeyError("'root_dir' not found.")

        self.train_val_fields_config = data_cfg.get("train_val_fields", [])
        self.test_fields_config = data_cfg.get("test_fields", [])
        self.validation_split_ratio = data_cfg.get("validation_split_ratio", 0.2)
        self.seed = self.full_cfg.get("experiment", {}).get("seed", 42)

        print(f"  __init__: Configured train_val_fields: {self.train_val_fields_config}")
        print(f"  __init__: Configured test_fields     : {self.test_fields_config}")

        # Parameters for finding the patches
        self.image_patch_folder_config = data_cfg.get("image_patch_folder", "RGB")
        self.gt_patch_folder_config = data_cfg.get("gt_patch_folder", "groundtruth")
        
        train_params_cfg = self.full_cfg.get("train", {})
        self.batch_size = train_params_cfg.get("batch_size", data_cfg.get("batch_size", 16)) # Defaulted to 16
        self.num_workers = train_params_cfg.get("workers", data_cfg.get("workers", 0))
        self.n_gpus = train_params_cfg.get("n_gpus", data_cfg.get("n_gpus", 1))

        self.dataset_init_args = {
            "root_dir": self.root_dir,
            "image_patch_folder": self.image_patch_folder_config,
            "gt_patch_folder": self.gt_patch_folder_config,
            "target_size": data_cfg.get("target_size", (512, 512)) # Should match actual patch size
        }
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self._has_setup_fit = False
        self._has_setup_test = False
        print("--- RedEdgeDataModule __init__ FINISHED ---")
        print("===========================================")

    def setup(self, stage: str = None):
        print("===========================================")
        print(f"--- RedEdgeDataModule setup(stage='{stage}') (Loading Existing Patches) ---")
        # ... (setup logic with random_split from your last working version) ...
        if stage == "fit" or stage is None:
            if not self._has_setup_fit:
                if self.train_val_fields_config:
                    print(f"  SETUP FIT: Initializing FULL dataset for splitting with fields: {self.train_val_fields_config}")
                    full_dataset_for_split = RedEdgeDataset(fields=self.train_val_fields_config, **self.dataset_init_args)
                    if len(full_dataset_for_split) > 0:
                        num_samples = len(full_dataset_for_split)
                        val_size = int(self.validation_split_ratio * num_samples)
                        if val_size == 0 and num_samples > 1 : val_size = 1 
                        if val_size >= num_samples and num_samples > 0 : val_size = num_samples -1 
                        train_size = num_samples - val_size
                        if train_size > 0 and val_size > 0:
                            print(f"  SETUP FIT: Splitting {num_samples} samples into train_size={train_size}, val_size={val_size}")
                            self.train_dataset, self.val_dataset = random_split(
                                full_dataset_for_split, [train_size, val_size],
                                generator=torch.Generator().manual_seed(self.seed)
                            )
                            print(f"  SETUP FIT: train_dataset INITIALIZED with {len(self.train_dataset)} items (from split).")
                            print(f"  SETUP FIT: val_dataset INITIALIZED with {len(self.val_dataset)} items (from split).")
                        elif train_size > 0:
                            print(f"  SETUP FIT: Using all {num_samples} samples for training.")
                            self.train_dataset = full_dataset_for_split; self.val_dataset = None
                        else:
                            print(f"  SETUP FIT: Not enough data to split."); self.train_dataset = None; self.val_dataset = None
                    else: print("  SETUP FIT: full_dataset_for_split IS EMPTY."); self.train_dataset = None; self.val_dataset = None
                else: print("  SETUP FIT: self.train_val_fields_config IS EMPTY."); self.train_dataset = None; self.val_dataset = None
                self._has_setup_fit = True
            else: print(f"  SETUP FIT: Already called for stage 'fit'. Skipping.")

        if stage == "test" or stage is None:
            if not self._has_setup_test:
                if self.test_fields_config:
                    print(f"  SETUP TEST: Initializing test_dataset with fields: {self.test_fields_config}")
                    self.test_dataset = RedEdgeDataset(fields=self.test_fields_config, **self.dataset_init_args)
                    print(f"  SETUP TEST: test_dataset items: {len(self.test_dataset) if self.test_dataset else 'None or 0'}")
                else: print("  SETUP TEST: self.test_fields_config IS EMPTY."); self.test_dataset = None
                self._has_setup_test = True
            else: print(f"  SETUP TEST: Already called for stage 'test'. Skipping.")
        print(f"--- RedEdgeDataModule setup(stage='{stage}') FINISHED ---")
        print(f"  After setup: train_dataset len: {len(self.train_dataset) if self.train_dataset else 'None'}")
        print(f"  After setup: val_dataset len: {len(self.val_dataset) if self.val_dataset else 'None'}")
        print("===========================================")
        
    def train_dataloader(self): 
        print("===========================================")
        print("--- RedEdgeDataModule train_dataloader() CALLED ---")
        if not self.train_dataset or len(self.train_dataset) == 0:
            print("  ERROR in train_dataloader: self.train_dataset is None or empty. " \
                  "This indicates an issue in the setup phase or config. Returning None.")
            return None
        return DataLoader(self.train_dataset, batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        print("===========================================")
        print("--- RedEdgeDataModule val_dataloader() CALLED ---")
        if not self.val_dataset or len(self.val_dataset) == 0:
            print("  WARN in val_dataloader: self.val_dataset is None or empty. Returning None.")
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self): 
        print("===========================================")
        print("--- RedEdgeDataModule test_dataloader() CALLED ---")
        if not self.test_dataset or len(self.test_dataset) == 0:
            print("  ERROR in test_dataloader: self.test_dataset is None or empty. Returning None.")
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)


class RedEdgeDataset(Dataset):
    def __init__(self, root_dir, fields, 
                 image_patch_folder, gt_patch_folder, 
                 target_size=(512,512)):
        super().__init__()
        self.root_dir = root_dir
        self.fields = fields
        self.image_patch_folder_name = image_patch_folder
        self.gt_patch_folder_name = gt_patch_folder     
        self.target_size = target_size 
        self.transform = Transforms()

        self.file_index = []
        print(f"--- Initializing RedEdgeDataset (Loading Existing Patches) for fields: {fields} ---")
        print(f"  Root dir: {os.path.abspath(root_dir)}")
        print(f"  Image patches expected in: ...field_id/{self.image_patch_folder_name}/<patch_filename>.png")
        print(f"  GT patches expected in:    ...field_id/{self.gt_patch_folder_name}/<patch_filename>.png")

        for field_id in self.fields:
            print(f"  Processing field for existing patches: {field_id}")
            
            current_image_patch_dir = os.path.join(self.root_dir, field_id, self.image_patch_folder_name)
            current_gt_patch_dir = os.path.join(self.root_dir, field_id, self.gt_patch_folder_name)

            if not os.path.isdir(current_image_patch_dir):
                print(f"    WARNING: Image patch directory not found: {current_image_patch_dir}")
                continue
            if not os.path.isdir(current_gt_patch_dir):
                print(f"    WARNING: GT patch directory not found: {current_gt_patch_dir}")
                continue
            
            for patch_filename in os.listdir(current_image_patch_dir):
                if not patch_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')): 
                    continue

                image_patch_full_path = os.path.join(current_image_patch_dir, patch_filename)
                
                gt_patch_full_path = os.path.join(current_gt_patch_dir, patch_filename) 
                
                if os.path.exists(gt_patch_full_path):
                    self.file_index.append({
                        "image_patch_path": image_patch_full_path,
                        "gt_patch_path": gt_patch_full_path,
                        "name_id": f"{field_id}/patch_{patch_filename}"
                    })
                else:
                    print(f"    WARNING: No corresponding GT patch found at '{gt_patch_full_path}' for image patch: {image_patch_full_path}")
        
        if not self.file_index:
            print(f"FINAL WARNING (Existing Patches): No patch pairs were found for fields {fields}.")
        else:
            print(f"--- RedEdgeDataset (Existing Patches) Initialized for fields {fields}. Found {len(self.file_index)} patch pairs. ---")

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, index):
        entry = self.file_index[index]
        sample = {}

        # Load image patch
        image_path = entry["image_patch_path"]
        if image_path.lower().endswith(('.tif', '.tiff')):
            img_data_np = tifffile.imread(image_path)
            if img_data_np.ndim == 2: img_data_np = np.stack([img_data_np]*3, axis=-1) # Ensure 3 channels for RGB
            elif img_data_np.ndim == 3 and img_data_np.shape[2] > 3: img_data_np = img_data_np[:,:,:3] # Take first 3 if more
            if img_data_np.dtype != np.uint8:
                if img_data_np.max() <= 1.0 : img_data_np = (img_data_np * 255)
                elif img_data_np.max() > 255: img_data_np = (img_data_np / img_data_np.max() * 255)
                img_data_np = img_data_np.astype(np.uint8)
            image_pil = Image.fromarray(img_data_np)
        else:
            image_pil = Image.open(image_path).convert("RGB")

        if image_pil.size != self.target_size:
            print(f"Resizing image patch {entry['name_id']} from {image_pil.size} to {self.target_size}")
            image_pil = image_pil.resize(self.target_size, Image.BILINEAR)
        sample["image"] = np.array(image_pil).astype(np.uint8)

        # Load GT patch (logic remains same as before)
        gt_patch_path = entry["gt_patch_path"]
        gt_np_rgb = None
        if gt_patch_path.lower().endswith(('.tif', '.tiff')):
            tiff_img_data = tifffile.imread(gt_patch_path)
            if tiff_img_data.ndim == 2: gt_np_rgb = np.stack([tiff_img_data]*3, axis=-1)
            elif tiff_img_data.ndim == 3 and tiff_img_data.shape[2] >= 3: gt_np_rgb = tiff_img_data[:, :, :3]
            else: raise ValueError(f"Unsupported TIFF GT patch: {gt_patch_path}, shape: {tiff_img_data.shape}")
            if gt_np_rgb.dtype != np.uint8:
                if gt_np_rgb.max() <= 1.0: gt_np_rgb = (gt_np_rgb * 255)
                elif gt_np_rgb.max() > 255 : gt_np_rgb = (gt_np_rgb / gt_np_rgb.max() * 255)
                gt_np_rgb = gt_np_rgb.astype(np.uint8)
            gt_pil_from_tiff = Image.fromarray(gt_np_rgb)
            if gt_pil_from_tiff.size != self.target_size: gt_pil_from_tiff = gt_pil_from_tiff.resize(self.target_size, Image.NEAREST)
            gt_np_rgb = np.array(gt_pil_from_tiff)
        else:
            gt_pil = Image.open(gt_patch_path).convert("RGB")
            if gt_pil.size != self.target_size: gt_pil = gt_pil.resize(self.target_size, Image.NEAREST)
            gt_np_rgb = np.array(gt_pil)
        
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