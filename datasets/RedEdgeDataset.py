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
        self.full_cfg = full_cfg
        data_cfg = self.full_cfg.get("data", {})
        if not data_cfg:
            raise ValueError("'data' section not found.")

        self.root_dir = data_cfg.get("root_dir")
        if self.root_dir is None:
            raise KeyError("'root_dir' not found.")

        self.train_val_fields_config = data_cfg.get("train_val_fields", [])
        self.test_fields_config = data_cfg.get("test_fields", [])
        self.validation_split_ratio = data_cfg.get("validation_split_ratio", 0.2)
        self.seed = self.full_cfg.get("experiment", {}).get("seed", 42)

        self.image_patch_folder_config = data_cfg.get("image_patch_folder", "RGB")
        self.gt_patch_folder_config = data_cfg.get("gt_patch_folder", "groundtruth")

        train_params_cfg = self.full_cfg.get("train", {})
        self.batch_size = train_params_cfg.get("batch_size", data_cfg.get("batch_size", 16))
        self.num_workers = train_params_cfg.get("workers", data_cfg.get("workers", 0))
        self.n_gpus = train_params_cfg.get("n_gpus", data_cfg.get("n_gpus", 1))

        self.dataset_init_args = {
            "root_dir": self.root_dir,
            "image_patch_folder": self.image_patch_folder_config,
            "gt_patch_folder": self.gt_patch_folder_config,
            "target_size": data_cfg.get("target_size", (512, 512))
        }
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self._has_setup_fit = False
        self._has_setup_test = False

    def setup(self, stage: str = None):
        if stage in ("fit", None) and not self._has_setup_fit:
            if self.train_val_fields_config:
                full_dataset_for_split = RedEdgeDataset(fields=self.train_val_fields_config, **self.dataset_init_args)
                if len(full_dataset_for_split) > 0:
                    num_samples = len(full_dataset_for_split)
                    val_size = int(self.validation_split_ratio * num_samples)
                    if val_size == 0 and num_samples > 1:
                        val_size = 1
                    if val_size >= num_samples and num_samples > 0:
                        val_size = num_samples - 1
                    train_size = num_samples - val_size
                    if train_size > 0 and val_size > 0:
                        self.train_dataset, self.val_dataset = random_split(
                            full_dataset_for_split, [train_size, val_size],
                            generator=torch.Generator().manual_seed(self.seed)
                        )
                    elif train_size > 0:
                        self.train_dataset = full_dataset_for_split
                        self.val_dataset = None
                    else:
                        self.train_dataset = None
                        self.val_dataset = None
                else:
                    self.train_dataset = None
                    self.val_dataset = None
            else:
                self.train_dataset = None
                self.val_dataset = None
            self._has_setup_fit = True

        if stage in ("test", None) and not self._has_setup_test:
            if self.test_fields_config:
                self.test_dataset = RedEdgeDataset(fields=self.test_fields_config, **self.dataset_init_args)
            else:
                self.test_dataset = None
            self._has_setup_test = True

    def train_dataloader(self): 
        if not self.train_dataset or len(self.train_dataset) == 0:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        if not self.val_dataset or len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self): 
        if not self.test_dataset or len(self.test_dataset) == 0:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size // self.n_gpus if self.n_gpus > 0 else self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )


class RedEdgeDataset(Dataset):
    def __init__(self, root_dir, fields, image_patch_folder, gt_patch_folder, target_size=(512,512)):
        super().__init__()
        self.root_dir = root_dir
        self.fields = fields
        self.image_patch_folder_name = image_patch_folder
        self.gt_patch_folder_name = gt_patch_folder     
        self.target_size = target_size 
        self.transform = Transforms()
        self.file_index = []

        for field_id in self.fields:
            current_image_patch_dir = os.path.join(self.root_dir, field_id, self.image_patch_folder_name)
            current_gt_patch_dir = os.path.join(self.root_dir, field_id, self.gt_patch_folder_name)

            if not os.path.isdir(current_image_patch_dir) or not os.path.isdir(current_gt_patch_dir):
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

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, index):
        entry = self.file_index[index]
        sample = {}

        image_path = entry["image_patch_path"]
        if image_path.lower().endswith(('.tif', '.tiff')):
            img_data_np = tifffile.imread(image_path)
            if img_data_np.ndim == 2:
                img_data_np = np.stack([img_data_np] * 3, axis=-1)
            elif img_data_np.ndim == 3 and img_data_np.shape[2] > 3:
                img_data_np = img_data_np[:, :, :3]
            if img_data_np.dtype != np.uint8:
                if img_data_np.max() <= 1.0:
                    img_data_np = img_data_np * 255
                elif img_data_np.max() > 255:
                    img_data_np = img_data_np / img_data_np.max() * 255
                img_data_np = img_data_np.astype(np.uint8)
            image_pil = Image.fromarray(img_data_np)
        else:
            image_pil = Image.open(image_path).convert("RGB")

        if image_pil.size != self.target_size:
            image_pil = image_pil.resize(self.target_size, Image.BILINEAR)
        sample["image"] = np.array(image_pil).astype(np.uint8)

        gt_patch_path = entry["gt_patch_path"]
        if gt_patch_path.lower().endswith(('.tif', '.tiff')):
            tiff_img_data = tifffile.imread(gt_patch_path)
            if tiff_img_data.ndim == 2:
                gt_np_rgb = np.stack([tiff_img_data] * 3, axis=-1)
            elif tiff_img_data.ndim == 3 and tiff_img_data.shape[2] >= 3:
                gt_np_rgb = tiff_img_data[:, :, :3]
            else:
                raise ValueError(f"Unsupported TIFF GT patch: {gt_patch_path}, shape: {tiff_img_data.shape}")
            if gt_np_rgb.dtype != np.uint8:
                if gt_np_rgb.max() <= 1.0:
                    gt_np_rgb = gt_np_rgb * 255
                elif gt_np_rgb.max() > 255:
                    gt_np_rgb = gt_np_rgb / gt_np_rgb.max() * 255
                gt_np_rgb = gt_np_rgb.astype(np.uint8)
            gt_pil = Image.fromarray(gt_np_rgb)
            if gt_pil.size != self.target_size:
                gt_pil = gt_pil.resize(self.target_size, Image.NEAREST)
            gt_np_rgb = np.array(gt_pil)
        else:
            gt_pil = Image.open(gt_patch_path).convert("RGB")
            if gt_pil.size != self.target_size:
                gt_pil = gt_pil.resize(self.target_size, Image.NEAREST)
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