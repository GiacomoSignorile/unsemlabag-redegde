import os
import random
import numpy as np
from PIL import Image
import tifffile # Ensure this is imported
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from utils.transforms import Transforms


class RedEdgeDataModule(LightningDataModule):
    def __init__(self, full_cfg):
        super().__init__()
        self.full_cfg = full_cfg
        data_cfg = self.full_cfg.get("data", {})
        if not data_cfg: raise ValueError("'data' section not found.")
        self.root_dir = data_cfg.get("root_dir") # This will be "./samples/RedEdge_Patches_224"
        if self.root_dir is None: raise KeyError("'root_dir' not found.")

        self.train_fields = data_cfg.get("train_fields", [])
        self.val_fields = data_cfg.get("val_fields", self.train_fields)
        self.test_fields = data_cfg.get("test_fields", []) # Explicitly get test_fields

        # New params from config for patched data
        self.patched_image_subfolder_template = data_cfg.get("patched_image_subfolder_template", "images/{composite_rgb_name}")
        self.composite_rgb_name = data_cfg.get("composite_rgb_name", "RGB")
        self.patched_gt_subfolder_parent = data_cfg.get("patched_gt_subfolder_parent", "semantics")
        
        train_params_cfg = self.full_cfg.get("train", {})
        self.batch_size = train_params_cfg.get("batch_size", data_cfg.get("batch_size", 4))
        self.num_workers = train_params_cfg.get("workers", data_cfg.get("workers", 0))
        self.n_gpus = train_params_cfg.get("n_gpus", data_cfg.get("n_gpus", 1))
        
        self.dataset_init_args = {
            "root_dir": self.root_dir,
            "patched_image_subfolder_template": self.patched_image_subfolder_template,
            "composite_rgb_name": self.composite_rgb_name,
            "patched_gt_subfolder_parent": self.patched_gt_subfolder_parent,
            "target_size": data_cfg.get("target_size", (224, 224)) # Should match patch size
        }
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.train_fields:
                self.train_dataset = RedEdgeDataset(fields=self.train_fields, **self.dataset_init_args)
            if self.val_fields:
                self.val_dataset = RedEdgeDataset(fields=self.val_fields, **self.dataset_init_args) # No is_train for Unsemlabag Transforms
        if stage == "test" or stage is None:
            if self.test_fields:
                 self.test_dataset = RedEdgeDataset(fields=self.test_fields, **self.dataset_init_args)
    
    # Dataloaders remain the same
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
    def __init__(self, root_dir, fields, 
                 patched_image_subfolder_template, composite_rgb_name, 
                 patched_gt_subfolder_parent, 
                 target_size=(224,224)): # target_size is now mainly for reference
        super().__init__()
        self.root_dir = root_dir
        self.fields = fields
        self.patched_image_subfolder = patched_image_subfolder_template.format(composite_rgb_name=composite_rgb_name)
        self.patched_gt_subfolder_parent = patched_gt_subfolder_parent
        self.target_size = target_size # Patches should already be this size
        self.transform = Transforms()

        self.file_index = []
        print(f"--- Initializing RedEdgeDataset (Patched) ---")
        print(f"Root dir: {os.path.abspath(root_dir)}")
        print(f"Fields to load patches from: {fields}")
        print(f"Image patches expected in: ...field_id/{self.patched_image_subfolder}/<patch_name>.png")
        print(f"GT patches expected in:    ...field_id/{self.patched_gt_subfolder_parent}/<original_gt_basename>/<patch_name>.png")

        for field_id in self.fields:
            print(f"Processing field for patches: {field_id}")
            
            current_image_patch_dir = os.path.join(self.root_dir, field_id, self.patched_image_subfolder)
            current_gt_patch_parent_dir = os.path.join(self.root_dir, field_id, self.patched_gt_subfolder_parent)

            if not os.path.isdir(current_image_patch_dir):
                print(f"  WARNING: Image patch directory not found: {current_image_patch_dir}")
                continue
            if not os.path.isdir(current_gt_patch_parent_dir):
                print(f"  WARNING: GT patch parent directory not found: {current_gt_patch_parent_dir}")
                continue

            # Find subfolders in GT patch parent dir (these are named after original GT files)
            # e.g., 'first000_gt', 'another_gt_basename'
            gt_patch_subfolders = [d for d in os.listdir(current_gt_patch_parent_dir) if os.path.isdir(os.path.join(current_gt_patch_parent_dir, d))]
            if not gt_patch_subfolders:
                print(f"  WARNING: No GT patch subfolders found in {current_gt_patch_parent_dir}")
                continue
            
            # For simplicity, assume we use all GTs if multiple, or just one if it's always one-to-one
            # With current run_preprocessing.py, there's one GT subfolder per original GT file.
            # If an RGB.png corresponds to ONE first000_gt.png, then we just need to find its patch folder.
            # Let's assume the first GT subfolder found is the one corresponding to the images in patched_image_subfolder
            # This assumption holds if your run_preprocessing.py creates RGB/ and first000_gt/ (for example)
            
            # A more robust way: if you know the original GT filename associated with the RGB patches
            # For now, let's try to be a bit more general if there's only one GT subfolder.
            # If there are multiple, this logic might need to be smarter or the run_preprocessing.py
            # should ensure a clear mapping.
            
            # Let's iterate through image patches and try to find a corresponding GT patch
            for image_patch_filename in os.listdir(current_image_patch_dir):
                if not image_patch_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_patch_full_path = os.path.join(current_image_patch_dir, image_patch_filename)
                found_corresponding_gt = False
                
                for gt_subfolder_name in gt_patch_subfolders: # e.g., "first000_gt"
                    gt_patch_full_path = os.path.join(current_gt_patch_parent_dir, gt_subfolder_name, image_patch_filename) # Use same patch filename
                    
                    if os.path.exists(gt_patch_full_path):
                        self.file_index.append({
                            "image_patch_path": image_patch_full_path,
                            "gt_patch_path": gt_patch_full_path,
                            "name_id": f"{field_id}/patch_{image_patch_filename}"
                        })
                        found_corresponding_gt = True
                        break # Found corresponding GT patch for this image patch
                
                if not found_corresponding_gt:
                    print(f"  WARNING: No corresponding GT patch found for image patch: {image_patch_full_path} in any of {gt_patch_subfolders}")
        
        if not self.file_index:
            print(f"FINAL WARNING (Patched): No patch pairs were added to the index.")
        else:
            print(f"--- RedEdgeDataset (Patched) Initialized. Found {len(self.file_index)} patch pairs. ---")

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, index):
        entry = self.file_index[index]
        sample = {}

        # Load image patch (should already be target_size and RGB)
        image_pil = Image.open(entry["image_patch_path"]).convert("RGB")
        # Patches are already 224x224, so resize might be redundant but harmless if target_size is also 224x224
        if image_pil.size != self.target_size:
            image_pil = image_pil.resize(self.target_size, Image.BILINEAR)
        sample["image"] = np.array(image_pil).astype(np.uint8) # For Unsemlabag's Transforms

        # Load GT patch
        gt_patch_path = entry["gt_patch_path"]
        gt_np_rgb = None
        if gt_patch_path.lower().endswith(('.tif', '.tiff')):
            tiff_img_data = tifffile.imread(gt_patch_path)
            if tiff_img_data.ndim == 2: gt_np_rgb = np.stack([tiff_img_data]*3, axis=-1)
            elif tiff_img_data.ndim == 3 and tiff_img_data.shape[2] >= 3: gt_np_rgb = tiff_img_data[:, :, :3]
            else: raise ValueError(f"Unsupported TIFF GT patch: {gt_patch_path}")
            if gt_np_rgb.dtype != np.uint8: # Scale if necessary
                if gt_np_rgb.max() <= 1.0 and gt_np_rgb.min() >= 0.0: gt_np_rgb = (gt_np_rgb * 255)
                elif gt_np_rgb.max() > 255 : gt_np_rgb = (gt_np_rgb / gt_np_rgb.max() * 255)
                gt_np_rgb = gt_np_rgb.astype(np.uint8)
            gt_pil_from_tiff = Image.fromarray(gt_np_rgb) # Now HWC uint8
            if gt_pil_from_tiff.size != self.target_size: # Resize if patch wasn't exact (should be)
                 gt_pil_from_tiff = gt_pil_from_tiff.resize(self.target_size, Image.NEAREST)
            gt_np_rgb = np.array(gt_pil_from_tiff)
        else:
            gt_pil = Image.open(gt_patch_path).convert("RGB")
            if gt_pil.size != self.target_size: # Resize if patch wasn't exact
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