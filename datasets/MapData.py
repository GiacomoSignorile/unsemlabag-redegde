import glob
import os
import random

import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision # For potential use if directly adapting RoWeeder's GT loading

# Assuming Transforms class is defined elsewhere as in Unsemlabag
# from utils.transforms import Transforms
# For now, let's make a placeholder if it's not available
class Transforms:
    def __call__(self, sample):
        # Placeholder: Add actual transformations here (e.g., ToTensor, augmentations)
        # For now, just ensure image is HWC and then CHW for PyTorch
        if "image" in sample and isinstance(sample["image"], np.ndarray):
            sample["image"] = np.transpose(sample["image"], (2, 0, 1)) # HWC to CHW
        return sample


class MapData(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_dir = cfg["data"]["root_dir"]
        self.fields_to_load = cfg["data"]["fields"] # e.g., ["000", "001"]
        self.channels_to_load = cfg["data"].get("channels", ["R", "G", "B"]) # Default to RGB
        self.gt_folder_name = cfg["data"].get("gt_folder_name", "groundtruth") # usually "groundtruth"

        # For validation, you might want to use a different set of fields or actual GT
        self.val_fields = cfg["data"].get("val_fields", self.fields_to_load)
        self.val_gt_folder_name = cfg["data"].get("val_gt_folder_name", self.gt_folder_name)


        self.train_dataset = None
        self.val_dataset = None
        # self.setup() # Call setup explicitly or let PL call it

    def setup(self, stage=None):
        # Called on every GPU
        if stage == "fit" or stage is None:
            self.train_dataset = WeedMapStyleData(
                root_dir=self.root_dir,
                fields=self.fields_to_load,
                image_channels=self.channels_to_load,
                gt_folder_name=self.gt_folder_name,
                # Pass transform_cfg if Unsemlabag's Transforms takes config
            )
            self.val_dataset = WeedMapStyleData(
                root_dir=self.root_dir,
                fields=self.val_fields, # Use validation fields
                image_channels=self.channels_to_load,
                gt_folder_name=self.val_gt_folder_name, # Use validation GT folder
                # Pass transform_cfg if Unsemlabag's Transforms takes config
            )
        # Add test_dataset setup if needed for a "test" stage

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg["train"]["batch_size"] // self.cfg["train"].get("n_gpus", 1),
            num_workers=self.cfg["train"]["workers"],
            pin_memory=True,
            shuffle=True,
        )
        # self.len = len(self.train_dataset) # PL handles this
        return loader

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup(stage="fit") # or a specific validation setup
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg["train"]["batch_size"] // self.cfg["train"].get("n_gpus", 1),
            num_workers=self.cfg["train"]["workers"],
            pin_memory=True,
            shuffle=False,
        )
        # self.len = len(self.val_dataset) # PL handles this
        return loader

    def test_dataloader(self):
        # If you want to test on WeedMap GTs (not Unsemlabag's generated labels)
        # you can implement this similar to val_dataloader with test_fields
        if self.cfg["data"].get("test_fields"):
            test_dataset = WeedMapStyleData(
                root_dir=self.root_dir,
                fields=self.cfg["data"]["test_fields"],
                image_channels=self.channels_to_load,
                gt_folder_name=self.gt_folder_name,
            )
            return DataLoader(
                test_dataset,
                batch_size=self.cfg["train"]["batch_size"] // self.cfg["train"].get("n_gpus", 1),
                num_workers=self.cfg["train"]["workers"],
                pin_memory=True,
                shuffle=False,
            )
        return None # Original Unsemlabag doesn't test on generated labels


class WeedMapStyleData(Dataset):
    def __init__(self, root_dir, fields, image_channels=["R", "G", "B"], gt_folder_name="groundtruth", transform_cfg=None):
        super().__init__()
        self.root_dir = root_dir
        self.fields = fields
        self.image_channels = image_channels # e.g., ["R", "G", "B"]
        self.gt_folder_name = gt_folder_name
        self.transform = Transforms() # Or initialize with transform_cfg if needed

        self.file_index = []
        for field_name in self.fields:
            field_gt_path = os.path.join(self.root_dir, field_name, self.gt_folder_name)
            if not os.path.isdir(field_gt_path):
                print(f"Warning: Ground truth path not found for field {field_name}: {field_gt_path}")
                continue
            
            # Check if GT folder contains another "groundtruth" subfolder (common mistake)
            # RoWeeder's original code had a check for this:
            # if os.path.isdir(os.path.join(v, os.listdir(v)[0])):
            #     self.gt_folders[k] = os.path.join(v, "groundtruth")
            # Simplified: assume field_gt_path is the correct direct parent of GT images
            
            for filename in os.listdir(field_gt_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    # Construct paths for each image channel
                    img_channel_paths = {}
                    valid_entry = True
                    for ch_name in self.image_channels:
                        ch_path = os.path.join(self.root_dir, field_name, ch_name, filename)
                        if not os.path.exists(ch_path):
                            # print(f"Warning: Image channel {ch_name} not found for {field_name}/{filename}")
                            valid_entry = False
                            break
                        img_channel_paths[ch_name] = ch_path
                    
                    if valid_entry:
                        gt_path = os.path.join(field_gt_path, filename)
                        self.file_index.append({
                            "image_channels": img_channel_paths,
                            "gt": gt_path,
                            "name": f"{field_name}/{filename}"
                        })
        
        if not self.file_index:
            raise ValueError(f"No files found for fields {self.fields} in {self.root_dir} with structure 'field/channel/file' and 'field/{gt_folder_name}/file'. Check paths and config.")

        self.size = None # To match Unsemlabag's original option, though usually not needed
        self.real_size = len(self.file_index)

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        # index = index if self.size is None else random.randint(0, self.real_size - 1) # If self.size is used
        entry = self.file_index[index]
        sample = {}

        # --- Image Loading & Normalization (Unsemlabag style) ---
        # Load specified channels and stack them
        # We assume the target is RGB for Unsemlabag
        
        # For Unsemlabag, it expects a 3-channel RGB image.
        # We will load R, G, B specified in self.image_channels
        # If other channels are specified, this part needs more logic (e.g. how to map to 3 channels)
        
        pil_images = []
        # Ensure R, G, B are loaded in that order if present for standard RGB
        # This assumes self.image_channels is something like ["R", "G", "B"] or a subset in order
        for ch_key in ["R", "G", "B"]: # Prioritize R, G, B for a standard RGB image
            if ch_key in entry["image_channels"]:
                pil_images.append(Image.open(entry["image_channels"][ch_key]).convert("L")) # Load as grayscale
        
        if len(pil_images) != 3 and len(self.image_channels) == 3:
             # Fallback if R,G,B not explicitly found but 3 channels are expected
            pil_images = []
            for ch_name in self.image_channels: # Load in the specified order
                 pil_images.append(Image.open(entry["image_channels"][ch_name]).convert("L"))


        if not pil_images:
            raise ValueError(f"Could not load R,G,B channels for {entry['name']}. Found channels: {list(entry['image_channels'].keys())}")
        
        # Resize all to 1024x1024 (Unsemlabag's default)
        target_size = (1024, 1024)
        resized_channels = [img.resize(target_size, Image.BILINEAR) for img in pil_images]
        
        # Stack to form a 3-channel image (H, W, C)
        image_np_list = [np.array(img) for img in resized_channels]
        if len(image_np_list) == 1: # If only one channel was loaded (e.g. grayscale)
            image = np.stack([image_np_list[0]]*3, axis=-1) # Repeat to make it 3-channel
        elif len(image_np_list) == 3:
            image = np.stack(image_np_list, axis=-1)
        else:
            raise ValueError(f"Expected 1 or 3 channels to form an image, got {len(image_np_list)} for {entry['name']}")


        # Normalize each channel (Unsemlabag style)
        norm_channels = []
        for i in range(image.shape[2]):
            ch_data = image[:, :, i].astype(np.float32)
            ch_norm = (ch_data - ch_data.mean()) / (ch_data.std() + 1e-17)
            norm_channels.append(ch_norm[:, :, np.newaxis])
        
        sample["image"] = np.concatenate(norm_channels, axis=-1) # Shape (H, W, C)

        # --- Semantic Label Processing ---
        # WeedMap GTs are typically RGB: R=weed, G=crop, Black/Blue=soil/background
        # Unsemlabag expects: 0=soil, 1=crop, 2=weed
        # The "unknown" class (3) in Unsemlabag is for its *generated* pseudo-labels.
        # When loading true GTs, we should map directly to 0, 1, 2.

        gt_pil = Image.open(entry["gt"]).convert("RGB").resize(target_size, Image.NEAREST) # Use NEAREST for labels
        gt_np = np.array(gt_pil)
        
        height, width, _ = gt_np.shape
        semantics_map = np.zeros((height, width), dtype=np.uint8)  # Default to soil (class 0)
        
        # Crop (Green pixels)
        # Check for reasonably pure green: G high, R and B low
        crop_mask = (gt_np[:, :, 1] > 128) & (gt_np[:, :, 0] < 100) & (gt_np[:, :, 2] < 100)
        semantics_map[crop_mask] = 1 # Class 1 for crop
        
        # Weed (Red pixels)
        # Check for reasonably pure red: R high, G and B low
        weed_mask = (gt_np[:, :, 0] > 128) & (gt_np[:, :, 1] < 100) & (gt_np[:, :, 2] < 100)
        semantics_map[weed_mask] = 2 # Class 2 for weed

        sample["semantics"] = semantics_map.astype(np.uint8)
        sample["name"] = entry["name"]

        if self.transform is not None:
            sample = self.transform(sample) # Expects HWC, will convert to CHW if needed

        return sample