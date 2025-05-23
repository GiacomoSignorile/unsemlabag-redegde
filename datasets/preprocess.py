import os
import torch
import torchvision
import numpy as np
import tifffile as tiff # Make sure this is installed in your Docker env
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# rotate_ortho and crop_black_borders can remain here if you might use them later
# For now, we'll focus on the patching function.

def divide_single_file_into_patches(input_file_path, base_output_folder, output_subfolder_name, patch_size):
    """
    Divides a single input image file into patches and saves them.

    Args:
        input_file_path (str): Path to the single image file to be patched.
        base_output_folder (str): The base directory where the output_subfolder_name will be created.
                                  (e.g., ./samples/RedEdge_Patches/000/images)
        output_subfolder_name (str): Name of the subfolder to create within base_output_folder
                                     to store the patches (e.g., "RGB" or "first000_gt").
        patch_size (int): The size (width and height) of the patches.
    """
    full_output_path_for_patches = os.path.join(base_output_folder, output_subfolder_name)
    os.makedirs(full_output_path_for_patches, exist_ok=True)
    
    try:
        # Handle TIFF and other image types separately for opening
        if input_file_path.lower().endswith(('.tif', '.tiff')):
            img_data_np = tiff.imread(input_file_path)
            # Ensure img_data_np is HWC if it's multi-channel, or HW if single
            # tifffile often returns HWC for RGB, or HW for grayscale
            if img_data_np.ndim == 2: # Grayscale
                img_data_np = np.expand_dims(img_data_np, axis=-1) # HW -> HWC (C=1)
            # Ensure it's uint8 for PILToTensor if it's something like int16/32 or float
            if img_data_np.dtype != np.uint8:
                if img_data_np.max() > 255 or img_data_np.min() < 0: # Needs scaling
                    img_data_np = (img_data_np - img_data_np.min()) / (img_data_np.max() - img_data_np.min() + 1e-6) * 255
                img_data_np = img_data_np.astype(np.uint8)
            img_pil = Image.fromarray(img_data_np.squeeze()) # Squeeze if C=1 for L mode, else from HWC
        else:
            img_pil = Image.open(input_file_path)
    except Exception as e:
        print(f"Error opening or processing {input_file_path}: {e}")
        return

    # PILToTensor expects a PIL Image. It converts HWC [0-255] uint8 to CHW [0-1] float
    # or L [0-255] uint8 to 1HW [0-1] float.
    # Let's convert to tensor and ensure it's CHW
    try:
        tensor = torchvision.transforms.PILToTensor()(img_pil) # This gives CHW uint8 tensor
    except Exception as e:
        print(f"Error converting {input_file_path} to tensor with PILToTensor: {e}")
        # Fallback if PILToTensor fails (e.g. if img_pil was already numpy due to tiff)
        if isinstance(img_pil, np.ndarray): # Should not happen if above logic is correct
            if img_pil.ndim == 2: img_pil = np.expand_dims(img_pil, axis=0) # H,W -> 1,H,W
            elif img_pil.ndim == 3: img_pil = img_pil.transpose(2,0,1) # H,W,C -> C,H,W
            tensor = torch.from_numpy(img_pil)
        else:
            return


    if tensor.ndim == 2: # Should be CHW from PILToTensor, but defensive
        tensor = tensor.unsqueeze(0) 
    
    C, H, W = tensor.shape

    right_padding = (patch_size - W % patch_size) % patch_size
    bottom_padding = (patch_size - H % patch_size) % patch_size

    tensor_padded = torch.nn.functional.pad(
        tensor, (0, right_padding, 0, bottom_padding), mode="constant", value=0
    )

    # Unfold expects float input
    unfolded_patches = torch.nn.functional.unfold(
        tensor_padded.float(), 
        kernel_size=patch_size,
        stride=patch_size,
    )
    
    # Cast back to original tensor dtype after unfold (which used float)
    unfolded_patches = unfolded_patches.type(tensor.dtype)

    patches_rearranged = rearrange(unfolded_patches, "(c p1 p2) n -> n c p1 p2", c=C, p1=patch_size, p2=patch_size)
    
    print(f"Saving {patches_rearranged.shape[0]} patches for {os.path.basename(input_file_path)} into {output_subfolder_name}...")

    for i, patch_tensor in enumerate(tqdm(patches_rearranged, desc=f"Saving {output_subfolder_name} patches")):
        # Convert CHW tensor patch to PIL Image for saving
        # ToPILImage expects CHW tensor
        patch_pil_to_save = torchvision.transforms.ToPILImage()(patch_tensor)
        patch_pil_to_save.save(os.path.join(full_output_path_for_patches, f"{i}.png"))
        
    print(f"Patches for {os.path.basename(input_file_path)} saved to: {full_output_path_for_patches}")

# You can keep rotate_ortho and crop_black_borders here if you want
# def rotate_ortho(...): ...
# def crop_black_borders(...): ...