# File: run_all_fields_unsupervised.py

import os
import yaml
import numpy as np
from PIL import Image
import subprocess # To call other python scripts
import shutil

# --- Configuration for all fields ---
# This data would ideally come from a metadata file or be more structured
# For now, we define it here.
# IMPORTANT: Replace with your ACTUAL paths and dimensions!
FIELDS_DATA = {
    "000": {
        "ortho_rgb_path": "./samples/rotated_ortho2/000/RGB.png", # Path to the full RGB orthomosaic
        "dimensions_wh_px": [6846, 2802] # [width_pixels, height_pixels]
    },
    "001": {
        "ortho_rgb_path": "./samples/rotated_ortho2/001/RGB.png", # UPDATE THIS PATH
        "dimensions_wh_px": [6646, 2282] # UPDATE THIS with actual dimensions of 001/RGB.png
    },
    "002": {
        "ortho_rgb_path": "./samples/rotated_ortho2/002/RGB.png", # UPDATE THIS PATH
        "dimensions_wh_px": [6945, 3699] # UPDATE THIS
    },
    "003": {
        "ortho_rgb_path": "./samples/rotated_ortho2/003/RGB.png", # UPDATE THIS PATH
        "dimensions_wh_px": [6835, 2310] # UPDATE THIS
    },
    "004": {
        "ortho_rgb_path": "./samples/rotated_ortho2/004/RGB.png", # UPDATE THIS PATH
        "dimensions_wh_px": [4692, 3169] # UPDATE THIS
    }
}

BASE_CONFIG_PATH = "./config/config.yaml" # Path to your main template config
TEMP_CONFIG_PATH = "./config/temp_processing_config.yaml" # Temporary config for each run

# --- Output Directories ---
BASE_PSEUDO_LABEL_MAP_DIR = "./results/pseudo_labels_all_fields"
BASE_PATCHED_DATA_DIR = "./samples/RedEdge_Pseudo_Patches_All_Fields_512" # Or _224 if patch_size is 224

# --- Common Parameters (can be overridden by base_config.yaml if structured differently) ---
GSD = 0.001 # meters/pixel - assuming this is constant for all your rotated_ortho2 images
SENSOR_RESOLUTION_PATCH_SIZE = [512, 512] # [width, height] - desired patch size
                                       # This is also used as sensor.resolution for main.py & map_to_dataset.py
SENSOR_ANGLE = [45, 45] # degrees

# Hardcoded step sizes from Unsemlabag's generate_poses
GENERATE_POSES_X_STEP_M = 1.024
GENERATE_POSES_Y_STEP_M = 0.512
GENERATE_POSES_TILE_DIM_M = 1.024 # The FOV size generate_poses is tiling for


def calculate_mapper_poses(ortho_width_px, ortho_height_px, gsd, x_step_m, y_step_m, tile_dim_m):
    ortho_width_m = ortho_width_px * gsd
    ortho_height_m = ortho_height_px * gsd

    if ortho_width_m >= tile_dim_m:
        num_cols = int(np.floor((ortho_width_m - tile_dim_m) / x_step_m)) + 1
    else:
        num_cols = 1
    
    if ortho_height_m >= tile_dim_m:
        num_rows = int(np.floor((ortho_height_m - tile_dim_m) / y_step_m)) + 1
    else:
        num_rows = 1
    return [num_cols, num_rows]


def run_command(command_list):
    print(f"\nExecuting: {' '.join(command_list)}")
    process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stdout:
        print(f"STDOUT:\n{stdout.decode()}")
    if stderr:
        print(f"STDERR:\n{stderr.decode()}")
    if process.returncode != 0:
        print(f"ERROR: Command failed with exit code {process.returncode}")
        raise Exception(f"Command failed: {' '.join(command_list)}")
    print("Command executed successfully.")


def main():
    # Load the base configuration
    if not os.path.exists(BASE_CONFIG_PATH):
        print(f"Base config file not found at {BASE_CONFIG_PATH}. Please create it.")
        return
    with open(BASE_CONFIG_PATH, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(BASE_PSEUDO_LABEL_MAP_DIR, exist_ok=True)
    os.makedirs(BASE_PATCHED_DATA_DIR, exist_ok=True)

    for field_id, field_data in FIELDS_DATA.items():
        print(f"\n========================================================")
        print(f"PROCESSING FIELD: {field_id}")
        print(f"========================================================")

        current_ortho_rgb_path = field_data["ortho_rgb_path"]
        current_ortho_dims_wh_px = field_data["dimensions_wh_px"]

        if not os.path.exists(current_ortho_rgb_path):
            print(f"WARNING: Orthomosaic for field {field_id} not found at {current_ortho_rgb_path}. Skipping.")
            continue
        
        # --- 1. Prepare config and run main.py (Label Generation) ---
        print(f"\n--- Stage 1: Generating Pseudo-Label Map for Field {field_id} ---")
        cfg_for_label_gen = base_cfg.copy() # Start with a copy of the base config

        # Override paths and dimensions for the current field
        cfg_for_label_gen["data"]["maps"] = current_ortho_rgb_path
        current_pseudo_map_path = os.path.join(BASE_PSEUDO_LABEL_MAP_DIR, f"{field_id}_pseudo_map.png")
        cfg_for_label_gen["data"]["map_out_name"] = current_pseudo_map_path
        
        cfg_for_label_gen["simulator"]["gsd"] = GSD
        cfg_for_label_gen["simulator"]["world_range"] = current_ortho_dims_wh_px
        cfg_for_label_gen["simulator"]["sensor"]["resolution"] = SENSOR_RESOLUTION_PATCH_SIZE
        cfg_for_label_gen["simulator"]["sensor"]["angle"] = SENSOR_ANGLE
        
        cfg_for_label_gen["mapper"]["map_boundary"] = current_ortho_dims_wh_px
        cfg_for_label_gen["mapper"]["ground_resolution"] = [GSD, GSD]
        calculated_poses = calculate_mapper_poses(
            current_ortho_dims_wh_px[0], current_ortho_dims_wh_px[1], GSD,
            GENERATE_POSES_X_STEP_M, GENERATE_POSES_Y_STEP_M, GENERATE_POSES_TILE_DIM_M
        )
        cfg_for_label_gen["mapper"]["poses"] = calculated_poses
        print(f"  Configured mapper.poses for field {field_id}: {calculated_poses}")


        with open(TEMP_CONFIG_PATH, "w") as f:
            yaml.dump(cfg_for_label_gen, f)
        
        try:
            run_command(["python3", "main.py", "-c", TEMP_CONFIG_PATH])
        except Exception as e:
            print(f"ERROR during label generation for field {field_id}: {e}. Skipping to next field.")
            continue # Skip to next field if label generation fails

        # --- 2. Prepare config and run map_to_dataset.py (Patch Extraction) ---
        if not os.path.exists(current_pseudo_map_path):
            print(f"WARNING: Pseudo-label map for field {field_id} not found at {current_pseudo_map_path} after generation. Skipping patching.")
            continue

        print(f"\n--- Stage 2: Extracting Patches for Field {field_id} ---")
        cfg_for_patching = base_cfg.copy() # Start fresh or use cfg_for_label_gen

        # map_to_dataset.py uses these keys from the config:
        cfg_for_patching["data"]["maps"] = current_ortho_rgb_path # Original ortho
        cfg_for_patching["data"]["map_out_name"] = current_pseudo_map_path # Generated pseudo-label map

        # It also uses simulator.sensor.resolution for patch size
        # and mapper.map_boundary & mapper.poses for iteration bounds
        cfg_for_patching["simulator"]["sensor"]["resolution"] = SENSOR_RESOLUTION_PATCH_SIZE
        cfg_for_patching["mapper"]["map_boundary"] = current_ortho_dims_wh_px
        cfg_for_patching["mapper"]["poses"] = calculated_poses # Use the same poses grid

        with open(TEMP_CONFIG_PATH, "w") as f:
            yaml.dump(cfg_for_patching, f)

        field_patch_export_dir = os.path.join(BASE_PATCHED_DATA_DIR, field_id)
        # map_to_dataset.py creates images/ and semantics/ inside this export dir
        
        try:
            run_command(["python3", "map_to_dataset.py", "-c", TEMP_CONFIG_PATH, "-e", field_patch_export_dir, "-d", "300"]) # Added -d 300 for dpi
        except Exception as e:
            print(f"ERROR during patch extraction for field {field_id}: {e}. Skipping to next field.")
            continue

    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)
    print("\n\nAll fields processed.")
    print(f"Generated pseudo-label maps are in: {BASE_PSEUDO_LABEL_MAP_DIR}")
    print(f"Generated training patches are in: {BASE_PATCHED_DATA_DIR}")
    print(f"You can now update your main training config.yaml's 'data.root_dir' to '{BASE_PATCHED_DATA_DIR}'")

if __name__ == "__main__":
    main()