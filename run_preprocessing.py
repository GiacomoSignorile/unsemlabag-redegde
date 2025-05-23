import os
import shutil # For removing temp directories if needed, though not used in this version

# Assuming preprocess.py is in datasets/ folder. Adjust if it's elsewhere (e.g., utils/)
from datasets.preprocess import divide_single_file_into_patches

def main():
    # --- Configuration ---
    base_input_dir = "./samples/RedEdge"    # Where your original orthos are
    base_output_dir = "./samples/RedEdge_Patches_224" # New directory for patched data
    fields_to_process = ["000", "001", "002", "003", "004"] # Or load from a config
    
    patch_size = 224 # As per RoWeeder's training config
    
    # Filename of the composite RGB image within each field's 'composite-png' folder
    composite_rgb_filename = "RGB.png" 
    # --- End Configuration ---

    if os.path.exists(base_output_dir):
        print(f"Output directory {base_output_dir} already exists. Please remove or rename it if you want to re-generate patches.")
        # choice = input("Do you want to remove it and continue? (yes/no): ")
        # if choice.lower() == 'yes':
        #     shutil.rmtree(base_output_dir)
        # else:
        #     print("Exiting.")
        #     return
    os.makedirs(base_output_dir, exist_ok=True)


    for field_id in fields_to_process:
        print(f"\nProcessing field: {field_id}")

        # Define paths for the current field
        field_input_path = os.path.join(base_input_dir, field_id)
        field_output_path = os.path.join(base_output_dir, field_id)
        
        os.makedirs(os.path.join(field_output_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(field_output_path, "semantics"), exist_ok=True)

        # --- 1. Process the RGB composite image ---
        rgb_image_input_path = os.path.join(field_input_path, "composite-png", composite_rgb_filename)
        
        if os.path.exists(rgb_image_input_path):
            print(f"  Patching RGB image: {rgb_image_input_path}")
            divide_single_file_into_patches(
                input_file_path=rgb_image_input_path,
                base_output_folder=os.path.join(field_output_path, "images"),
                output_subfolder_name="RGB", # Patches will be in .../images/RGB/0.png etc.
                patch_size=patch_size
            )
        else:
            print(f"  WARNING: RGB image not found: {rgb_image_input_path}")

        # --- 2. Process the Ground Truth image(s) ---
        gt_input_folder = os.path.join(field_input_path, "groundtruth")
        if os.path.isdir(gt_input_folder):
            for gt_filename in os.listdir(gt_input_folder):
                if gt_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    gt_image_input_path = os.path.join(gt_input_folder, gt_filename)
                    gt_output_subfolder_name = os.path.splitext(gt_filename)[0] # e.g., "first000_gt"
                    
                    print(f"  Patching Ground Truth: {gt_image_input_path}")
                    divide_single_file_into_patches(
                        input_file_path=gt_image_input_path,
                        base_output_folder=os.path.join(field_output_path, "semantics"),
                        output_subfolder_name=gt_output_subfolder_name, # Patches in .../semantics/first000_gt/0.png
                        patch_size=patch_size
                    )
                else:
                    print(f"  Skipping non-image file in groundtruth: {gt_filename}")
        else:
            print(f"  WARNING: Ground truth folder not found: {gt_input_folder}")

    print("\nPreprocessing into patches complete.")
    print(f"Patched data saved in: {base_output_dir}")

if __name__ == "__main__":
    main()