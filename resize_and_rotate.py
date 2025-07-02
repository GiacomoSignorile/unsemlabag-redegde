import os
from PIL import Image

def resize_and_rotate_images(input_dir, output_dir, dimensions_map, rotation_angle=90):
    """
    Resize images back to their original dimensions and rotate them by a specified angle.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save processed images.
        dimensions_map (dict): A dictionary mapping file names to their original dimensions (width, height).
        rotation_angle (int): Angle to rotate the image (default is 90 degrees).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, file)

                try:
                    with Image.open(input_path) as img:
                        # Get the original dimensions for this file
                        original_size = dimensions_map.get(file)
                        if not original_size:
                            print(f"No dimensions found for {file}, skipping.")
                            continue

                        # Resize the image to its original dimensions
                        img = img.resize(original_size, Image.Resampling.LANCZOS)

                        # Rotate the image by the specified angle
                        img = img.rotate(rotation_angle, expand=True)

                        # Save the processed image
                        img.save(output_path)

                        print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "results/image_to_rotate"
    output_directory = "results/rotated_maps"
    dimensions_map = {
        "field_000_vertical_generated_label.png": (2802, 6847),
        "field_001_vertical_generated_label.png": (2287, 6645),
        "field_002_vertical_generated_label.png": (3699, 6945),
        "field_003_vertical_generated_label.png": (2311, 6836),
        "field_004_vertical_generated_label.png": (3170, 4692),
    }  # Replace with actual file names and dimensions

    resize_and_rotate_images(input_directory, output_directory, dimensions_map)