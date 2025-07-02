import os
from PIL import Image, ImageEnhance

def resize_and_enhance_images(input_dir, output_dir, size=(2266, 8750), enhancement_factor=1.5):
    """
    Resize and enhance all RGB.png images in the specified directory and its subdirectories.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save processed images.
        size (tuple): Target size for resizing (width, height).
        enhancement_factor (float): Factor to enhance image brightness.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("RGB.png"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)

                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, file)

                try:
                    with Image.open(input_path) as img:
                        # Resize the image
                        img = img.resize(size, Image.Resampling.LANCZOS)

                        # Enhance the image brightness
                        enhancer = ImageEnhance.Brightness(img)
                        img = enhancer.enhance(enhancement_factor)

                        # Save the processed image
                        img.save(output_path)

                        print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "samples/rotated_ortho2"
    output_directory = "samples/processed_images"

    resize_and_enhance_images(input_directory, output_directory)