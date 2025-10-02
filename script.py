import os
import shutil
from math import ceil

# Define the input and output directories
input_dir = r"C:\Users\dhanu\OneDrive\Desktop\1pharma\opencv_annoted"
output_base_dir = r"C:\Users\dhanu\OneDrive\Desktop\1pharma"

# Create three output folders
output_dirs = [
    os.path.join(output_base_dir, "subset1"),
    os.path.join(output_base_dir, "subset2"),
    os.path.join(output_base_dir, "subset3")
]

# Create output directories if they don't exist
for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

# Get list of image files
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

# Calculate number of images per subset
total_images = len(image_files)
images_per_subset = ceil(total_images / 3)

# Split images into three subsets
subset1 = image_files[:images_per_subset]
subset2 = image_files[images_per_subset:2 * images_per_subset]
subset3 = image_files[2 * images_per_subset:]

# Assign subsets to folders
subsets = [subset1, subset2, subset3]

# Copy images to respective folders
for i, (subset, output_dir) in enumerate(zip(subsets, output_dirs), 1):
    for image_name in subset:
        src_path = os.path.join(input_dir, image_name)
        dst_path = os.path.join(output_dir, image_name)
        shutil.copyfile(src_path, dst_path)
        print(f"Copied {image_name} to subset{i}")

print(f"Total images: {total_images}")
print(f"Images in subset1: {len(subset1)}")
print(f"Images in subset2: {len(subset2)}")
print(f"Images in subset3: {len(subset3)}")
print(f"Images split into: {', '.join(output_dirs)}")