import os

# Define the base directory and subset folders
base_dir = r"C:\Users\dhanu\OneDrive\Desktop\1pharma"
subsets = ["subset1", "subset2", "subset3"]

# Define valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png')

# Initialize total count
total_images = 0

# Count images in each subset
for subset in subsets:
    folder_path = os.path.join(base_dir, subset)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    # Count images
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    image_count = len(image_files)
    total_images += image_count
    
    print(f"{subset}: {image_count} images")
    
# Print total
print(f"Total images across all subsets: {total_images}")