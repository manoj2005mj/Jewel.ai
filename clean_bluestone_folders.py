import os
import shutil

def clean_empty_folders(base_folder):
    if not os.path.exists(base_folder):
        print(f"Folder '{base_folder}' does not exist.")
        return

    # Image extensions to check for
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'}

    deleted_count = 0
    
    # Iterate over all items in the base folder
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        
        # We only care about directories
        if os.path.isdir(item_path):
            has_images = False
            # Check content of the subfolder
            for filename in os.listdir(item_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in image_extensions:
                    has_images = True
                    break
            
            if not has_images:
                print(f"Deleting '{item}' (0 images)...")
                try:
                    shutil.rmtree(item_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {item}: {e}")

    print(f"\nCleanup complete. Deleted {deleted_count} folders.")

if __name__ == "__main__":
    target_folder = "downloaded_images_bluestone"
    # Ensure absolute path just in case, or relative to current working directory
    base_path = os.path.abspath(target_folder)
    print(f"Scanning: {base_path}")
    clean_empty_folders(base_path)
