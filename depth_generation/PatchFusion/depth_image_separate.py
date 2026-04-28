import os
import shutil

def sort_depth_images():
    """
    Sorts depth images from a source directory into two separate folders
    based on their filename suffixes (_uint16.png and .png).
    """
    
    # --- Configuration ---
    
    # Define the base path from your example
    # We'll place the new folders alongside 'pretrained_models'
    base_dir = "/Vaibhav/shivasish1/sam2/PatchFusion/"
    
    # 1. Source Directory (where your mixed files are)
    source_dir = os.path.join(base_dir, "pretrained_models")
    
    # 2. Destination for '_uint16.png' files
    dest_dir_uint16 = os.path.join(base_dir, "sorted_depths_uint16")
    
    # 3. Destination for standard '.png' files
    dest_dir_standard = os.path.join(base_dir, "sorted_depths_standard")
    
    # --- End Configuration ---

    print(f"Starting image sort process...")

    # Create destination directories if they don't already exist
    try:
        os.makedirs(dest_dir_uint16, exist_ok=True)
        print(f"Ensured directory exists: {dest_dir_uint16}")
        
        os.makedirs(dest_dir_standard, exist_ok=True)
        print(f"Ensured directory exists: {dest_dir_standard}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        return

    # Check if the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory not found at: {source_dir}")
        print("Please check the 'source_dir' variable in the script.")
        return

    print(f"Scanning source directory: {source_dir}\n")
    
    # Counters for summary
    moved_count_uint16 = 0
    moved_count_standard = 0
    skipped_count = 0

    # Iterate over every file in the source directory
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        
        # Make sure we're only processing files, not subdirectories
        if not os.path.isfile(src_path):
            continue
            
        try:
            # --- Sorting Logic ---
            
            # First, check for the more specific type: '_uint16.png'
            if filename.endswith("_uint16.png"):
                dest_path = os.path.join(dest_dir_uint16, filename)
                shutil.move(src_path, dest_path)
                # print(f"Moved (uint16): {filename}") # Uncomment for verbose logging
                moved_count_uint16 += 1
                
            # Else, check if it's a '.png' file (that isn't '_uint16.png')
            elif filename.endswith(".png"):
                dest_path = os.path.join(dest_dir_standard, filename)
                shutil.move(src_path, dest_path)
                # print(f"Moved (standard): {filename}") # Uncomment for verbose logging
                moved_count_standard += 1
                
            else:
                # This file is not a .png or _uint16.png, so we skip it
                skipped_count += 1
                
        except Exception as e:
            print(f"Error moving file {filename}: {e}")
            skipped_count += 1

    # --- Print Summary ---
    print("\n--- Sorting Complete ---")
    print(f"Moved {moved_count_uint16} '_uint16.png' files to: {dest_dir_uint16}")
    print(f"Moved {moved_count_standard} standard '.png' files to: {dest_dir_standard}")
    print(f"Skipped {skipped_count} other files/directories.")

if __name__ == "__main__":
    sort_depth_images()