import os
import shutil
import random
import math

def split_data_directory():
    """
    Splits a single folder of images into train, val, and test sets.
    This mimics the behavior of the 'splitfolders' tool.
    """
    
    # --- Configuration ---
    
    # 1. Input Directory (The folder with all your images)
    input_dir = "/Vaibhav/shivasish1/PDFNet/newdepthdata/sorted_depths_uint16"
    
    # 2. Output Directory (Where 'train', 'val', 'test' will be created)
    output_dir = "/Vaibhav/shivasish1/PDFNet/newdepthdata/split_data_uint16"
    
    # 3. Split Ratios
    train_ratio = 0.7
    val_ratio = 0.2
    # test_ratio will be what's left over (0.1)
    
    # 4. Random Seed (for reproducible results)
    random_seed = 42
    
    # --- End Configuration ---

    print(f"Starting data split...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}\n")

    # Set the seed for reproducibility
    random.seed(random_seed)

    # --- 1. Get all filenames ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    try:
        # Get all entries in the directory
        all_files = os.listdir(input_dir)
        
        # Filter for files only (ignore subdirectories)
        filenames = [f for f in all_files if os.path.isfile(os.path.join(input_dir, f))]
        
        if not filenames:
            print(f"Error: No files found in {input_dir}")
            return
            
        print(f"Found {len(filenames)} total files to split.")
        
    except Exception as e:
        print(f"Error reading directory: {e}")
        return

    # --- 2. Shuffle and split the list ---
    random.shuffle(filenames)
    
    total_count = len(filenames)
    train_count = math.floor(total_count * train_ratio)
    val_count = math.floor(total_count * val_ratio)
    
    # The rest go to 'test'
    train_files = filenames[:train_count]
    val_files = filenames[train_count : train_count + val_count]
    test_files = filenames[train_count + val_count :]

    print(f"Splitting into: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    # --- 3. Helper function to copy files ---
    def copy_files(file_list, set_name):
        """
        Copies a list of files to a specific set (train/val/test) folder.
        """
        # Create the target directory, e.g., .../split_data_uint16/train
        target_dir = os.path.join(output_dir, set_name)
        os.makedirs(target_dir, exist_ok=True)
        
        copied_count = 0
        for filename in file_list:
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(target_dir, filename)
            
            try:
                shutil.copy(src_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Warning: Could not copy file {filename}. Error: {e}")
                
        print(f"Successfully copied {copied_count} files to '{set_name}' set.")

    # --- 4. Execute the copying ---
    try:
        copy_files(train_files, "train")
        copy_files(val_files, "val")
        copy_files(test_files, "test")
        
        print("\n--- Data split complete! ---")
        print(f"All files copied to: {output_dir}")
        
    except Exception as e:
        print(f"\nAn error occurred during the copying process: {e}")

if __name__ == "__main__":
    split_data_directory()