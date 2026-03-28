import os
import json
import shutil

def process_dataset_split(base_dir, split_name):
    """
    Reads labels.json in the split directory, creates 'Empty' and 'Occupied' 
    subdirectories, and moves the images appropriately.
    """
    split_dir = os.path.join(base_dir, split_name)
    images_dir = os.path.join(split_dir, 'images')
    json_path = os.path.join(split_dir, 'labels.json')
    
    # Target directories
    empty_dir = os.path.join(split_dir, 'Empty')
    occupied_dir = os.path.join(split_dir, 'Occupied')
    
    if not os.path.exists(json_path):
        print(f"Skipping {split_name} dataset... (No labels.json found)")
        return
        
    if not os.path.exists(images_dir):
        print(f"Skipping {split_name} dataset... (No 'images' directory found)")
        return
        
    print(f"Processing {split_name} dataset...")
    
    # Create target directories
    os.makedirs(empty_dir, exist_ok=True)
    os.makedirs(occupied_dir, exist_ok=True)
    
    try:
        with open(json_path, 'r') as f:
            labels = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return
        
    empty_count = 0
    occupied_count = 0
    missing_count = 0
    
    for img_name, label in labels.items():
        # Handle case variations in json values
        label_normalized = label.strip().lower()
        
        src_image_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(src_image_path):
            missing_count += 1
            continue
            
        if label_normalized == 'empty':
            dst_image_path = os.path.join(empty_dir, img_name)
            # Use copy2 to preserve metadata. Alternatively, use shutil.move
            shutil.copy2(src_image_path, dst_image_path)
            empty_count += 1
        elif label_normalized == 'occupied':
            dst_image_path = os.path.join(occupied_dir, img_name)
            shutil.copy2(src_image_path, dst_image_path)
            occupied_count += 1
            
    # Print Summary
    print(f"Empty images: {empty_count}")
    print(f"Occupied images: {occupied_count}")
    if missing_count > 0:
        print(f"Missing images (listed in JSON but not in folder): {missing_count}")
    print("-" * 30)

def main():
    base_dataset_dir = 'dataset'
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        process_dataset_split(base_dataset_dir, split)
        
    print("Dataset preparation complete.")

if __name__ == '__main__':
    main()
