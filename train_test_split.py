import os
import shutil
import random
from pathlib import Path
import argparse

def create_directories(base_path):
    """Create necessary directories for train, test, and val splits."""
    splits = ['train', 'test', 'val']
    for split in splits:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(base_path, split, subdir), exist_ok=True)

def split_data(source_dir, split_ratios=(0.7, 0.2, 0.1)):
    """
    Split data into train, test, and validation sets.
    
    Args:
        source_dir: Path to directory containing 'images' and 'labels' subdirectories
        split_ratios: Tuple of (train, test, val) ratios that sum to 1.0
    """
    # Convert source_dir to Path object
    source_dir = Path(source_dir)
    
    # Get list of image files
    image_files = list((source_dir / 'images').glob('*'))
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Calculate split indices
    n_files = len(image_files)
    train_end = int(n_files * split_ratios[0])
    test_end = train_end + int(n_files * split_ratios[1])
    
    # Split files into train, test, val
    splits = {
        'train': image_files[:train_end],
        'test': image_files[train_end:test_end],
        'val': image_files[test_end:]
    }
    print(f"Splitting data into {len(splits['train'])} training, {len(splits['test'])} testing, and {len(splits['val'])} validation samples.")

    # Create output directories
    create_directories(source_dir.parent)
    
    # Move files to respective directories
    for split_name, files in splits.items():
        for img_path in files:
            # Get corresponding label path
            label_path = source_dir / 'labels' / img_path.name.replace(img_path.suffix, '.txt')
            
            if not label_path.exists():
                print(f"Warning: No label file found for {img_path}")
                continue
            
            # Define destination paths
            img_dest = source_dir.parent / split_name / 'images' / img_path.name
            label_dest = source_dir.parent / split_name / 'labels' / label_path.name
            
            # Move files
            shutil.move(str(img_path), str(img_dest))
            shutil.move(str(label_path), str(label_dest))

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, test, and validation sets')
    parser.add_argument('source_dir', type=str, help='Path to directory containing images and labels subdirectories')
    args = parser.parse_args()
    split_data(args.source_dir)
    print("Dataset split complete!")

if __name__ == '__main__':
    main()