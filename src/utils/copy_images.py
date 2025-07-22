import os
import shutil
from pathlib import Path

path_prefix = "C:/Users/aliha/Alikhive/programming/AI/traffic_jam_cv/dlib_front_and_rear_vehicles_v1/"

def copy_images_from_list(input_txt, output_dir):
    with open(input_txt, 'r') as f:
        for line in f:
            items = line.strip().split()
            src_path = items[0]
            src_path = path_prefix + src_path
            
            try:
                # Get the filename with extension
                filename = os.path.basename(src_path)
                dst_path = os.path.join(output_dir, filename)
                
                # Copy the image (preserves metadata)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")
                
            except FileNotFoundError:
                print(f"Error: Source file not found - {src_path}")
            except Exception as e:
                print(f"Error copying {src_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy labels for the dataset')
    parser.add_argument('--images_input', required=True, help='File containing the image entries')
    parser.add_argument('--labels_input', required=True, help='File containing the normalized labels')
    parser.add_argument('--output_dir', required=True, help='Destination directory for the labels')
    
    args = parser.parse_args()
    copy_images_from_list(args.input_txt, args.output_dir)
