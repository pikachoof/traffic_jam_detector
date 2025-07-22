import os
import shutil
from pathlib import Path

path_prefix = "C:/Users/aliha/Alikhive/programming/AI/traffic_jam_cv/dlib_front_and_rear_vehicles_v1/"

def copy_labels(images_txt, labels_txt, output_dir):
    with open(images_txt, 'r') as f_images, open(labels_txt, 'r') as f_labels:
        for images_line, labels_line in zip(f_images, f_labels):
            items_images = images_line.strip().split()
            image_name = items_images[0]

            image_name = os.path.splitext(os.path.basename(image_name))[0]
            output_path = os.path.join(output_dir, f"{image_name}.txt")

            with open(output_path, 'a') as f_output:
                f_output.write(labels_line.strip() + '\n')

            print(f"Created: {output_path}")
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Copy labels for the dataset')
    parser.add_argument('--images_txt', required=True, help='File containing image paths')
    parser.add_argument('--labels_txt', required=True, help='File containg normalized labels')
    parser.add_argument('--output_dir', required=True, help='Destination directory for the labels')
    
    args = parser.parse_args()
    copy_labels(args.images_txt, args.labels_txt, args.output_dir)
