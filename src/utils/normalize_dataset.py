import os
from PIL import Image

path_prefix = "C:/Users/aliha/Alikhive/programming/AI/traffic_jam_cv/dlib_front_and_rear_vehicles_v1/"

def normalize_dataset(input_file, output_file):
    with open(input_file, 'r') as fi, open(output_file, 'w') as fo:
        for line in fi:
            lines = line.strip().split()
            image_path = lines[0]
            class_name, top, left, width, height = lines[1:6]
            top = int(top)
            left = int(left)
            width = int(width)
            height = int(height)

            image_path = path_prefix + image_path

            with Image.open(image_path) as img:
                img_width, img_height = img.size
                top_normalized = (top + height / 2) / img_height
                left_normalized = (left + width / 2) / img_width
                width_normalized = width / img_width
                height_normalized = height / img_height
                fo.write(f"{class_name} {left_normalized} {top_normalized} {width_normalized} {height_normalized}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Normalize Dataset from TXT file.')
    parser.add_argument('--i', type=str, required=True, help='Path to input TXT file with dataset')
    parser.add_argument('--o', type=str, required=True, help='Path to output TXT with result')

    args = parser.parse_args()
    normalize_dataset(input_file=args.i, output_file=args.o)
