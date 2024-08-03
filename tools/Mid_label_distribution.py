# Label distribution analysis tools for S.MID
import os
import numpy as np
import argparse
from tqdm import tqdm
def list_label_files(folder_path):
    label_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".label"):
                label_files.append(os.path.join(root, file))
    return label_files

def calculate_class_statistics(label_files):
    label_total = [0, 1, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 116, 117, 118, 119, 120, 121, 122,
                 200, 201, 202, 203, 204]
    frame_label = {label: 0 for label in label_total}

    for frame_idx in tqdm(label_files, desc="Processing files"):
        labelm = np.fromfile(frame_idx, dtype=np.uint32).reshape(-1, 4)
        for label in label_total:
            frame_label[label] += np.sum(labelm == label) / 1000
    return frame_label

def map_and_aggregate_classes(frame_label,class_num = 14):
    map = {
        0: 0, 1: 0, 101: 1, 102: 2, 103: 3, 104: 3, 105: 1, 106: 4, 107: 5, 108: 0,
        109: 6, 110: 7, 111: 8, 112: 9, 116: 10, 117: 11, 118: 0, 119: 12, 120: 13,
        121: 14, 122: 0, 200: 0, 201: 0, 202: 0, 203: 0, 204: 3
    }
    new_array_list = np.zeros(class_num)
    for label, count in frame_label.items():
        mapped_class = map[label]
        if mapped_class != 0:
            new_array_list[mapped_class - 1] += count
    return new_array_list

def print_class_statistics(new_array_list):
    for i, count in enumerate(new_array_list, start=1):
        print(f"Class {i}: {count:.3f}k")

def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in S.MID dataset.")
    parser.add_argument("source_folder", type=str, help="Path to the folder containing the label files.")
    args = parser.parse_args()

    # Set folders
    label_files = list_label_files(args.source_folder)

    # Original label distribution dict
    label_distribution_dict = calculate_class_statistics(label_files)

    # For single-frame LiDAR semantic segmentation
    new_array_list = map_and_aggregate_classes(label_distribution_dict)
    print_class_statistics(new_array_list)

if __name__ == "__main__":
    main()