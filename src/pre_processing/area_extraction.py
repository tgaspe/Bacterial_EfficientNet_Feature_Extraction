import numpy as np
import tifffile
import os
import multiprocessing
import argparse
import csv
from tqdm import tqdm

def compute_cell_area(path_tuple):
    full_path, relative_path = path_tuple
    try:
        img_array = tifffile.imread(full_path)
        if len(img_array.shape) != 3 or img_array.shape[2] != 4:
            raise ValueError("Image must have 4 channels")
        first_channel = img_array[:, :, 0]
        area = np.count_nonzero(first_channel)
        return (relative_path, area)
    except ValueError as e:
        return (relative_path, np.nan)  # Return NaN for ValueError
    except Exception as e:
        return (relative_path, str(e))  # Log other exceptions

def main():
    parser = argparse.ArgumentParser(description="Compute cell areas from TIFF images in parallel, handling subdirectories.")
    parser.add_argument('--input-dir', type=str, required=True, help="Directory containing 'wild_type' and 'mutants' folders")
    parser.add_argument('--output-csv', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--num-processes', type=int, default=None, help="Number of processes to use (default: all available cores)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_csv = args.output_csv
    num_processes = args.num_processes

    if not os.path.exists(input_dir):
        raise ValueError("Input directory does not exist")

    wild_type_dir = os.path.join(input_dir, "wild_type")
    mutant_dir = os.path.join(input_dir, "mutants")

    if not os.path.exists(wild_type_dir) or not os.path.exists(mutant_dir):
        raise ValueError("Input directory must contain 'wild_type' and 'mutants' folders")

    # Collect image paths from all subdirectories
    wild_type_images = []
    for root, _, files in os.walk(wild_type_dir):
        for filename in files:
            if filename.endswith((".tif", ".tiff")):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir)
                wild_type_images.append((full_path, relative_path))

    mutant_images = []
    for root, _, files in os.walk(mutant_dir):
        for filename in files:
            if filename.endswith((".tif", ".tiff")):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir)
                mutant_images.append((full_path, relative_path))

    image_paths = wild_type_images + mutant_images

    print(f"Found {len(wild_type_images)} images in wild_type and {len(mutant_images)} images in mutants.")

    with open(output_csv, 'w', newline='') as f, open('errors.log', 'w') as error_log:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'area'])
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap(compute_cell_area, image_paths), total=len(image_paths), desc="Processing images"):
                if isinstance(result[1], (int, float)) or np.isnan(result[1]):
                    writer.writerow([result[0], result[1] if not np.isnan(result[1]) else 'NaN'])
                else:
                    error_log.write(f"{result[0]}: {result[1]}\n")

    print(f"Results saved to {output_csv}")
    if os.path.getsize('errors.log') > 0:
        print("Some images failed to process. Check errors.log for details.")

if __name__ == "__main__":
    main()