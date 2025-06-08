import os
import numpy as np
from PIL import Image
import tifffile
from scipy.ndimage import center_of_mass
from multiprocessing import Pool
from tqdm import tqdm
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Data loader cache
image_cache = {}
mask_cache = {}

def preload_data(files, stacked_imgs, mask_dir, max_cache_size=50 * 1024 * 1024 * 1024):  # 50 GB
    """Preload images and masks into memory asynchronously."""
    global image_cache, mask_cache
    image_cache.clear()
    mask_cache.clear()
    
    def load_image(file_path):
        return tifffile.memmap(file_path, mode='r')
    
    def load_mask(mask_path):
        return tifffile.memmap(mask_path, mode='r')
    
    # Estimate memory per file (32 MB image + 32 MB mask)
    mem_per_file = 64 * 1024 * 1024  # 64 MB
    max_files = min(len(files), max_cache_size // mem_per_file)
    files_to_load = files[:max_files]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit image and mask loading tasks
        image_futures = {f: executor.submit(load_image, os.path.join(stacked_imgs, f)) for f in files_to_load}
        mask_futures = {
            f: executor.submit(load_mask, os.path.join(mask_dir, os.path.splitext(f)[0] + "_cp_masks.tif"))
            for f in files_to_load
        }
        
        # Collect results
        for f in files_to_load:
            try:
                image_cache[f] = image_futures[f].result()
                mask_cache[f] = mask_futures[f].result()
            except Exception as e:
                logger.error(f"Failed to preload {f}: {str(e)}")
                image_cache[f] = None
                mask_cache[f] = None
    
    logger.info(f"Preloaded {len(image_cache)} files")

def log_processed_file(filename, processed_file):
    """Log processed files."""
    try:
        with open(processed_file, 'a') as f:
            f.write(filename + '\n')
    except Exception as e:
        logger.error(f"Failed to log {filename}: {str(e)}")

def patch_generator(image_path, mask_path, output_dir, patch_size=75, file_name=None):
    """Generate patches for all labels, saving each patch as a separate uncompressed TIFF file."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Use cached data if available, otherwise load from disk
        img_array = image_cache.get(file_name) if file_name in image_cache else tifffile.memmap(image_path, mode='r')
        mask = mask_cache.get(file_name) if file_name in mask_cache else np.array(Image.open(mask_path), dtype=np.uint16)
        
        # Compute labels and centroids
        unique_labels = np.unique(mask)[1:]
        centroids = center_of_mass(mask, labels=mask, index=unique_labels)
        centroids = [(int(y), int(x)) for y, x in centroids if y is not None and x is not None]
        
        image_height, image_width = mask.shape
        half_size = (patch_size + 1) // 2
        
        # Extract and write patches
        total_patches = 0
        for label, (center_y, center_x) in zip(unique_labels, centroids):
            top_left_x = center_x - half_size
            top_left_y = center_y - half_size
            bottom_right_x = center_x + half_size
            bottom_right_y = center_y + half_size
            
            if (top_left_x < 0 or top_left_y < 0 or 
                bottom_right_x > image_width or bottom_right_y > image_height):
                continue
            
            patch_view = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            patch_mask = mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            condition = (patch_mask == label)
            patch = np.where(condition[..., np.newaxis] if patch_view.ndim == 3 else condition, patch_view, 0)
            
            # Save each patch as a separate TIFF file without compression
            output_path = os.path.join(output_dir, f"{basename}_patch_{label}.tif")
            tifffile.imwrite(output_path, patch, compression=None)
            total_patches += 1
        
        return total_patches
    
    except MemoryError as e:
        logger.error(f"Memory error processing {basename}: {str(e)}")
        return 0
    except OSError as e:
        logger.error(f"I/O error processing {basename}: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error processing {basename}: {str(e)}")
        return 0
    finally:
        # Clean up memory-mapped array if not cached
        if file_name not in image_cache and isinstance(img_array, np.memmap):
            img_array._mmap.close()
            del img_array

def process_file(args):
    """Process a single file."""
    file, stacked_imgs, mask_dir, outdir, patch_size, processed_file = args
    try:
        base = os.path.splitext(file)[0]
        mask = base + "_cp_masks.tif"
        mask_path = os.path.join(mask_dir, mask)
        
        if not os.path.exists(mask_path) and file not in mask_cache:
            logger.error(f"Mask file {mask} not found for {file}")
            return 0
        
        patches = patch_generator(
            image_path=os.path.join(stacked_imgs, file),
            mask_path=mask_path,
            output_dir=outdir,
            patch_size=patch_size,
            file_name=file
        )
        
        log_processed_file(file, processed_file)
        
        return patches
    except Exception as e:
        logger.error(f"Error processing {file}: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Generate patches from images and masks.")
    parser.add_argument('--input-dir', type=str, 
                        default='/scratch/leuven/359/vsc35907/feature_extraction_data/stacked_images/mutants/',
                        help='Directory containing input TIFF images and masks subdirectory')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/leuven/359/vsc35907/feature_extraction_data/patches7/mutants/',
                        help='Directory to save output patches')
    parser.add_argument('--processed-file', type=str, default='processed.txt',
                        help='File to track processed TIFF files')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='Number of parallel processes (default: 4 for 8 CPUs)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of files per batch (default: 100)')
    args = parser.parse_args()
    
    stacked_imgs = args.input_dir
    outdir = args.output_dir
    mask_dir = os.path.join(stacked_imgs, "masks")
    processed_file = os.path.join(os.getcwd(), args.processed_file) if not os.path.isabs(args.processed_file) else args.processed_file
    
    logger.info(f"Scanning {stacked_imgs} for TIFF files")
    all_files = [f for f in os.listdir(stacked_imgs) if f.endswith(('.tif', '.tiff'))]
    
    processed_files = set()
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r') as f:
                processed_files = {line.strip() for line in f if line.strip().endswith(('.tif', '.tiff'))}
            logger.info(f"Loaded {len(processed_files)} processed files")
        except Exception as e:
            logger.error(f"Failed to read processed file: {str(e)}")
    else:
        os.makedirs(os.path.dirname(processed_file) or '.', exist_ok=True)
        with open(processed_file, 'w') as f:
            pass
        logger.info(f"Created new processed file at {processed_file}")
    
    files = [f for f in all_files if f not in processed_files]
    logger.info(f"Found {len(files)} unprocessed TIFF files")
    
    if not files:
        logger.warning("No TIFF files to process.")
        return
    
    try:
        # Process files in batches
        total_patches = 0
        batch_size = args.batch_size
        for i in tqdm(range(0, len(files), batch_size), desc="Processing batches", total=(len(files) + batch_size - 1) // batch_size):
            batch_files = files[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_files)} files")
            
            # Preload data for the batch
            preload_data(batch_files, stacked_imgs, mask_dir)
            
            # Process the batch
            process_args = [(f, stacked_imgs, mask_dir, outdir, 75, processed_file) for f in batch_files]
            with Pool(processes=args.num_processes) as pool:
                for patches in tqdm(pool.imap_unordered(process_file, process_args), 
                                  total=len(batch_files), 
                                  desc=f"Batch {i//batch_size + 1}"):
                    total_patches += patches
            
            # Clear caches to free memory
            image_cache.clear()
            mask_cache.clear()
        
        logger.info(f"Total patches generated: {total_patches}")
    
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")

if __name__ == "__main__":
    main()