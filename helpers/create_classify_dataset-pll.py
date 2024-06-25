import os
#import pyfastcopy
import shutil
import random
import re
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_folder_structure(root_dir, train_dir, val_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

def extract_writer_and_page(filename):
    # Example filename format: {cluster}_{writer}-IMG_MAX_{page}_{patch}.png
    pattern = r'\d+_(\d+)-IMG_MAX_(\d+)'
    match = re.search(pattern, filename)
    if match:
        writer = match.group(1)
        page = match.group(2)
        return writer, page
    else:
        return None, None

def copy_file(src_dest_pair):
    src, dest = src_dest_pair
    shutil.copy(src, dest)

def copy_files(src_dir, dest_train_dir, dest_val_dir, val_pages):
    def get_dest_path(src_path, dest_train_dir, dest_val_dir, val_pages):
        filename = os.path.basename(src_path)
        writer, page = extract_writer_and_page(filename)
        if writer and page:
            writer_and_page = f"{writer}-IMG_MAX_{page}"
            if (writer, page) in val_pages:
                dest_path = os.path.join(dest_val_dir, writer_and_page, filename)
            else:
                dest_path = os.path.join(dest_train_dir, writer_and_page, filename)
            return dest_path
        else:
            return None

    jobs = []
    for root, _, files in os.walk(src_dir):
        for filename in files:
            src_path = os.path.join(root, filename)
            dest_path = get_dest_path(src_path, dest_train_dir, dest_val_dir, val_pages)
            if dest_path:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                jobs.append((src_path, dest_path))

    # Use ThreadPoolExecutor to copy files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
        breakpoint()
        futures = [executor.submit(copy_file, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying files"):
            pass  # Wait for all jobs to complete

def main(train_dataset_dir, train_classify_dir, val_classify_dir):
    create_folder_structure(train_dataset_dir, train_classify_dir, val_classify_dir)
    #breakpoint()
    # Step 1: Collecting unique pages for validation (max 1 page per author)
    writers_pages = defaultdict(set)
    for root, _, files in tqdm(os.walk(train_dataset_dir), 'Collecting unique pages'):
        for filename in files:
            #breakpoint()
            writer, page = extract_writer_and_page(filename)
            if writer and page:
                #breakpoint()
                writers_pages[writer].add(page)

    # Step 2: Determine number of writers for validation set
    unique_writers = list(writers_pages.keys())
    num_val_writers = max(1, len(unique_writers) // 10)  # Ensure at least one writer for validation

    # Step 3: Randomly select writers for validation set
    val_writers = random.sample(unique_writers, num_val_writers)

    # Step 4: Collect pages for validation set
    val_pages = set()
    for writer in tqdm(val_writers, 'Collect pages for validation set'):
        page = random.sample(writers_pages[writer], 1)[0]
        val_pages.add((writer, page))

    # Step 5: Copy images to train_classify and val_classify folders
    copy_files(train_dataset_dir, train_classify_dir, val_classify_dir, val_pages)

if __name__ == "__main__":
    train_dataset_dir = '/cluster/qy41tewa/rl-map/dataset/train/icdar2017_train_sift_patches_binarized_1000'
    train_classify_dir = '/cluster/qy41tewa/rl-map/dataset/classify_new/train'
    val_classify_dir = '/cluster/qy41tewa/rl-map/dataset/classify_new/val'
    

    main(train_dataset_dir, train_classify_dir, val_classify_dir)
