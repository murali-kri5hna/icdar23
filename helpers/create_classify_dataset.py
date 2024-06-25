import os
#import shutil
import pyfastcopy
import random
from tqdm import tqdm
from collections import defaultdict

def create_folder_structure(root_dir, train_dir, val_dir):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

def extract_writer_and_page(filename):
    # Example filename format: {cluster}_{writer}-IMG_MAX_{page}_{patch}.png
    pattern = r'(\w+)-IMG_MAX_(\d+)'
    match = re.search(pattern, filename)
    if match:
        writer = match.group(1)
        page = match.group(2)
        return f"{writer}-IMG_MAX_{page}"
    else:
        return None

def copy_files(src_dir, dest_train_dir, dest_val_dir, val_pages):
    for root, _, files in os.walk(src_dir):
        for filename in tqdm(files, desc=f'Copying {src_dir}'):
            writer_and_page = extract_writer_and_page(filename)
            if writer_and_page:
                src_path = os.path.join(root, filename)
                if writer_and_page in val_pages:
                    dest_path = os.path.join(dest_val_dir, writer_and_page, filename)
                else:
                    dest_path = os.path.join(dest_train_dir, writer_and_page, filename)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)

def main(train_dataset_dir, train_classify_dir, val_classify_dir):
    create_folder_structure(train_dataset_dir, train_classify_dir, val_classify_dir)

    # Step 1: Collecting unique pages for validation (max 1 page per author)
    writers_pages = defaultdict(set)
    for root, _, files in os.walk(train_dataset_dir):
        for filename in files:
            writer_and_page = extract_writer_and_page(filename)
            if writer_and_page:
                writer, page = writer_and_page.split('-IMG_MAX_')
                writers_pages[writer].add(page)

    # Step 2: Determine number of writers for validation set
    unique_writers = list(writers_pages.keys())
    num_val_writers = max(1, len(unique_writers) // 10)  # Ensure at least one writer for validation

    # Step 3: Randomly select writers for validation set
    val_writers = random.sample(unique_writers, num_val_writers)

    # Step 4: Collect pages for validation set
    val_pages = set()
    for writer in val_writers:
        val_pages.update(random.sample(writers_pages[writer], min(1, len(writers_pages[writer]))))

    # Step 5: Copy images to train_classify and val_classify folders
    copy_files(train_dataset_dir, train_classify_dir, val_classify_dir, val_pages)

if __name__ == "__main__":
    train_dataset_dir = '/path/to/your/train_dataset'
    train_classify_dir = '/path/to/your/train_classify'
    val_classify_dir = '/path/to/your/val_classify'

    main(train_dataset_dir, train_classify_dir, val_classify_dir)
