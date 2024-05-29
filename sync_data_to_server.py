import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import threading

# Set up logging
logging.basicConfig(filename='settings_sync_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def copy_folder(src, dst, lock, pbar):
    """Copy folder from src to dst without overwriting existing files/folders."""
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        logging.info(f"Copied folder {src} to {dst}")
    else:
        for item in os.listdir(src):
            pbar.update(1)  # Update progress bar after each item
            s_item = os.path.join(src, item)
            d_item = os.path.join(dst, item)
            if os.path.isdir(s_item):
                copy_folder(s_item, d_item, lock, pbar)  # Recursive call for nested folders
            else:
                if not os.path.exists(d_item):
                    shutil.copy2(s_item, d_item)  # Copy file if it doesn't exist in destination
                    logging.info(f"Copied file {s_item} to {d_item}")
            # with lock:
            #     pbar.update(1)

def get_folders(directory):
    """Get a list of all folders in a directory."""
    return [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def read_settings(file_path):
    """Read source and destination directories from a settings file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        src_dir = None
        dst_dir = None
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines
            if src_dir is None:
                src_dir = line
            elif dst_dir is None:
                dst_dir = line
    if not src_dir or not dst_dir:
        raise ValueError("Source or destination directory not specified in settings file.")
    return src_dir, dst_dir

def main():
    # Define the settings file path
    settings_file = os.path.join('settings', 'settings_server_sync.txt')

    # Read source and destination directories from settings file
    src_directory, dst_directory = read_settings(settings_file)

    # Ensure the source directory exists
    if not os.path.exists(src_directory):
        raise FileNotFoundError(f"Source directory {src_directory} does not exist")

    # Ensure the destination directory exists, if not, create it
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    # Get list of folders in source directory
    folders = get_folders(src_directory)

    # Count total number of files and directories to copy
    total_items = sum(len(files) + len(dirs) for _, dirs, files in os.walk(src_directory))

    # Initialize a lock for thread-safe progress bar updates
    lock = threading.Lock()

    # Get the number of threads
    num_threads = os.cpu_count() or 1

    print(f"Using {num_threads} threads.")

    # Use ThreadPoolExecutor for parallel copying
    with tqdm(total=total_items, desc="Copying", unit="item") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            # Submit a copy task for each folder
            for folder in folders:
                future = executor.submit(copy_folder, folder, os.path.join(dst_directory, os.path.basename(folder)), lock, pbar)
                futures.append(future)
            
            # Monitor futures for completion
            for future in as_completed(futures):
                future.result()

if __name__ == "__main__":
    main()
