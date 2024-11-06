import os
import sys
import logging
import requests
import tarfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import hf_hub_download
from config import KAGGLE_DATASET, HF_DATASET, DOWNLOAD_DIR, LOG_FILE, HF_TOKEN


ADDITIONAL_FILES = [
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/train_filtered.txt",
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/valid_filtered.txt",
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/test_filtered.txt"
]

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def log_message(message):
    logging.info(message)


def download_from_kaggle():
    log_message("Attempting to download dataset from Kaggle.")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=DOWNLOAD_DIR, unzip=True)
        move_and_cleanup_kaggle_download()
        log_message("Successfully downloaded and organized dataset from Kaggle.")
        return True
    except Exception as e:
        log_message(
            f"Failed to download dataset from Kaggle: {e}. Please ensure that you have a valid token in your ~/.kaggle/kaggle.json file.")
        return False


def move_and_cleanup_kaggle_download():
    source_dir = os.path.join(DOWNLOAD_DIR, "Data", "genres_original")
    target_dir = os.path.join(DOWNLOAD_DIR, "genres")

    if os.path.exists(source_dir):
        log_message(f"Moving {source_dir} to {target_dir}")
        shutil.move(source_dir, target_dir)

    # Data directory cleanup
    data_dir = os.path.join(DOWNLOAD_DIR, "Data")
    if os.path.exists(data_dir):
        log_message(f"Deleting the now-empty {data_dir} directory")
        shutil.rmtree(data_dir)


def download_from_huggingface():
    """Attempt to download dataset from Hugging Face."""
    log_message("Attempting to download dataset from Hugging Face.")
    try:
        hf_hub_download(repo_id=HF_DATASET, filename="data/genres.tar.gz", local_dir=DOWNLOAD_DIR, token=HF_TOKEN,
                        repo_type="dataset")
        move_and_extract_hf_download()
        return True
    except Exception as e:
        log_message(f"Failed to download dataset from Hugging Face: {e}")
        return False


def move_and_extract_hf_download():
    nested_file_path = os.path.join(DOWNLOAD_DIR, "data", "genres.tar.gz")
    target_path = os.path.join(DOWNLOAD_DIR, "genres.tar.gz")

    if os.path.exists(nested_file_path):
        log_message(f"Moving {nested_file_path} to {target_path}")
        shutil.move(nested_file_path, target_path)

    log_message(f"Extracting {target_path}...")
    try:
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall(path=DOWNLOAD_DIR)
        log_message(f"Successfully extracted {target_path}")

        os.remove(target_path)
        log_message(f"Removed archive file {target_path} after extraction.")
    except Exception as e:
        log_message(f"Failed to extract {target_path}: {e}")


def download_index_files():
    log_message("Downloading index GTZAN files...")
    success = True
    for url in ADDITIONAL_FILES:
        filename = os.path.join(".", os.path.basename(url))
        try:
            response = requests.get(url)
            response.raise_for_status()  # Checking response status
            with open(filename, 'wb') as f:
                f.write(response.content)
            log_message(f"Downloaded {filename}")
        except Exception as e:
            log_message(f"Failed to download {url}: {e}")
            success = False
    return success


if __name__ == "__main__":
    if download_from_kaggle():
        download_index_files()
        sys.exit(0)

    if download_from_huggingface():
        download_index_files()
        sys.exit(0)

    log_message("Failed to download dataset from any source.")
    sys.exit(1)
