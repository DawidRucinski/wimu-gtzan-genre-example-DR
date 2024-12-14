import os
import sys
import logging
import requests
import tarfile
import shutil
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import hf_hub_download
from config import KAGGLE_DATASET, HF_DATASET, DOWNLOAD_DIR, LOG_FILE, HF_TOKEN
from requests.exceptions import HTTPError, ConnectionError
from kaggle.rest import ApiException

# Index files for the GTZAN dataset
ADDITIONAL_FILES = [
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/train_filtered.txt",
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/valid_filtered.txt",
    "https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/test_filtered.txt"
]

def setup_logging(log_file):
    """Set up logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def download_from_kaggle(dataset=KAGGLE_DATASET, download_dir=DOWNLOAD_DIR, unzip=True):
    """Download dataset from Kaggle."""
    logging.info("Attempting to download dataset from Kaggle.")
    os.makedirs(download_dir, exist_ok=True)
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=download_dir, unzip=unzip)
        move_and_cleanup_kaggle_download(download_dir)
        logging.info("Successfully downloaded and organized dataset from Kaggle.")
    except ApiException as api_error:
        logging.error(f"Kaggle API error: {api_error}")
        raise RuntimeError(f"Kaggle API error: {api_error}")
    except KeyboardInterrupt:
        logging.error("Kaggle download interrupted by user.")
        sys.exit(1)
    except Exception as general_error:
        logging.error(f"Unexpected error during Kaggle download: {general_error}")
        raise RuntimeError(f"Kaggle download encountered an unexpected error: {general_error}")

def move_and_cleanup_kaggle_download(download_dir):
    """Move and clean up Kaggle download directory."""
    source_dir = os.path.join(download_dir, "Data", "genres_original")
    target_dir = os.path.join(download_dir, "genres")

    if os.path.exists(source_dir):
        logging.info(f"Moving {source_dir} to {target_dir}")
        shutil.move(source_dir, target_dir)

    data_dir = os.path.join(download_dir, "Data")
    if os.path.exists(data_dir):
        logging.info(f"Deleting the now-empty {data_dir} directory")
        shutil.rmtree(data_dir)

def download_from_huggingface(dataset=HF_DATASET, download_dir=DOWNLOAD_DIR, filename="genres.tar.gz", token=HF_TOKEN):
    """Download dataset from Hugging Face."""
    logging.info("Attempting to download dataset from Hugging Face.")
    os.makedirs(download_dir, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=dataset,
            filename=os.path.join("data", filename),
            local_dir=download_dir,
            token=token,
            repo_type="dataset"
        )
        move_and_extract_hf_download(download_dir, filename)
        logging.info("Successfully downloaded and organized dataset from Hugging Face.")
    except HTTPError as http_error:
        logging.error(f"HTTP error during Hugging Face download: {http_error}")
        raise RuntimeError(f"HTTP error during Hugging Face download: {http_error}")
    except ConnectionError as connection_error:
        logging.error(f"Connection error during Hugging Face download: {connection_error}")
        raise RuntimeError(f"Connection error during Hugging Face download: {connection_error}")
    except KeyboardInterrupt:
        logging.error("Hugging Face download interrupted by user.")
        sys.exit(1)
    except Exception as general_error:
        logging.error(f"Unexpected error during Hugging Face download: {general_error}")
        raise RuntimeError(f"Hugging Face download encountered an unexpected error: {general_error}")

def move_and_extract_hf_download(download_dir, filename="genres.tar.gz"):
    """Move and extract the downloaded file from Hugging Face."""
    nested_file_path = os.path.join(download_dir, "data", filename)
    target_path = os.path.join(download_dir, filename)

    if os.path.exists(nested_file_path):
        logging.info(f"Moving {nested_file_path} to {target_path}")
        shutil.move(nested_file_path, target_path)

    logging.info(f"Extracting {target_path}...")
    try:
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        logging.info(f"Successfully extracted {target_path}")

        os.remove(target_path)
        logging.info(f"Removed archive file {target_path} after extraction.")
    except tarfile.TarError as tar_error:
        logging.error(f"Failed to extract {target_path}: {tar_error}")
        raise RuntimeError(f"Failed to extract {target_path}: {tar_error}")

def download_index_files(download_dir=DOWNLOAD_DIR, index_files=ADDITIONAL_FILES):
    """Download additional index files."""
    logging.info("Downloading index GTZAN files...")
    os.makedirs(download_dir, exist_ok=True)
    for url in index_files:
        filename = os.path.join(download_dir, os.path.basename(url))
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded {filename}")
        except HTTPError as http_error:
            logging.error(f"HTTP error during index file download {url}: {http_error}")
            raise RuntimeError(f"HTTP error during index file download {url}: {http_error}")
        except ConnectionError as connection_error:
            logging.error(f"Connection error during index file download {url}: {connection_error}")
            raise RuntimeError(f"Connection error during index file download {url}: {connection_error}")
        except KeyboardInterrupt:
            logging.error("Index file download interrupted by user.")
            sys.exit(1)
        except Exception as general_error:
            logging.error(f"Unexpected error during index file download {url}: {general_error}")
            raise RuntimeError(f"Unexpected error during index file download {url}: {general_error}")

def main():
    """Main function to coordinate downloads."""
    parser = argparse.ArgumentParser(description="Download datasets from Kaggle or Hugging Face.")
    parser.add_argument("--kaggle_dataset", type=str, default=KAGGLE_DATASET, help="Kaggle dataset ID.")
    parser.add_argument("--hf_dataset", type=str, default=HF_DATASET, help="Hugging Face dataset repository.")
    parser.add_argument("--download_dir", type=str, default=DOWNLOAD_DIR, help="Directory to save downloaded files.")
    parser.add_argument("--log_file", type=str, default=LOG_FILE, help="Path to the log file.")
    parser.add_argument("--hf_token", type=str, default=HF_TOKEN, help="Hugging Face authentication token.")

    args = parser.parse_args()

    setup_logging(args.log_file)

    try:
        # Try downloading from Kaggle first
        logging.info("Starting Kaggle download attempt.")
        download_from_kaggle(dataset=args.kaggle_dataset, download_dir=args.download_dir)
        download_index_files(download_dir=args.download_dir)
        sys.exit(0)
    except RuntimeError as kaggle_error:
        logging.warning(f"Kaggle download failed: {kaggle_error}")

    try:
        # If Kaggle download fails, try Hugging Face
        logging.info("Starting Hugging Face download attempt.")
        download_from_huggingface(
            dataset=args.hf_dataset,
            download_dir=args.download_dir,
            token=args.hf_token
        )
        download_index_files(download_dir=args.download_dir)
        sys.exit(0)
    except RuntimeError as hf_error:
        logging.warning(f"Hugging Face download failed: {hf_error}")

    logging.error("Failed to download dataset from any source.")
    sys.exit(1)

if __name__ == "__main__":
    main()
