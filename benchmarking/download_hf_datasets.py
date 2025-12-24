# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
"""Download HuggingFace datasets and organize files by codec."""
from pathlib import Path
from datasets import load_dataset
import logging
import os
import requests
import tarfile
import tempfile
from global_vars import LOSSLESS, RAW_FILES, TOTAL_IMAGES, DATASET_MAPPING_LOSSY, DATASET_MAPPING_LOSSLESS, CODEC_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dataset_mapping():
    """Get the appropriate dataset mapping based on LOSSLESS flag."""
    return DATASET_MAPPING_LOSSLESS if LOSSLESS else DATASET_MAPPING_LOSSY


def get_tar_path(url: str, tar_cache: dict) -> str:
    """Get local path to tar file, downloading if necessary."""
    if url in tar_cache:
        return tar_cache[url]
    
    if os.path.exists(url):
        tar_cache[url] = url
        return url
    
    logger.info(f"Downloading tar: {url}")
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tar')
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    temp_file.close()
    tar_cache[url] = temp_file.name
    return temp_file.name


def extract_file_from_tar(tar_path: str, file_key: str, expected_ext: str = "") -> bytes:
    """Extract a specific file from a tar archive."""
    with tarfile.open(tar_path, 'r:*') as tar:
        members = tar.getnames()
        logger.debug(f"Looking for: {file_key} (ext: {expected_ext})")
        
        # Try exact match first
        if file_key in members:
            return tar.extractfile(file_key).read()
        
        # Try with expected extension appended
        if expected_ext and not file_key.endswith(expected_ext):
            key_with_ext = f"{file_key}{expected_ext}"
            if key_with_ext in members:
                return tar.extractfile(key_with_ext).read()
        
        # Try common extensions if expected_ext not provided
        for ext in ['.jpg', '.jpeg', '.vc6', '.jp2']:
            if not file_key.endswith(ext):
                key_with_ext = f"{file_key}{ext}"
                if key_with_ext in members:
                    return tar.extractfile(key_with_ext).read()
        
        # Try to find by filename only (without path)
        filename = Path(file_key).name
        for member_name in members:
            member_filename = Path(member_name).name
            if member_filename == filename or member_name.endswith(f"/{filename}"):
                return tar.extractfile(member_name).read()
        
        # Try filename with extension
        if expected_ext:
            filename_with_ext = f"{filename}{expected_ext}"
            for member_name in members:
                if Path(member_name).name == filename_with_ext:
                    return tar.extractfile(member_name).read()
        
        # Log available files for debugging
        logger.error(f"File {file_key} not found in tar. Available files (first 20): {members[:20]}")
        raise FileNotFoundError(f"File {file_key} not found in tar {tar_path}")


def download_dataset_files(dataset_name: str, codec_dir: Path, split: str, target_base_filenames: dict = None):
    """Download files from a HuggingFace dataset to the codec directory.
    
    Args:
        dataset_name: Name of the dataset
        codec_dir: Directory to save files
        split: Dataset split to use
        target_base_filenames: If provided, set of base filenames to download.
                                If None, download first TOTAL_IMAGES and return mapping.
    """
    codec_dir.mkdir(parents=True, exist_ok=True)
    
    codec_name = get_dataset_mapping()[dataset_name]
    expected_ext = CODEC_EXTENSIONS.get(codec_name, "")
    
    # Check if all required files already exist - skip dataset loading if so
    if target_base_filenames is not None:
        # For subsequent codecs: check if all target files exist
        existing_files = {f.stem for f in codec_dir.iterdir() 
                         if f.is_file() and f.suffix == expected_ext}
        missing_files = target_base_filenames - existing_files
        if not missing_files:
            logger.info(f"All {len(target_base_filenames)} required files already exist in {codec_dir}. Skipping download.")
            return None
        logger.info(f"Found {len(existing_files)}/{len(target_base_filenames)} files. Need to download {len(missing_files)} files.")
        # Only look for missing files
        target_base_filenames = missing_files
    else:
        # For first codec: check if we already have TOTAL_IMAGES files
        existing_files = [f for f in codec_dir.iterdir() 
                         if f.is_file() and f.suffix == expected_ext]
        if len(existing_files) >= TOTAL_IMAGES:
            logger.info(f"Already have {len(existing_files)} files (need {TOTAL_IMAGES}) in {codec_dir}. Collecting base filenames from existing files.")
            # Collect base filenames from existing files (use filename as key since we don't have original keys)
            base_filename_to_key = {}
            for f in sorted(existing_files)[:TOTAL_IMAGES]:
                base_filename = f.stem
                base_filename_to_key[base_filename] = base_filename
            return base_filename_to_key
    
    # Only load dataset if we need to download files
    logger.info(f"Loading dataset {dataset_name} with split '{split}'...")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Downloading files to {codec_dir}...")
    
    tar_cache = {}
    downloaded_count = 0
    skipped_count = 0
    base_filename_to_key = {}  # base_filename -> key (for first codec)
    
    try:
        items_processed = 0
        for idx, item in enumerate(dataset):
            items_processed += 1
            
            # If target_base_filenames provided, check if we've found all files
            if target_base_filenames is not None:
                # Check how many files we still need
                existing_now = {f.stem for f in codec_dir.iterdir() 
                               if f.is_file() and f.suffix == expected_ext}
                still_needed = target_base_filenames - existing_now
                if not still_needed:
                    logger.info(f"All required files found. Stopping iteration after {items_processed} items.")
                    break
                
                # Log progress when searching for matching files
                if items_processed % 100 == 0:
                    logger.info(f"Processed {items_processed} items, found {len(existing_now)}/{len(target_base_filenames)} files for {codec_name}")
            
            if '__url__' not in item or '__key__' not in item:
                logger.warning(f"Item {idx} missing __url__ or __key__, skipping")
                continue
            
            url = item['__url__']
            key = item['__key__']
            
            # Extract base filename from key (remove path and extension)
            base_filename = Path(key).stem
            
            # If target_base_filenames provided, only process matching base filenames
            if target_base_filenames is not None:
                if base_filename not in target_base_filenames:
                    continue
                logger.debug(f"Found matching file: {base_filename} -> {key}")
            else:
                # First codec: collect base filenames from first TOTAL_IMAGES
                if len(base_filename_to_key) >= TOTAL_IMAGES:
                    break
                base_filename_to_key[base_filename] = key
            
            # Construct filename with correct extension
            filename = f"{base_filename}{expected_ext}"
            output_path = codec_dir / filename
            
            # Skip if file already exists
            if output_path.exists():
                skipped_count += 1
                continue
            
            # Extract from tar
            if url.endswith('.tar') or '.tar' in url:
                logger.info(f"Extracting {key} from tar for {codec_name}...")
                tar_path = get_tar_path(url, tar_cache)
                file_data = extract_file_from_tar(tar_path, key, expected_ext)
                output_path.write_bytes(file_data)
                downloaded_count += 1
                
                if (downloaded_count + skipped_count) % 10 == 0:
                    logger.info(f"Progress: {downloaded_count} downloaded, {skipped_count} skipped for {codec_name}")
            else:
                logger.warning(f"Item {idx} URL is not a tar file: {url}")
        
        logger.info(f"Finished processing {items_processed} items for {codec_name}")
    finally:
        # Clean up temporary tar files
        for temp_tar_path in tar_cache.values():
            if temp_tar_path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(temp_tar_path)
                except Exception:
                    pass
    
    logger.info(f"Completed {dataset_name}: {downloaded_count} downloaded, {skipped_count} skipped")
    
    # Return base_filename_to_key mapping if this was the first codec
    if target_base_filenames is None:
        return base_filename_to_key
    return None


def check_datasets_exist(base_dir: Path, dataset_mapping: dict) -> bool:
    """Check if all codec directories exist and have at least TOTAL_IMAGES files."""
    for dataset_name, codec_name in dataset_mapping.items():
        codec_dir = base_dir / codec_name
        expected_ext = CODEC_EXTENSIONS.get(codec_name, "")
        
        if not codec_dir.exists():
            logger.info(f"Directory {codec_dir} does not exist. Download needed.")
            return False
        
        # Count files with expected extension
        file_count = sum(1 for f in codec_dir.iterdir() 
                        if f.is_file() and f.suffix == expected_ext)
        
        if file_count < TOTAL_IMAGES:
            logger.info(f"Directory {codec_dir} has only {file_count} files, need {TOTAL_IMAGES}. Download needed.")
            return False
    
    logger.info(f"All codec directories exist with at least {TOTAL_IMAGES} files. Skipping download.")
    return True


def download_all_datasets():
    """Download all datasets and organize by codec."""
    base_dir = Path(RAW_FILES)
    split = "test"
    dataset_mapping = get_dataset_mapping()
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Using split: {split}")
    logger.info(f"LOSSLESS mode: {LOSSLESS}")
    logger.info(f"Total images limit: {TOTAL_IMAGES}")
    
    # Check if datasets already exist
    if check_datasets_exist(base_dir, dataset_mapping):
        return
    
    # First pass: Download from first codec and collect base filenames
    first_dataset = list(dataset_mapping.items())[0]
    dataset_name, codec_name = first_dataset
    codec_dir = base_dir / codec_name
    logger.info(f"\nProcessing first codec {dataset_name} -> {codec_dir} (collecting base filenames)...")
    base_filename_to_key = download_dataset_files(dataset_name, codec_dir, split, target_base_filenames=None)
    
    target_base_filenames = set(base_filename_to_key.keys())
    logger.info(f"Collected {len(target_base_filenames)} files from first codec. Using same base filenames for all other codecs.")
    
    # Second pass: Download files with matching base filenames from all other codecs
    for dataset_name, codec_name in list(dataset_mapping.items())[1:]:
        codec_dir = base_dir / codec_name
        logger.info(f"\nProcessing {dataset_name} -> {codec_dir}")
        download_dataset_files(dataset_name, codec_dir, split, target_base_filenames=target_base_filenames)
    
    logger.info(f"\nAll downloads complete. Files are in: {base_dir}")


def main():
    """Entry point for standalone execution."""
    download_all_datasets()


if __name__ == "__main__":
    main()
