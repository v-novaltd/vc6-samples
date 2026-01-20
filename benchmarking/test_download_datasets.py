# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
"""Test cases for downloading HuggingFace datasets."""
import argparse
import sys
from pathlib import Path

import pytest
from download_hf_datasets import (
    get_dataset_mapping,
    download_dataset_files,
    check_datasets_exist,
    download_all_datasets,
)
from global_vars import RAW_FILES, TOTAL_IMAGES, CODEC_EXTENSIONS


@pytest.fixture(scope="session")
def base_filename_mapping():
    """Download first codec and return base filename mapping for other codecs."""
    base_dir = Path(RAW_FILES)
    split = "test"
    dataset_mapping = get_dataset_mapping()
    
    # Check if datasets already exist
    if check_datasets_exist(base_dir, dataset_mapping):
        pytest.skip("All datasets already exist")
    
    # Download first codec and collect base filenames
    first_dataset = list(dataset_mapping.items())[0]
    dataset_name, codec_name = first_dataset
    codec_dir = base_dir / codec_name
    
    base_filename_to_key = download_dataset_files(
        dataset_name, codec_dir, split, target_base_filenames=None
    )
    
    return set(base_filename_to_key.keys())


class TestDownloadDatasets:
    """Test cases for downloading datasets for each codec."""
    
    @pytest.mark.parametrize("codec_name", ["VC-6", "JPEG", "J2K", "J2K_HT"])
    def test_download_codec_dataset(self, codec_name, base_filename_mapping):
        """Download dataset for a specific codec."""
        base_dir = Path(RAW_FILES)
        split = "test"
        dataset_mapping = get_dataset_mapping()
        
        # Find dataset name for this codec
        dataset_name = None
        for ds_name, cd_name in dataset_mapping.items():
            if cd_name == codec_name:
                dataset_name = ds_name
                break
        
        if dataset_name is None:
            pytest.skip(f"No dataset found for codec {codec_name}")
        
        codec_dir = base_dir / codec_name
        expected_ext = CODEC_EXTENSIONS.get(codec_name, "")
        
        # Skip if already downloaded
        if codec_dir.exists():
            files = [f for f in codec_dir.iterdir() if f.is_file() and f.suffix == expected_ext]
            if len(files) >= TOTAL_IMAGES:
                pytest.skip(f"{codec_name} dataset already exists with {len(files)} files")
        
        # Download files
        first_codec_name = list(dataset_mapping.values())[0]
        if codec_name == first_codec_name:
            # First codec - already downloaded in fixture, just verify
            result = base_filename_mapping
            assert result is not None and len(result) > 0, f"Failed to download {codec_name} dataset"
        else:
            # Other codecs - use base filenames from first codec
            result = download_dataset_files(
                dataset_name, codec_dir, split, target_base_filenames=base_filename_mapping
            )
            assert result is None, f"Failed to download {codec_name} dataset"
        
        # Verify files were downloaded
        assert codec_dir.exists(), f"Codec directory {codec_dir} was not created"
        files = [f for f in codec_dir.iterdir() if f.is_file() and f.suffix == expected_ext]
        assert len(files) >= TOTAL_IMAGES, (
            f"Expected at least {TOTAL_IMAGES} {codec_name} files, found {len(files)}"
        )


def _verify_downloads():
    base_dir = Path(RAW_FILES)
    dataset_mapping = get_dataset_mapping()
    failures = []
    for dataset_name, codec_name in dataset_mapping.items():
        codec_dir = base_dir / codec_name
        expected_ext = CODEC_EXTENSIONS.get(codec_name, "")
        if not codec_dir.exists():
            failures.append(f"Missing directory for {codec_name}: {codec_dir}")
            continue
        file_count = sum(
            1 for f in codec_dir.iterdir()
            if f.is_file() and f.suffix == expected_ext
        )
        if file_count < TOTAL_IMAGES:
            failures.append(
                f"{codec_name} has {file_count} files, expected at least {TOTAL_IMAGES}"
            )
    if failures:
        raise RuntimeError("Dataset download verification failed:\n- " + "\n- ".join(failures))


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark datasets and verify counts.",
        add_help=True,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output for downloads."
    )
    args, _ = parser.parse_known_args()
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.INFO)

    try:
        download_all_datasets()
        _verify_downloads()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
