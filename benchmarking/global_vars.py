# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
import itertools
from collections import defaultdict
import logging

# Global variables for test_nvimagecodec_performance.py
batch_sizes = [64]
MAX_VC6_BATCH = 16
LOSSLESS = False
resize_dims = [(834, 834), (417, 417)]
resize_params = list(itertools.product(batch_sizes, resize_dims))

DATASET_DIR = "./huggingface"
RAW_FILES = DATASET_DIR + "/lossless" if LOSSLESS else DATASET_DIR + "/lossy"

TOTAL_IMAGES = 64

# Dataset to codec directory mapping
DATASET_MAPPING_LOSSY = {
    "V-NovaLtd/UHD-IQA-VC6": "VC-6",
    "V-NovaLtd/UHD-IQA-JPG": "JPEG",
    "V-NovaLtd/UHD-IQA-J2K": "J2K",
    "V-NovaLtd/UHD-IQA-JPH": "J2K_HT",
}

DATASET_MAPPING_LOSSLESS = {
    "V-NovaLtd/UHD-IQA-VC6-Lossless": "VC-6",
    "V-NovaLtd/UHD-IQA-J2K-Lossless": "J2K",
    "V-NovaLtd/UHD-IQA-JPH-Lossless": "J2K_HT",
}

CODEC_EXTENSIONS = {
    "VC-6": ".vc6",
    "JPEG": ".jpg",
    "J2K": ".jp2",
    "J2K_HT": ".jp2",
}

# Global number of batches for all tests
NUM_BATCHES = 10  # Set as needed
WARMUP_RUNS = 5

test_results = defaultdict(dict)
LOGGER = logging.getLogger("DECODE_PERF")
LOGGER.setLevel(logging.INFO)
