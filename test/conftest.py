import os
import sys
import glob
import pytest
from typing import NamedTuple

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "encode"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "decode"))

import encoder
import decoder
import batch_encoder
import batch_decoder_experimental


# =============================================================================
# Variant definitions using NamedTuples for clarity
# =============================================================================

class EncoderVariant(NamedTuple):
    name: str
    module: object
    supports_cpu: bool
    supports_lossless: bool


class DecoderVariant(NamedTuple):
    name: str
    module: object
    supports_cpu: bool
    cuda_only: bool


ENCODER_VARIANTS = [
    EncoderVariant("EncoderSync", encoder, supports_cpu=True, supports_lossless=True),
    EncoderVariant("BatchEncoder", batch_encoder, supports_cpu=False, supports_lossless=True),
]

DECODER_VARIANTS = [
    DecoderVariant("BatchDecoder", decoder, supports_cpu=True, cuda_only=False),
    DecoderVariant("BatchDecoder_exp", batch_decoder_experimental, supports_cpu=False, cuda_only=True),
]

ALL_BACKENDS = ["cuda", "opencl", "cpu"]
GPU_BACKENDS = ["cuda", "opencl"]


# =============================================================================
# Pytest configuration
# =============================================================================

def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        required=True,
        help="Path to folder containing input PNG images for testing",
    )


def pytest_generate_tests(metafunc):
    if "input_image" in metafunc.fixturenames:
        input_dir = metafunc.config.getoption("input")
        if not os.path.isdir(input_dir):
            pytest.fail(f"Input directory does not exist: {input_dir}")

        png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
        if not png_files:
            pytest.fail(f"No PNG files found in input directory: {input_dir}")

        ids = [os.path.basename(f) for f in png_files]
        metafunc.parametrize("input_image", png_files, ids=ids)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory."""
    return str(tmp_path)
