"""Helper functions for VC-6 tests."""

import os
import pytest
import numpy as np
from PIL import Image

import utils


def get_image_dimensions(image_path):
    """Get width and height of an image."""
    with Image.open(image_path) as img:
        return img.width, img.height


def load_codec_or_skip(backend):
    """Load codec for given backend, skip test if not available."""
    try:
        return utils.load_codec(backend)
    except SystemExit:
        pytest.skip(f"{backend} backend not available")


def check_cupy_or_skip():
    """Check if cupy is available, skip test if not."""
    try:
        import cupy
        return cupy
    except ModuleNotFoundError:
        pytest.skip("cupy not available for BatchDecoder_exp")


def encode_image(vc6codec, encoder, image_path, width, height, encoded_dir, use_gpu, mode):
    """
    Encode an image using the specified encoder variant.

    Returns:
        Path to encoded file.
    """
    max_width = width + 2000
    max_height = height + 2000

    if encoder.name == "EncoderSync":
        encoder.module.encode_images(
            vc6codec=vc6codec,
            image_list=[image_path],
            max_width=max_width,
            max_height=max_height,
            dst_dir=encoded_dir,
            use_gpu=use_gpu,
            mode=mode
        )
    elif encoder.name == "BatchEncoder":
        encoder.module.batch_encode_images(
            vc6codec=vc6codec,
            image_list=[image_path],
            max_width=max_width,
            max_height=max_height,
            batch_size=1,
            dst_dir=encoded_dir,
            mode=mode
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder.name}")

    return utils.get_output_path(image_path, encoded_dir, ".vc6")


def decode_image(vc6codec, decoder, encoded_path, width, height, decoded_dir, use_gpu):
    """
    Decode a VC-6 image using the specified decoder variant.

    Returns:
        Path to decoded file.
    """
    max_width = max(width + 128, 2048)
    max_height = max(height + 128, 2048)

    if decoder.name == "BatchDecoder":
        decoder.module.decode_images(
            vc6codec=vc6codec,
            image_list=[encoded_path],
            max_width=max_width,
            max_height=max_height,
            batch_size=1,
            loq=0,
            dst_dir=decoded_dir,
            use_gpu=use_gpu
        )
    elif decoder.name == "BatchDecoder_exp":
        decoder.module.decode_images(
            vc6codec=vc6codec,
            image_list=[encoded_path],
            max_width=max_width,
            max_height=max_height,
            batch_size=1,
            loq=0,
            dst_dir=decoded_dir
        )
    else:
        raise ValueError(f"Unknown decoder: {decoder.name}")

    return utils.get_output_path(encoded_path, decoded_dir, ".rgb")


def compare_image_to_raw(original_image_path, decoded_raw_path):
    """
    Compare an original image to a decoded raw RGB file.

    Returns:
        True if images are identical, False otherwise.
    """
    decoded_arr = np.fromfile(decoded_raw_path, dtype=np.uint8)
    with Image.open(original_image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        original_arr = np.asarray(img).flatten()

    return np.array_equal(decoded_arr, original_arr)
