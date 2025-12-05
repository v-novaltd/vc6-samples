# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Image Encoder Script using VC-6 CUDA based GPU Codec.

This script encodes images in a given directory (or a single file) into the VC-6 format.
It uses the V-Nova VC-6 codec with CUDA GPU backend for encoding.

Features:
- Encodes each image into `.vc6` format.
"""

import os
import sys
import argparse
from typing import List

try:
    from vnova.vc6_cu12 import codec as vc6codec
    from vnova.vc6_cu12 import __version__ as vc6version
except ModuleNotFoundError:
    sys.exit(
        "Missing dependency: 'vnova.vc6_cu12'.\n"
        "This sample requires the VC-6 CUDA Python SDK.\n"
        "Please refer README.md for install instructions.\n"
        "Please install them and re-run this program."
    )
else:
    print(f"VC-6 CUDA SDK available : {vc6version}")

from PIL import Image

def get_input_paths(root: str) -> List[str]:
    """
    Get all file paths in a directory.

    Args:
        root (str): Path to the directory containing images.

    Returns:
        List[str]: List of file paths.
    """
    try:
        _, _, files = next(os.walk(root))
        return [os.path.join(root, fn) for fn in files if os.path.isfile(os.path.join(root, fn))]
    except StopIteration:
        print(f"Provided directory is empty or invalid: {root}", file=sys.stderr)
        return []


def encode_images(image_list: List[str], max_width: int, max_height: int, dst_dir: str) -> None:
    """
    Encode a list of images into VC-6 format.

    Args:
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        dst_dir (str): Output directory for encoded files.
    """
    # Initialise a synchronous encoder with CUDA GPU backend with max height and max width and CPU memory type.
    vc6encoder = vc6codec.EncoderSync(
        max_width,
        max_height,
        vc6codec.CodecBackendType.GPU,
        vc6codec.PictureFormat.RGB_8
    )
    # Set VC-6 Encode parameters
    vc6encoder.set_profile_from_preset(vc6codec.EncoderProfilePreset.BETTER)
    vc6encoder.set_quality_from_preset(vc6codec.EncoderQualityPreset.CBR_MULTIPASS, 3)

    for path in image_list:
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                w, h = img.size
                raw_data = img.tobytes()

            basename = os.path.basename(path)
            encoded_name = os.path.splitext(basename)[0] + ".vc6"
            encoded_path = os.path.join(dst_dir, encoded_name)

            # Reconfigure and encode
            vc6encoder.reconfigure(w, h, vc6codec.PictureFormat.RGB_8)
            vc6encoder.write(raw_data, encoded_path)

            print(f"Encoded to file : {encoded_path}")
        except Exception as e:
            print(f"Failed to encode {path}: {e}", file=sys.stderr)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Encode all images in a directory (or single image) to VC-6 format.",
        add_help=False
    )
    parser.add_argument(
        "--help", action="help", help="Show this help message and exit"
    )
    parser.add_argument(
        "-w", "--maxwidth",
        type=int, default=2048,
        help="Maximum width of images (default: 2048)"
    )
    parser.add_argument(
        "-h", "--maxheight",
        type=int, default=2048,
        help="Maximum height of images (default: 2048)"
    )
    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Path to source directory containing images, or a single image file"
    )
    parser.add_argument(
        "-d", "--destination-dir",
        required=True,
        help="Directory to write encoded files"
    )
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_arguments()

    print("Using VC-6 version:", vc6version)
    os.makedirs(args.destination_dir, exist_ok=True)

    if os.path.isfile(args.source):
        image_list = [args.source]
    else:
        image_list = get_input_paths(args.source)

    if not image_list:
        print(f"No valid images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    encode_images(image_list, args.maxwidth, args.maxheight, args.destination_dir)


if __name__ == "__main__":
    main()