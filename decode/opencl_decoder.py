# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Image Decoder Script using VC-6 OpenCL based GPU Codec.

This script decodes images in a given directory (or a single file) into raw RGB format.
It uses the V-Nova VC-6 codec with OpenCL based GPU backend for decoding.

Features:
- Decodes each `.vc6` image into raw `.rgb` format.
"""

import os
import sys
import argparse
from typing import List

try:
    from vnova.vc6_opencl import codec as vc6codec
    from vnova.vc6_opencl import __version__ as vc6version
except ModuleNotFoundError:
    sys.exit(
        "Missing dependency: 'vnova.vc6_opencl'.\n"
        "This sample requires the VC-6 OpenCL Python SDK.\n"
        "Please refer README.md for install instructions.\n"
        "Please install them and re-run this program."
    )
else:
    print(f"VC-6 OpenCL SDK available : {vc6version}).")

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


def decode_images(image_list: List[str], max_width: int, max_height: int, batch: int, loq: int, dst_dir: str) -> None:
    """
    Decode a list of VC-6 images into Raw RGB format.

    Args:
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        batch (int): Batch Size.
        loq (int): Level of Quality.
        dst_dir (str): Output directory for decoded files.
    """
    try:
        num_images = len(image_list)
        input_bytes = []
        for path in image_list:
            with open(path, "rb") as input_file:
                input_bytes.append(input_file.read())
        # Initialise a batch decoder with OpenCL GPU backend with max height and max width and CPU memory type.
        vc6decoder = vc6codec.BatchDecoder(max_width, max_height, vc6codec.CodecBackendType.GPU, vc6codec.PictureFormat.RGB_8, vc6codec.ImageMemoryType.CPU, False, batch, num_images)
        decoded_images =  vc6decoder.decode(input_bytes, loq)
        for (path,img) in zip(image_list, decoded_images):
            basename = os.path.basename(path)
            decoded_name = str(basename.replace(os.path.splitext(basename)[1], ".rgb"))
            decoded_path = os.path.join(dst_dir, decoded_name)
            with open(decoded_path, "wb") as decoded_file:
                decoded_file.write(img.memoryview)
            print(f"Decoded to file : {decoded_path}")
    except Exception as e:
        print(f"Failed to decode {path}: {e}", file=sys.stderr)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Decode all VC-6 images in a directory (or single VC-6 image) to Raw RGB format.",
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
        "-b", "--batch",
        type=int, default=10,
        help="Batch size for decode (default: 10)"
    )
    parser.add_argument(
        "-l", "--loq",
        type=int, default=0,
        help="Level of Quality LOQ (default: 0)"
    )
    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Path to source directory containing images, or a single image file"
    )
    parser.add_argument(
        "-d", "--destination-dir",
        required=True,
        help="Directory to write decoded files"
    )
    return parser.parse_args()

def _is_vc6(path: str) -> bool:
    """
    Checks if the file is a VC-6 image based on its extension.

    Args:
        path (str): Path to the image file.
    """
    return os.path.splitext(path)[1].lower() == ".vc6"

def main() -> None:
    """Main function."""
    args = parse_arguments()

    print("Using VC-6 version:", vc6version)
    os.makedirs(args.destination_dir, exist_ok=True)

    if os.path.isfile(args.source):
        if not _is_vc6(args.source):
            print(f"Source file is not a .vc6 image: {args.source}", file=sys.stderr)
            sys.exit(1)
        image_list = [args.source]
    else:
        # get_input_paths should return files; we keep only .vc6
        image_list = [p for p in get_input_paths(args.source) if _is_vc6(p)]

    if not image_list:
        print(f"No .vc6 images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    decode_images(image_list, args.maxwidth, args.maxheight, args.batch, args.loq, args.destination_dir)


if __name__ == "__main__":
    main()