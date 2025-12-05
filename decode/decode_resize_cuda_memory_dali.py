# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Image Decoder Script using VC-6 CUDA based GPU Codec and resize the image with Nvidia DALI.

This script decodes images in a given directory (or a single file) into raw RGB format and 
then resizes the decoded images to a target resolution using Nvidia DALI.

It uses the V-Nova VC-6 codec with CUDA based GPU backend for decoding.

Features:
- Decodes each `.vc6` image into raw `.rgb` format and resizes the images using Nvidia DALI.
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
    
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.pipeline as pipeline
import cupy as cp


class ResizePipeline(pipeline.Pipeline):
    def __init__(self, batch_size, num_threads, device_id, decoded_images, width, height):
        super().__init__(batch_size, num_threads, device_id, seed=12)

        self.decoded_images = decoded_images

        self.input = fn.external_source(
            source=lambda: self.decoded_images,
            batch=True,
            device="gpu",
            dtype=types.UINT8,
            layout="HWC"
        )

        self.resize = fn.resize(self.input, size=(width, height))

    def define_graph(self):
        return self.resize

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


def decode_resize_images(image_list: List[str], max_width: int, max_height: int, batch: int, loq: int, resize_width: int, resize_height: int, dst_dir: str) -> None:
    """
    Decode a list of VC-6 images into Raw RGB format and pass the CUDA memory to Nvidia DALI to resize.

    Args:
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        batch (int): Batch Size.
        loq (int): Level of Quality.
        resize_width (int): Width to be resized to.
        resize_height (int): Height to be resized to.
        dst_dir (str): Output directory for decoded files.
    """
    try:
        num_images = len(image_list)
        input_bytes = []
        for path in image_list:
            with open(path, "rb") as input_file:
                input_bytes.append(input_file.read())
        # Initialise a batch decoder with CUDA GPU backend with max height and max width and CUDA memory type.
        vc6decoder = vc6codec.BatchDecoder(max_width, max_height, vc6codec.CodecBackendType.GPU, vc6codec.PictureFormat.RGB_8, vc6codec.ImageMemoryType.CUDA_DEVICE, False, batch, num_images)
        decoded_images =  vc6decoder.decode(input_bytes, loq)
        decoded_cuda_buffers = []
        
        for i, cudaMem in enumerate(decoded_images):
            probe_data = input_bytes[i][:1024]
            if len(probe_data) != 1024:
                continue
            probe = vc6codec.ProbeFrame(probe_data)
            if probe is None:
                print("Invalid vc6 file\n")
                continue
            width = probe.width
            height = probe.height
            arr = cp.asarray(cudaMem)
            for i in range(loq):
                width = (width + 1) // 2
                height = (height + 1) // 2
            arr = arr.reshape(height, width, 3)
            decoded_cuda_buffers.append(arr)
        # DALI Resize pipeline.
        resize_pipeline = ResizePipeline(batch_size=num_images, num_threads=8, device_id=0, decoded_images=decoded_cuda_buffers, width=resize_width, height=resize_height)
        resize_pipeline.build()
        resized_images = resize_pipeline.run()
        resized_images_cpu = resized_images[0].as_cpu().as_array()  # shape: (BATCH_SIZE, H, W, 3)

        for idx, img in enumerate(resized_images_cpu, start=0):
            basename = os.path.basename(image_list[idx])
            decoded_name = str(basename.replace(os.path.splitext(basename)[1], ".rgb"))
            decoded_path = os.path.join(dst_dir, decoded_name)
            # Save raw RGB bytes
            img.tofile(decoded_path)
            print(f"Decoded and resized to file : {decoded_path}")
    except Exception as e:
        print(f"Failed to decode {path}: {e}", file=sys.stderr)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Decode all VC-6 images in a directory (or single VC-6 image) to Raw RGB format and resize using Nvidia DALI.",
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
        "-rw", "--resizewidth",
        type=int, default=224,
        help="Resize Width (default: 224)"
    )
    parser.add_argument(
        "-rh", "--resizeheight",
        type=int, default=224,
        help="Resize height (default: 224)"
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

    decode_resize_images(image_list, args.maxwidth, args.maxheight, args.batch, args.loq, args.resizewidth, args.resizeheight, args.destination_dir)


if __name__ == "__main__":
    main()