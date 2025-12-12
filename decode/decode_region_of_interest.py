# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Image Decoder Script using VC-6 CUDA based GPU Codec and extracts Region of interest.

This script decodes images in a given directory (or a single file) into raw RGB format.
It uses the V-Nova VC-6 codec with CUDA based GPU backend for decoding and extracts provided
Level of Quality and Region of Interest.

Features:
- Decodes each `.vc6` image into raw `.rgb` format for the provided Region of Interest(ROI).
"""

import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_codec,
    create_base_argument_parser,
    add_loq_argument,
    get_source_files,
    setup_output_directory,
    get_output_path
)


def decode_images(vc6codec, image_list: List[str], max_width: int, max_height: int,
                  loq: int, start_x: int, start_y: int, roi_w: int, roi_h: int,
                  dst_dir: str) -> None:
    """
    Decode a list of VC-6 images to the specified Region of Interest in the provided Level of Quality into Raw RGB format.

    Args:
        vc6codec: The loaded VC-6 codec module.
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        loq (int): Level of Quality.
        start_x (int): Start x-coordinate for Region of Interest.
        start_y (int): Start y-coordinate for Region of Interest.
        roi_w (int): Width for Region of Interest.
        roi_h (int): Height for Region of Interest.
        dst_dir (str): Output directory for decoded files.
    """
    try:
        # Set region for decoding
        region = vc6codec.FrameRegion(loq,
                                      start_x,
                                      start_y,
                                      roi_w,
                                      roi_h)

        # Initialise an async decoder with CUDA GPU backend
        vc6decoder = vc6codec.DecoderAsync(
            max_width,
            max_height,
            vc6codec.CodecBackendType.GPU,
            vc6codec.PictureFormat.RGB_8,
            vc6codec.ImageMemoryType.CPU,
            enable_logs=False
        )

        for path in image_list:
            input_bytes = None
            with open(path, "rb") as input_file:
                input_bytes = input_file.read()
            if len(input_bytes) < 1024:
                continue
            probe = vc6codec.ProbeFrame(input_bytes)
            if probe is None:
                print("Invalid vc6 file\n")
                continue
            width = probe.width
            height = probe.height
            # Decode only required Region of Interest at the provided Level of Quality
            vc6decoder.reconfigure(width, height, vc6codec.PictureFormat.RGB_8)
            decoded_image = vc6decoder.decode(input_bytes, region)
            decoded_path = get_output_path(path, dst_dir, ".rgb")
            with open(decoded_path, "wb") as decoded_file:
                decoded_file.write(decoded_image.memoryview)
            print(f"Decoded to file : {decoded_path}")
    except Exception as e:
        print(f"Failed to decode: {e}", file=sys.stderr)


def main() -> None:
    parser = create_base_argument_parser(
        "Decode all VC-6 images in a directory (or single VC-6 image) for provided Region of Interest to Raw RGB format."
    )
    add_loq_argument(parser)
    parser.add_argument(
        "-roix", "--roistartx",
        type=int, default=0,
        help="Start x-coordinate for Region of Interest. (default: 0)"
    )
    parser.add_argument(
        "-roiy", "--roistarty",
        type=int, default=0,
        help="Start y-coordinate for Region of Interest. (default: 0)"
    )
    parser.add_argument(
        "-roiw", "--roiwidth",
        type=int, default=224,
        help="Width for Region of Interest. (default: 224)"
    )
    parser.add_argument(
        "-roih", "--roiheight",
        type=int, default=224,
        help="Height for Region of Interest. (default: 224)"
    )
    args = parser.parse_args()

    vc6codec, vc6version, modname = load_codec("cpu")
    print(f"Using VC-6 version: {vc6version} via {modname}")

    setup_output_directory(args.destination_dir)

    image_list = get_source_files(args.source, filter_vc6=True)

    if not image_list:
        print(f"No .vc6 images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    decode_images(vc6codec, image_list, args.maxwidth, args.maxheight, args.loq,
                  args.roistartx, args.roistarty, args.roiwidth, args.roiheight,
                  args.destination_dir)


if __name__ == "__main__":
    main()
