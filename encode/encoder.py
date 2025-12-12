# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Image Encoder Script using VC-6 Codec.

This script encodes images in a given directory (or a single file) into the VC-6 format.
It uses the V-Nova VC-6 codec with a user-selectable backend (CPU, CUDA, or OpenCL).

Features:
- Encodes each image into `.vc6` format.
- Supports CPU, CUDA, and OpenCL backends.
"""

import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_codec,
    create_base_argument_parser,
    add_backend_argument,
    add_mode_argument,
    get_source_files,
    setup_output_directory,
    get_output_path
)

from PIL import Image


def encode_images(vc6codec, image_list: List[str], max_width: int, max_height: int,
                  dst_dir: str, use_gpu: bool, mode: str) -> None:
    """
    Encode a list of images into VC-6 format.

    Args:
        vc6codec: The loaded VC-6 codec module.
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        dst_dir (str): Output directory for encoded files.
        use_gpu (bool): Whether to use GPU backend (True for cuda/opencl, False for cpu).
        mode (str): Encoding mode - 'lossy' or 'lossless'.
    """
    backend_type = vc6codec.CodecBackendType.GPU if use_gpu else vc6codec.CodecBackendType.CPU
    vc6encoder = vc6codec.EncoderSync(
        max_width,
        max_height,
        backend_type,
        vc6codec.PictureFormat.RGB_8
    )

    if mode == "lossless":
        vc6encoder.set_generic_preset(vc6codec.EncoderGenericPreset.LOSSLESS)
        # Note: LIGHT profile is currently required for BatchDecoder_exp compatibility.
        # This requirement will be removed in a future SDK version.
        vc6encoder.set_profile_from_preset(vc6codec.EncoderProfilePreset.LIGHT)
    else:  # lossy
        vc6encoder.set_profile_from_preset(vc6codec.EncoderProfilePreset.BETTER)
        vc6encoder.set_quality_from_preset(vc6codec.EncoderQualityPreset.CBR_MULTIPASS, 3)

    for path in image_list:
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                w, h = img.size
                raw_data = img.tobytes()

            encoded_path = get_output_path(path, dst_dir, ".vc6")

            vc6encoder.reconfigure(w, h, vc6codec.PictureFormat.RGB_8)
            vc6encoder.write(raw_data, encoded_path)

            print(f"Encoded to file : {encoded_path}")
        except Exception as e:
            print(f"Failed to encode {path}: {e}", file=sys.stderr)


def main() -> None:
    """Main function."""
    parser = create_base_argument_parser(
        "Encode all images in a directory (or single image) to VC-6 format."
    )
    add_backend_argument(parser, default="cuda", choices=["cpu", "cuda", "opencl"])
    add_mode_argument(parser)
    args = parser.parse_args()

    vc6codec, vc6version, modname = load_codec(args.backend)
    print(f"Using VC-6 version: {vc6version} via {modname}")
    print(f"Encoding mode: {args.mode}")

    setup_output_directory(args.destination_dir)

    image_list = get_source_files(args.source)

    if not image_list:
        print(f"No valid images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    use_gpu = args.backend in ("cuda", "opencl")
    encode_images(vc6codec, image_list, args.maxwidth, args.maxheight,
                  args.destination_dir, use_gpu, args.mode)


if __name__ == "__main__":
    main()
