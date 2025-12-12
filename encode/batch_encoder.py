# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Batch Image Encoder Script using VC-6 Codec.
"""

import sys
from pathlib import Path
from typing import List, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_codec,
    create_base_argument_parser,
    add_batch_size_argument,
    add_backend_argument,
    add_mode_argument,
    get_source_files,
    setup_output_directory,
    get_output_path
)

from PIL import Image


def load_images(image_list: List[str]) -> Tuple[List[Tuple[bytes, Tuple[int, int]]], List[str]]:
    """
    Load all images into memory and prepare them for batch encoding.

    Args:
        image_list (List[str]): List of image file paths.

    Returns:
        Tuple containing:
        - List[Tuple[bytes, Tuple[int, int]]]: List of tuples containing (raw_bytes, (width, height)).
        - List[str]: List of successfully loaded file paths (for output naming).
    """
    loaded_images = []
    loaded_paths = []

    for path in image_list:
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                w, h = img.size
                raw_data = img.tobytes()
                loaded_images.append((raw_data, (w, h)))
                loaded_paths.append(path)
                print(f"Loaded image: {path} ({w}x{h})")
        except Exception as e:
            print(f"Failed to load {path}: {e}", file=sys.stderr)

    return loaded_images, loaded_paths


def batch_encode_images(
    vc6codec,
    image_list: List[str],
    max_width: int,
    max_height: int,
    batch_size: int,
    dst_dir: str,
    mode: str
) -> None:
    """
    Batch encode a list of images into VC-6 format using parallel encoding.

    Args:
        vc6codec: The loaded VC-6 codec module.
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        batch_size (int): Number of parallel encoder backends.
        dst_dir (str): Output directory for encoded files.
        mode (str): Encoding mode - 'lossy' or 'lossless'.
    """
    if not image_list:
        print("No images to encode.", file=sys.stderr)
        return

    print(f"Loading {len(image_list)} images into memory...")

    loaded_images, loaded_paths = load_images(image_list)

    if not loaded_images:
        print("No images were successfully loaded.", file=sys.stderr)
        return

    print(f"Initializing BatchEncoder with {batch_size} parallel backends...")
    print(f"Max dimensions: {max_width}x{max_height}")

    # Note: BatchEncoder only supports CPU memory type for input
    num_images = len(loaded_images)
    vc6encoder = vc6codec.BatchEncoder(
        max_frame_width=max_width,
        max_frame_height=max_height,
        codec_backend_type=vc6codec.CodecBackendType.GPU,
        picture_format=vc6codec.PictureFormat.RGB_8,
        input_memory_type=vc6codec.ImageMemoryType.CPU,
        enable_logs=False,
        num_backends=batch_size,
        num_buffers=num_images
    )

    if mode == "lossless":
        vc6encoder.set_generic_preset(vc6codec.EncoderGenericPreset.LOSSLESS)
        # Note: LIGHT profile is currently required for BatchDecoder_exp compatibility.
        # This requirement will be removed in a future SDK version.
        vc6encoder.set_profile_from_preset(vc6codec.EncoderProfilePreset.LIGHT)
    else:  # lossy
        vc6encoder.set_profile_from_preset(vc6codec.EncoderProfilePreset.BETTER)
        vc6encoder.set_quality_from_preset(vc6codec.EncoderQualityPreset.CBR_MULTIPASS, 3.0)

    print(f"Batch encoding {len(loaded_images)} images...")

    try:
        encoded_buffers = vc6encoder.encode(loaded_images)

        print("Writing encoded files...")
        for path, encoded_buffer in zip(loaded_paths, encoded_buffers):
            try:
                encoded_path = get_output_path(path, dst_dir, ".vc6")
                with open(encoded_path, 'wb') as f:
                    f.write(encoded_buffer.memoryview)

                print(f"Encoded to file : {encoded_path}")
            except Exception as e:
                print(f"Failed to write encoded file for {path}: {e}", file=sys.stderr)

        print(f"Batch encoding completed. Total images encoded: {len(encoded_buffers)}")

    except Exception as e:
        print(f"Batch encoding failed: {e}", file=sys.stderr)
        raise


def main() -> None:
    parser = create_base_argument_parser(
        "Batch encode all images in a directory (or single image) to VC-6 format using parallel encoding."
    )
    add_batch_size_argument(parser, default=4)
    add_backend_argument(parser, default="cuda", choices=["cuda", "opencl"])
    add_mode_argument(parser)
    args = parser.parse_args()

    vc6codec, vc6version, modname = load_codec(args.backend)
    print(f"Using VC-6 BatchEncoder via {modname} (version {vc6version})")
    print(f"Encoding mode: {args.mode}")

    setup_output_directory(args.destination_dir)

    image_list = get_source_files(args.source)

    if not image_list:
        print(f"No valid images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    batch_encode_images(
        vc6codec,
        image_list,
        args.maxwidth,
        args.maxheight,
        args.batch_size,
        args.destination_dir,
        args.mode
    )


if __name__ == "__main__":
    main()
