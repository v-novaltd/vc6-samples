# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
"""
Next iteration of the Batch Decoder.
"""

import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import cupy
except ModuleNotFoundError:
    sys.exit(
        "Missing dependency: 'cupy'.\n"
        "This sample requires CuPy for CUDA memory handling.\n"
        "Please install it with: pip install cupy-cuda12x\n"
        "Please install it and re-run this program."
    )

from utils import (
    load_codec,
    create_base_argument_parser,
    add_batch_size_argument,
    add_loq_argument,
    get_source_files,
    setup_output_directory,
    get_output_path
)

def decode_images(vc6codec, image_list: List[str], max_width: int, max_height: int, batch_size: int, loq: int, dst_dir: str) -> None:
    """
    Decode a list of VC-6 images in batches into Raw RGB format.

    Args:
        vc6codec: The VC-6 codec module.
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        batch_size (int): Number of parallel backends.
        loq (int): Level of Quality (echelon).
        dst_dir (str): Output directory for decoded files.
    """
    try:
        num_images = len(image_list)
        input_bytes = []

        print(f"Reading {num_images} input files...")
        for path in image_list:
            with open(path, "rb") as input_file:
                input_bytes.append(input_file.read())

        # Initialize BatchDecoder_exp with CUDA device memory type
        # Note: BatchDecoder_exp ONLY supports ImageMemoryType.CUDA_DEVICE for output
        print(f"Initializing BatchDecoder_exp with batch size {batch_size}...")
        vc6decoder = vc6codec.BatchDecoder_exp(
            init_frame_width=max_width,
            init_frame_height=max_height,
            codec_backend_type=vc6codec.CodecBackendType.GPU,
            picture_format=vc6codec.PictureFormat.RGB_8,
            output_memory_type=vc6codec.ImageMemoryType.CUDA_DEVICE,
            enable_logs=False,
            num_backends=batch_size,
            num_buffers=num_images
        )

        print(f"Decoding {num_images} images with LOQ={loq}...")
        decoded_images = vc6decoder.decode(input_bytes, loq)

        print("Copying decoded images from CUDA device memory to CPU and saving...")
        for (path, img) in zip(image_list, decoded_images):
            cuda_array = cupy.asarray(img)
            cpu_array = cuda_array.get()
            decoded_path = get_output_path(path, dst_dir, ".rgb")

            with open(decoded_path, "wb") as decoded_file:
                decoded_file.write(cpu_array.tobytes())

            print(f"Decoded to file : {decoded_path}")

    except Exception as e:
        print(f"Failed to decode: {e}", file=sys.stderr)
        raise


def main() -> None:
    parser = create_base_argument_parser(
        "Decode all VC-6 images in a directory to Raw RGB format using BatchDecoder_exp."
    )
    add_batch_size_argument(parser, default=4)
    add_loq_argument(parser)
    args = parser.parse_args()

    vc6codec, vc6version, modname = load_codec("cuda")
    print(f"Using VC-6 version: {vc6version} via {modname}")
    print(f"Using CuPy version: {cupy.__version__}")
    setup_output_directory(args.destination_dir)

    image_list = get_source_files(args.source, filter_vc6=True)

    if not image_list:
        print(f"No .vc6 images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    decode_images(vc6codec, image_list, args.maxwidth, args.maxheight, args.batch_size, args.loq, args.destination_dir)


if __name__ == "__main__":
    main()
