# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    load_codec,
    create_base_argument_parser,
    add_batch_size_argument,
    add_loq_argument,
    add_backend_argument,
    get_source_files,
    setup_output_directory,
    get_output_path
)


def decode_images(vc6codec, image_list: List[str], max_width: int, max_height: int,
                  batch_size: int, loq: int, dst_dir: str, use_gpu: bool) -> None:
    """
    Decode a list of VC-6 images in batches into Raw RGB format.

    Args:
        vc6codec: The loaded VC-6 codec module.
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        batch_size (int): Batch Size.
        loq (int): Level of Quality.
        dst_dir (str): Output directory for decoded files.
        use_gpu (bool): Whether to use GPU backend (True for cuda/opencl, False for cpu).
    """
    try:
        num_images = len(image_list)
        input_bytes = []
        for path in image_list:
            with open(path, "rb") as input_file:
                input_bytes.append(input_file.read())

        vc6decoder = vc6codec.BatchDecoder(
            max_width,
            max_height,
            vc6codec.CodecBackendType.GPU if use_gpu else vc6codec.CodecBackendType.CPU,
            vc6codec.PictureFormat.RGB_8,
            vc6codec.ImageMemoryType.CPU,
            False,
            batch_size,
            num_images
        )

        decoded_images = vc6decoder.decode(input_bytes, loq)
        for (path, img) in zip(image_list, decoded_images):
            decoded_path = get_output_path(path, dst_dir, ".rgb")
            with open(decoded_path, "wb") as decoded_file:
                decoded_file.write(img.memoryview)
            print(f"Decoded to file : {decoded_path}")
    except Exception as e:
        print(f"Failed to decode: {e}", file=sys.stderr)


def main() -> None:
    """Main function."""
    parser = create_base_argument_parser(
        "Decode all VC-6 images in a directory to Raw RGB format."
    )
    add_batch_size_argument(parser, default=10)
    add_loq_argument(parser)
    add_backend_argument(parser, default="cuda", choices=["cpu", "cuda", "opencl"])
    args = parser.parse_args()

    vc6codec, vc6version, modname = load_codec(args.backend)
    print(f"Using VC-6 version: {vc6version} via {modname}")

    setup_output_directory(args.destination_dir)

    image_list = get_source_files(args.source, filter_vc6=True)

    if not image_list:
        print(f"No .vc6 images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    use_gpu = args.backend in ("cuda", "opencl")
    decode_images(vc6codec, image_list, args.maxwidth, args.maxheight,
                  args.batch_size, args.loq, args.destination_dir, use_gpu)


if __name__ == "__main__":
    main()
