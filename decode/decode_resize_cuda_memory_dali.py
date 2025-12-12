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

import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.pipeline as pipeline
import cupy as cp

from utils import (
    load_codec,
    create_base_argument_parser,
    add_batch_size_argument,
    add_loq_argument,
    get_source_files,
    setup_output_directory,
    get_output_path
)

# Load CUDA codec (required for DALI integration)
vc6codec, vc6version, _ = load_codec("cuda")
print(f"VC-6 CUDA SDK available : {vc6version}")


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


def decode_resize_images(image_list: List[str], max_width: int, max_height: int,
                         batch_size: int, loq: int, resize_width: int, resize_height: int,
                         dst_dir: str) -> None:
    """
    Decode a list of VC-6 images into Raw RGB format and pass the CUDA memory to Nvidia DALI to resize.

    Args:
        image_list (List[str]): List of image file paths.
        max_width (int): Maximum image width.
        max_height (int): Maximum image height.
        batch_size (int): Batch Size.
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
        vc6decoder = vc6codec.BatchDecoder(
            max_width,
            max_height,
            vc6codec.CodecBackendType.GPU,
            vc6codec.PictureFormat.RGB_8,
            vc6codec.ImageMemoryType.CUDA_DEVICE,
            False,
            batch_size,
            num_images
        )

        decoded_images = vc6decoder.decode(input_bytes, loq)
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
            for _ in range(loq):
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
            decoded_path = get_output_path(image_list[idx], dst_dir, ".rgb")
            # Save raw RGB bytes
            img.tofile(decoded_path)
            print(f"Decoded and resized to file : {decoded_path}")
    except Exception as e:
        print(f"Failed to decode: {e}", file=sys.stderr)


def main() -> None:
    parser = create_base_argument_parser(
        "Decode all VC-6 images in a directory (or single VC-6 image) to Raw RGB format and resize using Nvidia DALI."
    )
    add_batch_size_argument(parser, default=10)
    add_loq_argument(parser)
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
    args = parser.parse_args()

    print("Using VC-6 version:", vc6version)

    setup_output_directory(args.destination_dir)

    image_list = get_source_files(args.source, filter_vc6=True)

    if not image_list:
        print(f"No .vc6 images found at: {args.source}", file=sys.stderr)
        sys.exit(1)

    decode_resize_images(image_list, args.maxwidth, args.maxheight, args.batch_size,
                         args.loq, args.resizewidth, args.resizeheight,
                         args.destination_dir)


if __name__ == "__main__":
    main()
