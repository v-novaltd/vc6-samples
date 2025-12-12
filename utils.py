# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
"""
Common utilities for VC-6 sample scripts.

This module provides shared functionality used across encoder and decoder samples,
including file path utilities, codec loading, and argument parsing helpers.
"""

import os
import sys
import argparse
from typing import List, Optional


def get_input_paths(root: str) -> List[str]:
    """
    Get all file paths in a directory.

    Args:
        root (str): Path to the directory containing files.

    Returns:
        List[str]: List of file paths.
    """
    try:
        _, _, files = next(os.walk(root))
        return [os.path.join(root, fn) for fn in files if os.path.isfile(os.path.join(root, fn))]
    except StopIteration:
        print(f"Provided directory is empty or invalid: {root}", file=sys.stderr)
        return []


def is_vc6_file(path: str) -> bool:
    """
    Check if the file is a VC-6 image based on its extension.

    Args:
        path (str): Path to the file.

    Returns:
        bool: True if the file has a .vc6 extension.
    """
    return os.path.splitext(path)[1].lower() == ".vc6"


def get_output_path(input_path: str, dst_dir: str, new_extension: str) -> str:
    """
    Generate output file path by replacing the input file's extension.

    Args:
        input_path (str): Original input file path.
        dst_dir (str): Destination directory.
        new_extension (str): New file extension (e.g., '.vc6', '.rgb').

    Returns:
        str: Full output file path.
    """
    basename = os.path.basename(input_path)
    new_name = os.path.splitext(basename)[0] + new_extension
    return os.path.join(dst_dir, new_name)


def load_codec(backend: str):
    """
    Load the appropriate VC-6 codec module based on the selected backend.

    Args:
        backend (str): Backend type - 'cuda', 'opencl', or 'cpu'.
                      For 'cpu', will try opencl first, then cuda.

    Returns:
        Tuple of (codec module, version string, module name).
    """
    if backend == "cuda":
        try:
            from vnova.vc6_cu12 import codec as vc6codec
            from vnova.vc6_cu12 import __version__ as vc6version
            return vc6codec, vc6version, "vnova.vc6_cu12"
        except ModuleNotFoundError:
            sys.exit(
                "Missing dependency: 'vnova.vc6_cu12'.\n"
                "CUDA backend requires the VC-6 CUDA Python SDK.\n"
                "Please refer README.md for install instructions."
            )
    elif backend == "opencl":
        try:
            from vnova.vc6_opencl import codec as vc6codec
            from vnova.vc6_opencl import __version__ as vc6version
            return vc6codec, vc6version, "vnova.vc6_opencl"
        except ModuleNotFoundError:
            sys.exit(
                "Missing dependency: 'vnova.vc6_opencl'.\n"
                "OpenCL backend requires the VC-6 OpenCL Python SDK.\n"
                "Please refer README.md for install instructions."
            )
    elif backend == "cpu":
        # For CPU backend, try opencl first (more common), then cuda
        try:
            from vnova.vc6_opencl import codec as vc6codec
            from vnova.vc6_opencl import __version__ as vc6version
            return vc6codec, vc6version, "vnova.vc6_opencl"
        except ModuleNotFoundError:
            try:
                from vnova.vc6_cu12 import codec as vc6codec
                from vnova.vc6_cu12 import __version__ as vc6version
                return vc6codec, vc6version, "vnova.vc6_cu12"
            except ModuleNotFoundError:
                sys.exit(
                    "Missing dependency: need 'vnova.vc6_opencl' or 'vnova.vc6_cu12'.\n"
                    "This sample requires VC-6 Python SDK installed.\n"
                    "Please refer README.md for install instructions."
                )
    else:
        sys.exit(f"Unknown backend: {backend}. Use 'cuda', 'opencl', or 'cpu'.")


def create_base_argument_parser(description: str) -> argparse.ArgumentParser:
    """
    Create a base argument parser with common arguments.

    Args:
        description (str): Description for the argument parser.

    Returns:
        argparse.ArgumentParser: Parser with common arguments added.
    """
    parser = argparse.ArgumentParser(description=description, add_help=False)
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
        help="Directory to write output files"
    )
    return parser


def add_batch_size_argument(parser: argparse.ArgumentParser, default: int = 10) -> None:
    """Add batch size argument to parser."""
    parser.add_argument(
        "-b", "--batch-size",
        type=int, default=default,
        help=f"Batch size (default: {default})"
    )


def add_loq_argument(parser: argparse.ArgumentParser) -> None:
    """Add Level of Quality argument to parser."""
    parser.add_argument(
        "-l", "--loq",
        type=int, default=0,
        help="Level of Quality LOQ / echelon (default: 0)"
    )


def add_backend_argument(parser: argparse.ArgumentParser, default: str = "cuda",
                         choices: Optional[List[str]] = None) -> None:
    """Add backend selection argument to parser."""
    if choices is None:
        choices = ["cuda", "opencl"]
    parser.add_argument(
        "--backend",
        type=str, choices=choices, default=default,
        help=f"Backend to use (default: {default})"
    )


def add_mode_argument(parser: argparse.ArgumentParser) -> None:
    """Add encoding mode argument to parser."""
    parser.add_argument(
        "-m", "--mode",
        type=str, choices=["lossy", "lossless"], default="lossy",
        help="Encoding mode: lossy or lossless (default: lossy)"
    )


def get_source_files(source: str, filter_vc6: bool = False) -> List[str]:
    """
    Get list of source files from a path (file or directory).

    Args:
        source (str): Path to source file or directory.
        filter_vc6 (bool): If True, only return .vc6 files.

    Returns:
        List[str]: List of file paths.
    """
    if os.path.isfile(source):
        if filter_vc6 and not is_vc6_file(source):
            print(f"Source file is not a .vc6 image: {source}", file=sys.stderr)
            sys.exit(1)
        return [source]
    else:
        files = get_input_paths(source)
        if filter_vc6:
            files = [p for p in files if is_vc6_file(p)]
        return files


def setup_output_directory(dst_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(dst_dir, exist_ok=True)
