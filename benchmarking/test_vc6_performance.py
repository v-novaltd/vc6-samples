# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vars import (
    RAW_FILES,
    TOTAL_IMAGES,
    NUM_BATCHES,
    batch_sizes,
    MAX_VC6_BATCH,
    DEBUG_DUMP_IMAGES,
    CAPTURE_HW_STATS,
    test_results,
)
from debug_dump import dump_decoded_images
from hw_stats import HWStatsSampler

MAXWIDTH = 4500
MAXHEIGHT = 4500
LOQ = [0, 1, 2]
ENABLE_DEBUG = False
BPP = 2  # 3 is close to lossless


def _get_vc6_cuda():
    """Import the VC-6 CUDA codec module, skipping the test if unavailable."""
    try:
        from vnova.vc6_cuda import codec as vc6
    except ImportError as exc:
        pytest.skip(f"VC-6 CUDA codec unavailable: {exc}")
    return vc6

def _get_vc6_sizes(vc6, buffers, loq):
    sizes = []
    for buffer in buffers:
        probe_data = buffer[:1024]
        if len(probe_data) < 1024:
            sizes.append((0, 0))
            continue
        probe = vc6.ProbeFrame(probe_data)
        if probe is None:
            sizes.append((0, 0))
            continue
        width = probe.width
        height = probe.height
        for _ in range(loq):
            width = (width + 1) // 2
            height = (height + 1) // 2
        sizes.append((width, height))
    return sizes

class TestVC6CodecPerformance:
    codec = "VC-6"
    file_ending = ".vc6"

    def _load_images(self):
        """Load generated VC-6 asset buffers for decode benchmarks."""
        vc6_files = Path(RAW_FILES, self.codec)
        if not vc6_files.is_dir():
            pytest.fail(
                f"{self.codec} assets missing in {vc6_files}. Fetch the files from Huggingface dataset."
            )
        files = [
            child
            for child in sorted(vc6_files.iterdir())
            if child.is_file() and child.suffix.lower() == self.file_ending
        ]
        if len(files) < TOTAL_IMAGES:
            pytest.fail(
                f"Expected at least {TOTAL_IMAGES} {self.codec} files under {vc6_files}, found {len(files)}"
            )
        images = []
        for file_name in files[:TOTAL_IMAGES]:
            with file_name.open("rb") as handle:
                images.append((handle.read(), file_name.stem))
        return images

    @pytest.fixture(scope="class", autouse=True)
    def images(self):
        """Expose VC-6 asset buffers to decode tests."""
        return self._load_images()

    @pytest.mark.parametrize("loq", LOQ)
    @pytest.mark.parametrize("batch_size", batch_sizes)
    def test_decode_performance_batch_exp(self, loq, batch_size, images):
        try:
            """Measure VC-6 batch decoder throughput for the configured LOQ and batch size."""
            if not torch.cuda.is_available():
                pytest.skip("CUDA is required for VC-6 decode benchmarks.")

            vc6 = _get_vc6_cuda()
            decoder = vc6.BatchDecoder_exp(
                MAXWIDTH,
                MAXHEIGHT,
                vc6.CodecBackendType.GPU,
                vc6.PictureFormat.RGB_8,
                vc6.ImageMemoryType.CUDA_DEVICE,
                enable_logs=ENABLE_DEBUG,
                num_backends=0,
                num_buffers=batch_size,
            )

            timings = []
            total_needed = NUM_BATCHES * batch_size
            cycled_images = list(islice(cycle(images), total_needed))
            image_buffers = [item[0] for item in cycled_images[0:total_needed]]
            image_names = [item[1] for item in cycled_images[0:total_needed]]
            start = time.time()
            hw_stats = []
            for i in range(NUM_BATCHES):
                with torch.cuda.nvtx.range(f"VC6 batch size {batch_size}"):
                    j = i * batch_size
                    k = j + batch_size
                    batch_buffers = image_buffers[j:k]
                    sampler = HWStatsSampler()
                    sampler.start()
                    batch_start = time.time()
                    decoded = decoder.decode(batch_buffers, loq)
                    timings.append(time.time() - batch_start)
                    if CAPTURE_HW_STATS:
                        hw_stats.append(sampler.stop())
                    if DEBUG_DUMP_IMAGES and i == 0:
                        sizes = _get_vc6_sizes(vc6, batch_buffers, loq)
                        batch_names = image_names[j:k]
                        dump_decoded_images(
                            self.codec,
                            decoded,
                            names=batch_names,
                            sizes=sizes,
                            prefix=f"loq{loq}_batch{batch_size}",
                        )
            end = time.time()

            test_results["test_decode_performance_batch_exp"].update(
                {
                    f"batch={batch_size}_loq={loq}_codec={self.codec}": {
                        "codec": self.codec,
                        "batch_size": batch_size,
                        "loq": loq,
                        "num_batches": NUM_BATCHES,
                        "total_time": end - start,
                        "time_per_batch": timings,
                        "hw_stats": hw_stats if CAPTURE_HW_STATS else None,
                        "fn": "test_decode_performance_batch_exp",
                    }
                }
            )
        except:
            pytest.UsageError("Unknown error")

