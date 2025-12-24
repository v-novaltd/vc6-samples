# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
from pathlib import Path
from itertools import cycle, islice
import time

import pytest
import torch

nvimgcodec = pytest.importorskip(
    "nvidia.nvimgcodec", reason="nvimgcodec package is required for JPEG benchmarking"
)

from nvidia.dali import pipeline, fn

from global_vars import (
    RAW_FILES,
    LOSSLESS,
    TOTAL_IMAGES,
    NUM_BATCHES,
    batch_sizes,
    resize_params,
    test_results,
)

DALI_THREADS = 8
JPEG_PARAMS = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=80, color_spec=nvimgcodec.ColorSpec.YCC, jpeg_encode_params=nvimgcodec.JpegEncodeParams())
J2K_PARAMS = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=80, color_spec=nvimgcodec.ColorSpec.YCC, jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams())
# HT quality mode seems to be buggy and only supports QStep
J2K_HT_PARAMS = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=80, color_spec=nvimgcodec.ColorSpec.YCC, jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(ht=True))

class TestNVJPEGCodecPerformance:
    codec = "JPEG"
    file_ending = ".jpg"
    enc_params = JPEG_PARAMS

    @pytest.fixture(scope="class", autouse=True)
    def images(self):
        """Load codec assets into memory, failing if encode tests have not populated them."""
        if LOSSLESS and self.codec == "JPEG":
            pytest.skip("Lossless mode is not supported for JPEG codec")
        asset_dir = Path(RAW_FILES, self.codec)
        if not asset_dir.is_dir():
            pytest.fail(
                f"{self.codec} assets missing in {asset_dir}. Fetch the files from Huggingface dataset."
            )
        batch = [path for path in sorted(asset_dir.iterdir()) if path.suffix.lower() == self.file_ending]
        if len(batch) < TOTAL_IMAGES:
            pytest.fail(
                f"Expected at least {TOTAL_IMAGES} {self.codec} files under {asset_dir}, found {len(batch)}"
            )
        all_images = []
        for file_name in batch[:TOTAL_IMAGES]:
            all_images.append(file_name.read_bytes())
        return all_images
    
    @pytest.mark.parametrize("batch_size", batch_sizes)
    def test_decode_performance(self, batch_size, images):
        """Measure nvimgcodec decode throughput for pre-generated assets."""
        if LOSSLESS and self.codec == "JPEG":
            pytest.skip("Lossless mode is not supported for JPEG codec")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for nvimgcodec decode benchmarks.")
        nvdec = nvimgcodec.Decoder()
        timings = []
        total_needed = NUM_BATCHES * batch_size
        cycled_images = list(islice(cycle(images), total_needed))
        stream = torch.cuda.Stream()
        stream_handle = stream.cuda_stream
        start = time.time()
        for i in range(NUM_BATCHES):
            with torch.cuda.nvtx.range(f"{self.codec} batch size {batch_size}"):
                batch_start = time.time()
                j = i * batch_size
                k = j + batch_size
                decoded = nvdec.decode(cycled_images[j:k], cuda_stream=stream_handle)
                stream.synchronize()
                timings.append(time.time() - batch_start)
        end = time.time()
        assert isinstance(timings, list)
        # Store results in global dictionary
        test_results["test_decode_performance"].update({
            f"batch={batch_size}_codec={self.codec}": {
                "codec": self.codec,
                "batch_size": batch_size,
                "num_batches": NUM_BATCHES,
                "total_time": end - start,
                "time_per_batch": timings,
                "fn": "decode_performance"
            }
        })

    @pytest.mark.parametrize("batch_size,resize_dim", resize_params)
    def test_decode_resize_performance(self, batch_size, resize_dim, images):
        """Measure nvimgcodec decode followed by DALI resize throughput."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for nvimgcodec decode+resize benchmarks.")
        nvdec = nvimgcodec.Decoder()
        timings = []
        total_needed = NUM_BATCHES * batch_size
        cycled_images = list(islice(cycle(images), total_needed))
        stream = torch.cuda.Stream()
        stream_handle = stream.cuda_stream
        class ResizePipeline(pipeline.Pipeline):
            def __init__(self, batch_size, num_threads, device_id, width, height, cuda_stream):
                super().__init__(batch_size, num_threads, device_id, seed=12, prefetch_queue_depth=1)
                self.input = fn.external_source(name="decoded_images", batch=True, device="gpu", cuda_stream=cuda_stream)
                self.resize = fn.resize(self.input, size=(width, height))
            def define_graph(self):
                resized = self.resize
                return resized
        resize_pipeline = ResizePipeline(batch_size=batch_size, num_threads=DALI_THREADS, 
                                        device_id=0, width=resize_dim[0], height=resize_dim[1],
                                        cuda_stream=stream_handle)
        resize_pipeline.build()
        start = time.time()
        for i in range(NUM_BATCHES):
            batch_start = time.time()
            j = i * batch_size
            k = j + batch_size
            decoded = nvdec.decode(cycled_images[j:k], cuda_stream=stream_handle)
            resize_pipeline.feed_input("decoded_images", decoded, cuda_stream=stream)
            resized_images = resize_pipeline.run()
            stream.synchronize()
            timings.append(time.time() - batch_start)
        end = time.time()
        assert isinstance(timings, list)
        # Store results in global dictionary
        test_results["test_decode_resize_performance"].update({ 
            f"batch={batch_size}_size={resize_dim}_codec={self.codec}" : {
                "codec": self.codec,
                "batch_size": batch_size,
                "num_batches": NUM_BATCHES,
                "resize_dim": resize_dim,
                "total_time": end - start,
                "time_per_batch": timings,
            }
        })


class TestNVJ2KCodecPerformance(TestNVJPEGCodecPerformance):
    codec = "J2K"
    file_ending = ".jp2"
    enc_params = J2K_PARAMS


class TestNVJ2KHTCodecPerformance(TestNVJPEGCodecPerformance):
    codec = "J2K_HT"
    file_ending = ".jp2"
    enc_params = J2K_HT_PARAMS
