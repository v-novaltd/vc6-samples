# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import torch

from global_vars import MAX_VC6_BATCH, WARMUP_RUNS


def _collect_case_data(summary: Dict, case_name: str) -> Tuple[Iterable[int], Dict[str, Dict[int, float]]] | None:
    """Extract batch-wise metrics for a particular test case from the summary."""
    case_payload = summary.get(case_name)
    if not case_payload:
        return None

    batch_sizes = set()
    data = defaultdict(dict)
    is_experimental = "batch_exp" in case_name or "experimental" in case_name.lower()

    for codec, results in case_payload.items():
        for key, val in results.items():
            batch_match = re.search(r"batch=(\d+)", key)
            if not batch_match:
                continue
            batch = int(batch_match.group(1))
            device_match = re.search(r"device=([^_]+)", key)
            variant_match = re.search(r"variant=([^_]+)", key)
            loq_match = re.search(r"loq=(\d+)", key)
            if device_match:
                label = f"{codec} ({device_match.group(1)})"
            elif variant_match:
                label = f"{codec} ({variant_match.group(1)})"
            elif loq_match:
                codec_label = f"{codec} Experimental" if is_experimental else codec
                label = f"{codec_label} (LOQ={loq_match.group(1)})"
            else:
                label = f"{codec} Experimental" if is_experimental else codec
            batch_sizes.add(batch)
            data[label][batch] = val

    if not batch_sizes:
        return None

    return sorted(batch_sizes), data


def _plot_case(batch_sizes, data, title, output_filename, output_dir, batch_limit=None):
    """Render a line chart for the supplied case data and persist it to disk."""
    if batch_limit is not None:
        filtered_batches = [b for b in batch_sizes if b <= batch_limit]
    else:
        filtered_batches = batch_sizes

    if not filtered_batches:
        return

    plt.figure(figsize=(8, 6))
    for codec, metrics in data.items():
        # Convert from seconds to milliseconds and round to 6 decimal places
        y = [round(metrics.get(batch, float("nan")) * 1000, 6) for batch in filtered_batches]
        plt.plot(filtered_batches, y, marker="o", label=codec)
    plt.xlabel("Batch Size")
    plt.ylabel("Time per Image (ms)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = Path(output_dir) / output_filename
    plt.savefig(output_path)
    plt.close()


def plot_summary(summary_json, output_dir="."):
    """Generate decode, encode, and transcode plots from the summary JSON."""
    summary_path = Path(summary_json)
    output_dir = Path(output_dir)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    decode_payload = _collect_case_data(summary, "test_decode_performance")
    batch_exp_payload = _collect_case_data(summary, "test_decode_performance_batch_exp")
    
    # Combine both decode test cases into one dataset
    if decode_payload or batch_exp_payload:
        all_batches = set()
        combined_data = defaultdict(dict)
        
        if decode_payload:
            batches, data = decode_payload
            all_batches.update(batches)
            for label, metrics in data.items():
                combined_data[label].update(metrics)
        
        if batch_exp_payload:
            batches, data = batch_exp_payload
            all_batches.update(batches)
            for label, metrics in data.items():
                combined_data[label].update(metrics)
        
        if all_batches:
            sorted_batches = sorted(all_batches)
            _plot_case(
                sorted_batches,
                combined_data,
                "Codec Decode Performance Comparison",
                "codec_performance.png",
                output_dir=output_dir,
            )
            _plot_case(
                sorted_batches,
                combined_data,
                "Codec Decode Performance (≤ VC-6 Max Batch)",
                "codec_performance_vc6.png",
                output_dir=output_dir,
                batch_limit=MAX_VC6_BATCH,
            )

    encode_payload = _collect_case_data(summary, "test_encode_performance")
    if encode_payload:
        batches, data = encode_payload
        _plot_case(
            batches,
            data,
            "Codec Encode Performance Comparison",
            "codec_encode_performance.png",
            output_dir=output_dir,
        )
        _plot_case(
            batches,
            data,
            "Codec Encode Performance (≤ VC-6 Max Batch)",
            "codec_encode_performance_vc6.png",
            output_dir=output_dir,
            batch_limit=MAX_VC6_BATCH,
        )

    transcode_payload = _collect_case_data(summary, "test_transcode_performance")
    if transcode_payload:
        batches, data = transcode_payload
        _plot_case(
            batches,
            data,
            "Codec Transcode Performance Comparison",
            "codec_transcode_performance.png",
            output_dir=output_dir,
        )
        _plot_case(
            batches,
            data,
            "Codec Transcode Performance (≤ VC-6 Max Batch)",
            "codec_transcode_performance_vc6.png",
            output_dir=output_dir,
            batch_limit=MAX_VC6_BATCH,
        )



def _mean_time_per_image(entry: Dict) -> float:
    """Compute the per-image latency after discarding warmup batches."""
    batch_times = entry.get("time_per_batch", [])
    if not batch_times:
        return float("nan")
    effective = batch_times[WARMUP_RUNS:] if len(batch_times) > WARMUP_RUNS else batch_times
    tensor = torch.tensor(effective, dtype=torch.float32)
    if tensor.numel() == 0:
        tensor = torch.tensor(batch_times, dtype=torch.float32)
    avg_batch_time = tensor.mean().item()
    return avg_batch_time / entry["batch_size"]


def summarize(result_json: str | None = None, output_dir: str | None = None) -> Path:
    """Aggregate raw timings into per-image averages and emit summary JSON."""
    if result_json is None:
        result_path = Path("test_results.json")
    else:
        result_path = Path(result_json)
    if output_dir is None:
        output_dir = result_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with result_path.open("r") as f:
        results = json.load(f)
    summary = {}
    for case, parametrized_results in results.items():
        timings = defaultdict(dict)
        for key, data in parametrized_results.items():
            codec = data["codec"]
            timings[codec][key] = _mean_time_per_image(data)
        summary[case] = timings
        print(f"Case: {case}")
        print(json.dumps(timings, indent=4))
    summary_path = output_dir / "summmary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4)
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plotting tool for testing results")

    parser.add_argument("result_json", nargs="?", default=None)
    parser.add_argument("--plot", action="store_true", help="Plot codec performance from summmary.json")
    parser.add_argument("--output-dir", default=None, help="Directory to write summary and plots")
    args = parser.parse_args()

    summary_path = summarize(result_json=args.result_json, output_dir=args.output_dir)

    if args.plot:
        plot_summary(summary_path, output_dir=args.output_dir or Path(summary_path).parent)
