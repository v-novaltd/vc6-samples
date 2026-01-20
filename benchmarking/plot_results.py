# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
import argparse
import datetime
import html
import json
import os
import platform
import re
import statistics
import subprocess
import sys
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


def _mean(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def _run_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _read_cpu_info():
    system = platform.system()
    if system == "Windows":
        return _read_cpu_info_windows()
    if system == "Linux":
        return _read_cpu_info_linux()
    raise NotImplementedError(f"Unsupported platform for CPU info: {system}")


def _read_cpu_info_linux():
    model_name = None
    cores = None
    try:
        with open("/proc/cpuinfo", "r") as handle:
            for line in handle:
                if line.startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                elif line.startswith("cpu cores"):
                    try:
                        cores = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                if model_name and cores:
                    break
    except OSError:
        pass
    return model_name, cores


def _read_cpu_info_windows():
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | "
            "Select-Object -First 1 -Property Name,NumberOfCores | "
            "ConvertTo-Json -Compress)",
        ]
    )
    if output:
        try:
            payload = json.loads(output)
            return payload.get("Name"), payload.get("NumberOfCores")
        except json.JSONDecodeError:
            pass

    output = _run_command(
        ["wmic", "cpu", "get", "Name,NumberOfCores", "/format:list"]
    )
    if not output:
        return None, None
    name = None
    cores = None
    for line in output.splitlines():
        if line.startswith("Name="):
            name = line.split("=", 1)[1].strip()
        elif line.startswith("NumberOfCores="):
            try:
                cores = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return name, cores


def _read_mem_total_gib():
    system = platform.system()
    if system == "Windows":
        return _read_mem_total_gib_windows()
    if system == "Linux":
        return _read_mem_total_gib_linux()
    raise NotImplementedError(f"Unsupported platform for memory info: {system}")


def _read_mem_total_gib_windows():
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem | "
            "Select-Object -ExpandProperty TotalPhysicalMemory)",
        ]
    )
    if output:
        try:
            bytes_total = float(output.strip())
            return bytes_total / (1024 ** 3)
        except ValueError:
            pass
    output = _run_command(
        ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"]
    )
    if output:
        for line in output.splitlines():
            if line.startswith("TotalPhysicalMemory="):
                try:
                    bytes_total = float(line.split("=", 1)[1].strip())
                    return bytes_total / (1024 ** 3)
                except ValueError:
                    break
    return None


def _read_mem_total_gib_linux():
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kib = float(parts[1])
                        return kib / (1024 * 1024)
    except OSError:
        pass
    return None


def _query_nvidia_smi():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name, driver_version, mem_total = parts[:3]
        gpus.append(
            {
                "name": name,
                "driver_version": driver_version,
                "memory_total_mib": mem_total,
            }
        )
    return gpus


def _collect_machine_info():
    cpu_model, cpu_cores = _read_cpu_info()
    mem_total = _read_mem_total_gib()
    gpus = _query_nvidia_smi()
    try:
        import torch as torch_lib
    except Exception:
        torch_lib = None

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "cpu_logical": os.cpu_count(),
        "memory_total_gib": mem_total,
        "gpus": gpus,
        "torch_version": torch_lib.__version__ if torch_lib else None,
        "torch_cuda": torch_lib.version.cuda if torch_lib else None,
    }
    return info


def _collect_config_summary():
    try:
        import global_vars
    except Exception:
        return {}
    return {
        "batch_sizes": getattr(global_vars, "batch_sizes", None),
        "resize_dims": getattr(global_vars, "resize_dims", None),
        "resize_params": getattr(global_vars, "resize_params", None),
        "max_vc6_batch": getattr(global_vars, "MAX_VC6_BATCH", None),
        "lossless": getattr(global_vars, "LOSSLESS", None),
        "debug_dump_images": getattr(global_vars, "DEBUG_DUMP_IMAGES", None),
        "capture_hw_stats": getattr(global_vars, "CAPTURE_HW_STATS", None),
        "num_batches": getattr(global_vars, "NUM_BATCHES", None),
        "warmup_runs": getattr(global_vars, "WARMUP_RUNS", None),
        "total_images": getattr(global_vars, "TOTAL_IMAGES", None),
        "raw_files": getattr(global_vars, "RAW_FILES", None),
        "dataset_dir": getattr(global_vars, "DATASET_DIR", None),
    }

def _append_if_not_none(values, value):
    if value is not None:
        values.append(value)

def _append_by_index(by_index, idx, value):
    if idx is not None and value is not None:
        by_index[idx].append(value)

def _aggregate_hw_samples(samples):
    if not samples:
        return {
            "cpu_pct_avg": None,
            "gpu_util_avg_pct": None,
            "gpu_mem_avg_mib": None,
            "gpu_by_index": {},
        }
    cpu_values = []
    gpu_util_values = []
    gpu_mem_values = []
    gpu_util_by_idx = defaultdict(list)
    gpu_mem_by_idx = defaultdict(list)
    for sample in samples:
        _append_if_not_none(cpu_values, sample.get("cpu_pct"))
        for gpu in sample.get("gpus") or []:
            idx = gpu.get("gpu_index")
            util = gpu.get("util_pct")
            mem = gpu.get("mem_used_mib")
            _append_if_not_none(gpu_util_values, util)
            _append_by_index(gpu_util_by_idx, idx, util)
            _append_if_not_none(gpu_mem_values, mem)
            _append_by_index(gpu_mem_by_idx, idx, mem)
    gpu_by_index = {}
    for idx in sorted(set(gpu_util_by_idx) | set(gpu_mem_by_idx)):
        gpu_by_index[idx] = {
            "util_pct": _mean(gpu_util_by_idx.get(idx, [])),
            "mem_used_mib": _mean(gpu_mem_by_idx.get(idx, [])),
        }
    return {
        "cpu_pct_avg": _mean(cpu_values),
        "gpu_util_avg_pct": _mean(gpu_util_values),
        "gpu_mem_avg_mib": _mean(gpu_mem_values),
        "gpu_by_index": gpu_by_index,
    }


def _build_run_rows(entry):
    times = entry.get("time_per_batch") or []
    hw_stats = entry.get("hw_stats") or []
    batch_size = entry.get("batch_size")
    rows = []
    for idx, batch_time in enumerate(times):
        samples = hw_stats[idx] if idx < len(hw_stats) else []
        stats = _aggregate_hw_samples(samples)
        rows.append(
            {
                "run_index": idx,
                "batch_time_s": batch_time,
                "time_per_image_ms": (batch_time / batch_size * 1000) if batch_size else None,
                "cpu_pct_avg": stats["cpu_pct_avg"],
                "gpu_util_avg_pct": stats["gpu_util_avg_pct"],
                "gpu_mem_avg_mib": stats["gpu_mem_avg_mib"],
                "gpu_by_index": stats["gpu_by_index"],
                "is_warmup": idx < WARMUP_RUNS,
            }
        )
    return rows


def _slug(text):
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def _fmt(value, precision=3, suffix=""):
    if value is None:
        return "-"
    try:
        return f"{value:.{precision}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _render_kv_list(items):
    rows = []
    for label, value in items:
        rows.append(
            f"<div class=\"kv-row\"><div class=\"kv-key\">{html.escape(label)}</div>"
            f"<div class=\"kv-value\">{html.escape(str(value))}</div></div>"
        )
    return "\n".join(rows)


def _element_label(entry):
    codec = entry.get("codec", "Unknown")
    loq = entry.get("loq")
    if codec == "VC-6" and loq is not None:
        return f"{codec} (LOQ={loq})"
    return codec


def generate_html_report(result_json: str, output_dir: str | None = None) -> Path:
    result_path = Path(result_json)
    if output_dir is None:
        output_dir = result_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with result_path.open("r") as handle:
        results = json.load(handle)

    machine_info = _collect_machine_info()
    config_info = _collect_config_summary()

    element_entries = defaultdict(list)
    cases = []
    total_entries = 0
    total_runs = 0
    batch_sizes = set()
    for case, parametrized_results in results.items():
        cases.append(case)
        for key, entry in parametrized_results.items():
            element = _element_label(entry)
            element_entries[element].append((case, key, entry))
            total_entries += 1
            batch_sizes.add(entry.get("batch_size"))
            total_runs += len(entry.get("time_per_batch") or [])

    per_element_rows = defaultdict(list)
    all_run_rows = []
    for element, items in element_entries.items():
        for case, key, entry in items:
            run_rows = _build_run_rows(entry)
            for row in run_rows:
                row_record = {
                    "element": element,
                    "case": case,
                    "key": key,
                    "batch_size": entry.get("batch_size"),
                    **row,
                }
                all_run_rows.append(row_record)
                if not row["is_warmup"]:
                    per_element_rows[element].append(row_record)

    codec_summary_rows = []
    for element, rows in sorted(per_element_rows.items()):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row.get("batch_size")].append(row)
        for batch_size, group_rows in sorted(grouped.items(), key=lambda item: (item[0] is None, item[0])):
            time_values = [row["time_per_image_ms"] for row in group_rows if row["time_per_image_ms"] is not None]
            cpu_values = [row["cpu_pct_avg"] for row in group_rows if row["cpu_pct_avg"] is not None]
            gpu_util_values = [row["gpu_util_avg_pct"] for row in group_rows if row["gpu_util_avg_pct"] is not None]
            gpu_mem_values = [row["gpu_mem_avg_mib"] for row in group_rows if row["gpu_mem_avg_mib"] is not None]
            codec_summary_rows.append(
                {
                    "codec": element,
                    "batch_size": batch_size,
                    "runs": len(group_rows),
                    "time_per_image_ms_avg": _mean(time_values),
                    "cpu_pct_avg": _mean(cpu_values),
                    "gpu_util_avg_pct": _mean(gpu_util_values),
                    "gpu_mem_avg_mib": _mean(gpu_mem_values),
                }
            )

    report_title = "VC-6 Benchmark Report"
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = output_dir / "benchmark_report.html"

    tab_buttons = [
        '<button class="tab-btn active" data-tab="summary">Summary</button>'
    ]
    tab_sections = []

    for element in sorted(element_entries):
        tab_id = f"codec-{_slug(element)}"
        tab_buttons.append(f'<button class="tab-btn" data-tab="{tab_id}">{html.escape(element)}</button>')

        detail_blocks = []
        for case, key, entry in element_entries[element]:
            run_rows = _build_run_rows(entry)
            run_table_rows = []
            for row in run_rows:
                classes = "warmup" if row["is_warmup"] else ""
                run_table_rows.append(
                    "<tr class=\"{cls}\">"
                    "<td>{run}</td>"
                    "<td>{batch_time}</td>"
                    "<td>{per_image}</td>"
                    "<td>{cpu}</td>"
                    "<td>{gpu}</td>"
                    "<td>{mem}</td>"
                    "</tr>".format(
                        cls=classes,
                        run=row["run_index"],
                        batch_time=_fmt(row["batch_time_s"], 4, " s"),
                        per_image=_fmt(row["time_per_image_ms"], 4, " ms"),
                        cpu=_fmt(row["cpu_pct_avg"], 2, " %"),
                        gpu=_fmt(row["gpu_util_avg_pct"], 2, " %"),
                        mem=_fmt(row["gpu_mem_avg_mib"], 1, " MiB"),
                    )
                )

            raw_json = json.dumps(entry, indent=2)
            detail_blocks.append(
                f"""
                <details class="entry-card">
                  <summary>
                    <div class="entry-title">{html.escape(case)} <span>{html.escape(key)}</span></div>
                    <div class="entry-meta">batch {entry.get("batch_size", "-")} - runs {len(entry.get("time_per_batch") or [])}</div>
                  </summary>
                  <div class="entry-content">
                    <div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Run</th>
                            <th>Batch time</th>
                            <th>Time / image</th>
                            <th>CPU avg</th>
                            <th>GPU avg</th>
                            <th>GPU mem</th>
                          </tr>
                        </thead>
                        <tbody>
                          {''.join(run_table_rows)}
                        </tbody>
                      </table>
                    </div>
                    <pre>{html.escape(raw_json)}</pre>
                  </div>
                </details>
                """
            )

        tab_sections.append(
            f"""
            <section id="{tab_id}" class="tab-panel">
              <div class="section-title">
                <h2>{html.escape(element)} detail</h2>
                <p>Full per-run data, including warmups and raw JSON payloads.</p>
              </div>
              <div class="stack">
                {''.join(detail_blocks) if detail_blocks else '<div class="card">No data.</div>'}
              </div>
            </section>
            """
        )

    machine_rows = [
        ("Host", machine_info.get("hostname")),
        ("Platform", machine_info.get("platform")),
        ("Python", machine_info.get("python")),
        ("CPU model", machine_info.get("cpu_model")),
        ("CPU cores (physical)", machine_info.get("cpu_cores")),
        ("CPU cores (logical)", machine_info.get("cpu_logical")),
        ("Memory (GiB)", _fmt(machine_info.get("memory_total_gib"), 2)),
        ("Torch", machine_info.get("torch_version")),
        ("CUDA", machine_info.get("torch_cuda")),
    ]
    gpu_details = []
    for idx, gpu in enumerate(machine_info.get("gpus") or []):
        gpu_details.append(
            f"GPU {idx}: {gpu.get('name')} - {gpu.get('memory_total_mib')} MiB - driver {gpu.get('driver_version')}"
        )
    if gpu_details:
        machine_rows.append(("GPU", " | ".join(gpu_details)))

    config_rows = [
        ("Batch sizes", config_info.get("batch_sizes")),
        ("Resize dims", config_info.get("resize_dims")),
        ("Resize params", config_info.get("resize_params")),
        ("MAX_VC6_BATCH", config_info.get("max_vc6_batch")),
        ("LOSSLESS", config_info.get("lossless")),
        ("DEBUG_DUMP_IMAGES", config_info.get("debug_dump_images")),
        ("CAPTURE_HW_STATS", config_info.get("capture_hw_stats")),
        ("NUM_BATCHES", config_info.get("num_batches")),
        ("WARMUP_RUNS", config_info.get("warmup_runs")),
        ("TOTAL_IMAGES", config_info.get("total_images")),
        ("RAW_FILES", config_info.get("raw_files")),
        ("DATASET_DIR", config_info.get("dataset_dir")),
    ]

    overview_rows = [
        ("Cases", ", ".join(cases) if cases else "-"),
        ("Total entries", total_entries),
        ("Total runs", total_runs),
        ("Batch sizes (results)", sorted([b for b in batch_sizes if b is not None])),
        ("Warmup runs", WARMUP_RUNS),
        ("Generated at", generated_at),
    ]

    summary_table_rows = []
    for row in codec_summary_rows:
        summary_table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['codec'])}</td>"
            f"<td>{html.escape(str(row['batch_size']))}</td>"
            f"<td>{row['runs']}</td>"
            f"<td>{_fmt(row['time_per_image_ms_avg'], 4, ' ms')}</td>"
            f"<td>{_fmt(row['cpu_pct_avg'], 2, ' %')}</td>"
            f"<td>{_fmt(row['gpu_util_avg_pct'], 2, ' %')}</td>"
            f"<td>{_fmt(row['gpu_mem_avg_mib'], 1, ' MiB')}</td>"
            "</tr>"
        )

    chart_batches = sorted({row["batch_size"] for row in codec_summary_rows if row.get("batch_size") is not None})
    chart_series_map = defaultdict(dict)
    for row in codec_summary_rows:
        batch_size = row.get("batch_size")
        if batch_size is None:
            continue
        chart_series_map[row["codec"]][batch_size] = row.get("time_per_image_ms_avg")

    chart_series = []
    for name in sorted(chart_series_map):
        series_values = [chart_series_map[name].get(batch) for batch in chart_batches]
        chart_series.append(
            {
                "name": name,
                "type": "line",
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "data": series_values,
            }
        )

    per_run_rows = []
    for row in all_run_rows:
        if row["is_warmup"]:
            continue
        per_run_rows.append(
            "<tr>"
            f"<td>{html.escape(row['element'])}</td>"
            f"<td>{html.escape(row['case'])}</td>"
            f"<td>{html.escape(row['key'])}</td>"
            f"<td>{row['run_index']}</td>"
            f"<td>{_fmt(row['time_per_image_ms'], 4, ' ms')}</td>"
            f"<td>{_fmt(row['cpu_pct_avg'], 2, ' %')}</td>"
            f"<td>{_fmt(row['gpu_util_avg_pct'], 2, ' %')}</td>"
            f"<td>{_fmt(row['gpu_mem_avg_mib'], 1, ' MiB')}</td>"
            "</tr>"
        )

    summary_section = f"""
      <section id="summary" class="tab-panel active">
        <div class="section-title">
          <h2>Summary</h2>
          <p>Machine + config fingerprint, averages for non-warmup runs, and per-run metrics.</p>
        </div>
        <div class="stack summary-stack">
          <details class="card collapse">
            <summary>
              <h3>Environment details</h3>
              <span class="chev"></span>
            </summary>
            <div class="grid summary-grid">
              <div class="subcard">
                <h3>Machine details</h3>
                <div class="kv-list">
                  {_render_kv_list(machine_rows)}
                </div>
              </div>
              <div class="subcard">
                <h3>Run config</h3>
                <div class="kv-list">
                  {_render_kv_list(config_rows)}
                </div>
              </div>
              <div class="subcard">
                <h3>Test overview</h3>
                <div class="kv-list">
                  {_render_kv_list(overview_rows)}
                </div>
              </div>
            </div>
          </details>
          <div class="card">
            <div class="card-title">
              <h3>Element performance overview</h3>
              <div class="pill">x axis batch size</div>
              <div class="pill">y axis time per image (ms)</div>
            </div>
            <div id="codec-chart" class="chart"></div>
          </div>
          <div class="card">
            <div class="card-title">
              <h3>Element averages (non-warmup)</h3>
              <div class="pill">cases {len(cases)}</div>
              <div class="pill">entries {total_entries}</div>
              <div class="pill">runs {total_runs}</div>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Element</th>
                    <th>Batch size</th>
                    <th>Runs</th>
                    <th>Time / image</th>
                    <th>CPU avg</th>
                    <th>GPU avg</th>
                    <th>GPU mem</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(summary_table_rows) if summary_table_rows else '<tr><td colspan="7">No data.</td></tr>'}
                </tbody>
              </table>
            </div>
          </div>
          <div class="card">
            <h3>Per-run metrics (non-warmup)</h3>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Element</th>
                    <th>Test</th>
                    <th>Key</th>
                    <th>Run</th>
                    <th>Time / image</th>
                    <th>CPU avg</th>
                    <th>GPU avg</th>
                    <th>GPU mem</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(per_run_rows) if per_run_rows else '<tr><td colspan="8">No data.</td></tr>'}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
    """

    html_doc = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(report_title)}</title>
    <style>
      :root {{
        --bg: #0b0d13;
        --bg-alt: #131827;
        --card: rgba(21, 25, 39, 0.92);
        --accent: #ffb347;
        --accent-2: #4ad7d1;
        --ink: #f5f3e7;
        --muted: #a7aec6;
        --stroke: rgba(255, 255, 255, 0.08);
        --glow: rgba(255, 179, 71, 0.32);
        --shadow: 0 20px 45px rgba(8, 10, 18, 0.45);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
        background: radial-gradient(circle at top left, #182036 0%, #0b0d13 55%);
        min-height: 100vh;
      }}
      .backdrop {{
        position: fixed;
        inset: 0;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
      }}
      .orb {{
        position: absolute;
        border-radius: 999px;
        filter: blur(0px);
        opacity: 0.5;
      }}
      .orb.one {{
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(255, 179, 71, 0.55), transparent 70%);
        top: -120px;
        right: -120px;
        animation: float 10s ease-in-out infinite;
      }}
      .orb.two {{
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, rgba(74, 215, 209, 0.45), transparent 70%);
        bottom: -200px;
        left: -120px;
        animation: float 14s ease-in-out infinite;
      }}
      @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(18px); }}
      }}
      .page {{
        position: relative;
        z-index: 1;
        padding: 32px 32px 64px;
        max-width: 1280px;
        margin: 0 auto;
      }}
      header.hero {{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: flex-end;
        gap: 16px;
        padding: 24px 28px;
        background: linear-gradient(135deg, rgba(34, 40, 59, 0.95), rgba(17, 20, 33, 0.96));
        border-radius: 22px;
        border: 1px solid var(--stroke);
        box-shadow: var(--shadow);
      }}
      header.hero h1 {{
        font-family: "Fraunces", "Georgia", serif;
        font-size: clamp(2rem, 3vw, 2.8rem);
        margin: 0;
        letter-spacing: 0.02em;
      }}
      header.hero p {{
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 0.98rem;
      }}
      .hero-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }}
      .pill {{
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid var(--stroke);
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        color: var(--muted);
      }}
      nav.tabs {{
        margin: 28px 0 18px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }}
      .tab-btn {{
        background: transparent;
        color: var(--muted);
        border: 1px solid var(--stroke);
        padding: 10px 16px;
        border-radius: 999px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
      }}
      .tab-btn.active {{
        color: #0b0d13;
        background: var(--accent);
        border-color: transparent;
        box-shadow: 0 10px 30px var(--glow);
      }}
      .tab-panel {{
        display: none;
        animation: rise 0.4s ease;
      }}
      .tab-panel.active {{
        display: block;
      }}
      @keyframes rise {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}
      .section-title h2 {{
        margin: 0 0 6px;
        font-size: 1.6rem;
      }}
      .section-title p {{
        margin: 0 0 18px;
        color: var(--muted);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 18px;
      }}
      .card {{
        background: var(--card);
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid var(--stroke);
        box-shadow: var(--shadow);
      }}
      .card h3 {{
        margin: 0 0 12px;
        font-size: 1.2rem;
      }}
      .card-title {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
      }}
      .kv-list {{
        display: grid;
        gap: 8px;
      }}
      .kv-row {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        border-bottom: 1px dashed var(--stroke);
        padding-bottom: 6px;
      }}
      .kv-key {{
        color: var(--muted);
        font-size: 0.9rem;
      }}
      .kv-value {{
        font-weight: 600;
        font-size: 0.9rem;
        text-align: right;
        max-width: 65%;
        word-break: break-word;
      }}
      .table-wrap {{
        overflow-x: auto;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        min-width: 640px;
      }}
      th, td {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--stroke);
      }}
      th {{
        text-align: left;
        color: var(--muted);
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      tr:hover td {{
        background: rgba(255, 255, 255, 0.03);
      }}
      tr.warmup td {{
        color: rgba(255, 255, 255, 0.45);
      }}
      .stack {{
        display: grid;
        gap: 14px;
      }}
      .summary-stack {{
        gap: 18px;
      }}
      .summary-grid {{
        margin-top: 12px;
      }}
      .subcard {{
        background: rgba(15, 18, 30, 0.85);
        border-radius: 14px;
        border: 1px solid var(--stroke);
        padding: 14px;
      }}
      .subcard h3 {{
        margin: 0 0 10px;
        font-size: 1.05rem;
      }}
      .entry-card {{
        background: rgba(18, 22, 35, 0.9);
        border-radius: 16px;
        border: 1px solid var(--stroke);
        padding: 12px 16px;
      }}
      .collapse summary {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
      }}
      .collapse summary h3 {{
        margin: 0;
      }}
      .collapse .chev {{
        width: 12px;
        height: 12px;
        border-right: 2px solid var(--muted);
        border-bottom: 2px solid var(--muted);
        transform: rotate(45deg);
        transition: transform 0.2s ease;
      }}
      .collapse[open] .chev {{
        transform: rotate(-135deg);
      }}
      .collapse > .kv-list {{
        margin-top: 12px;
      }}
      .entry-card summary {{
        cursor: pointer;
        list-style: none;
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
      }}
      .entry-card summary::-webkit-details-marker {{
        display: none;
      }}
      .entry-title {{
        font-weight: 600;
      }}
      .entry-title span {{
        display: block;
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 4px;
      }}
      .entry-meta {{
        font-size: 0.82rem;
        color: var(--muted);
      }}
      .entry-content {{
        margin-top: 12px;
        display: grid;
        gap: 12px;
      }}
      .chart {{
        width: 100%;
        height: 360px;
      }}
      pre {{
        background: rgba(10, 12, 20, 0.8);
        border: 1px solid var(--stroke);
        border-radius: 12px;
        padding: 12px;
        font-size: 0.78rem;
        color: #e6e4d7;
        overflow-x: auto;
      }}
      @media (max-width: 720px) {{
        .page {{
          padding: 20px 18px 48px;
        }}
        header.hero {{
          padding: 18px;
        }}
        nav.tabs {{
          gap: 8px;
        }}
        .tab-btn {{
          padding: 8px 12px;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="backdrop">
      <div class="orb one"></div>
      <div class="orb two"></div>
    </div>
    <div class="page">
      <header class="hero">
        <div>
          <h1>{html.escape(report_title)}</h1>
          <p>Generated {html.escape(generated_at)} - results: {html.escape(str(result_path))}</p>
        </div>
        <div class="hero-metrics">
          <div class="pill">batch sizes {html.escape(str(sorted([b for b in batch_sizes if b is not None])))}</div>
          <div class="pill">warmup {WARMUP_RUNS}</div>
          <div class="pill">cases {len(cases)}</div>
        </div>
      </header>
      <nav class="tabs" role="tablist">
        {"".join(tab_buttons)}
      </nav>
      {summary_section}
      {"".join(tab_sections)}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
      const buttons = document.querySelectorAll('.tab-btn');
      const panels = document.querySelectorAll('.tab-panel');
      buttons.forEach(btn => {{
        btn.addEventListener('click', () => {{
          const target = btn.getAttribute('data-tab');
          buttons.forEach(b => b.classList.toggle('active', b === btn));
          panels.forEach(panel => panel.classList.toggle('active', panel.id === target));
          history.replaceState(null, '', '#' + target);
        }});
      }});
      const hash = window.location.hash.replace('#', '');
      if (hash) {{
        const btn = document.querySelector(`.tab-btn[data-tab="${hash}"]`);
        if (btn) btn.click();
      }}

      const chartEl = document.getElementById('codec-chart');
      if (chartEl && window.echarts) {{
        const chart = echarts.init(chartEl);
        const chartData = {{
          batches: {json.dumps(chart_batches)},
          series: {json.dumps(chart_series)},
        }};
        chart.setOption({{
          backgroundColor: 'transparent',
          tooltip: {{ trigger: 'axis' }},
          legend: {{
            textStyle: {{ color: '#a7aec6' }},
            top: 10,
          }},
          grid: {{ left: 40, right: 20, top: 50, bottom: 40 }},
          xAxis: {{
            type: 'category',
            data: chartData.batches,
            name: 'Batch size',
            axisLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.2)' }} }},
            axisLabel: {{ color: '#a7aec6' }},
          }},
          yAxis: {{
            type: 'value',
            name: 'Time per image (ms)',
            axisLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.2)' }} }},
            axisLabel: {{ color: '#a7aec6' }},
            splitLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.08)' }} }},
          }},
          series: chartData.series,
        }});
        window.addEventListener('resize', () => chart.resize());
      }}
    </script>
  </body>
</html>
"""

    output_path.write_text(html_doc)
    return output_path


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

    try:
        generate_html_report(args.result_json or str(summary_path.parent / "test_results.json"), output_dir=args.output_dir)
    except Exception as exc:
        print(f"Failed to generate HTML report: {exc}", file=sys.stderr)

    if args.plot:
        plot_summary(summary_path, output_dir=args.output_dir or Path(summary_path).parent)
