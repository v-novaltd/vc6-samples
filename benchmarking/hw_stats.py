# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
import logging
import subprocess
import threading
import time

import global_vars

LOGGER = logging.getLogger("HW_STATS")


class HWStatsSampler:
    def __init__(self, interval_sec=1.0):
        self.interval_sec = interval_sec
        self.samples = []
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None
        self._prev_cpu_times = None

    def start(self):
        if not global_vars.CAPTURE_HW_STATS:
            return
        self._start_time = time.time()
        self._prev_cpu_times = _read_cpu_times()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not global_vars.CAPTURE_HW_STATS:
            return []
        if self._thread is None:
            return []
        self._stop_event.set()
        self._thread.join()
        if not self.samples and self._prev_cpu_times:
            end_cpu_times = _read_cpu_times()
            cpu_pct = _cpu_usage_pct(self._prev_cpu_times, end_cpu_times)
            self.samples.append(
                {
                    "t": time.time() - self._start_time,
                    "cpu_pct": cpu_pct,
                    "gpus": _read_gpu_stats(),
                }
            )
        return self.samples

    def _run(self):
        last_cpu = self._prev_cpu_times
        while not self._stop_event.wait(self.interval_sec):
            now = time.time()
            curr_cpu = _read_cpu_times()
            cpu_pct = _cpu_usage_pct(last_cpu, curr_cpu)
            last_cpu = curr_cpu
            self.samples.append(
                {
                    "t": now - self._start_time,
                    "cpu_pct": cpu_pct,
                    "gpus": _read_gpu_stats(),
                }
            )


def _read_cpu_times():
    try:
        with open("/proc/stat", "r") as handle:
            for line in handle:
                if line.startswith("cpu "):
                    fields = [int(value) for value in line.split()[1:]]
                    total = sum(fields)
                    idle = fields[3] + fields[4] if len(fields) > 4 else fields[3]
                    return total, idle
    except OSError as exc:
        LOGGER.warning("Failed to read /proc/stat: %s", exc)
    return None


def _cpu_usage_pct(prev_times, curr_times):
    if not prev_times or not curr_times:
        return None
    prev_total, prev_idle = prev_times
    curr_total, curr_idle = curr_times
    total_delta = curr_total - prev_total
    idle_delta = curr_idle - prev_idle
    if total_delta <= 0:
        return 0.0
    return 100.0 * (total_delta - idle_delta) / total_delta


def _read_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        LOGGER.warning("nvidia-smi not found; skipping GPU stats.")
        return None

    if result.returncode != 0:
        LOGGER.warning("nvidia-smi failed: %s", result.stderr.strip())
        return None

    gpus = []
    for index, line in enumerate(result.stdout.strip().splitlines()):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            util_pct = float(parts[0])
            mem_used_mib = float(parts[1])
        except ValueError:
            continue
        gpus.append(
            {
                "gpu_index": index,
                "util_pct": util_pct,
                "mem_used_mib": mem_used_mib,
            }
        )
    return gpus
