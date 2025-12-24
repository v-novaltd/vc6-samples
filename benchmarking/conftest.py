# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2025 V-Nova International Ltd.
import pytest
import json
import datetime
from pathlib import Path
import logging
import global_vars
from plot_results import summarize, plot_summary

logger = logging.getLogger(__name__)

# Spin-up logic: runs before any tests
@pytest.fixture(scope="session", autouse=True)
def spin_up():
    """Prepare a clean workspace, capture benchmark results, and render plots."""
    timestamp = datetime.datetime.now().strftime("test_results_%d-%m-%y:%H-%M-%S")
    results_dir = Path(timestamp)
    results_dir.mkdir(parents=True, exist_ok=True)
    global_vars.RESULTS_DIR = str(results_dir)
    Path("latest_results_dir.txt").write_text(str(results_dir))

    yield

    results_path = results_dir / "test_results.json"
    with open(results_path, "w") as f:
        f.write(json.dumps(global_vars.test_results, indent=4))
    summary_path = summarize(result_json=str(results_path), output_dir=results_dir)
    plot_summary(str(summary_path), output_dir=results_dir)
