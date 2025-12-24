#!/bin/bash

# Set to 1 to prevent VC-6 from downloading images from GPU to CPU after decoding.
SKIP_VC6_GPU2CPU_DOWNLOAD=1
# Set to 1 to enable nsys profiling, 0 to run without nsys
NSYS_ENABLED=0
# Set to INFO to see all logs, ERROR to see only errors
ENABLE_DEBUG=0

if [ "$SKIP_VC6_GPU2CPU_DOWNLOAD" -eq 1 ]; then
  export VC6_EXPERIMENT_SKIP_DOWNLOAD=1
fi

DEBUG_ARG="-v"

if [ "$ENABLE_DEBUG" -eq 1 ]; then
  DEBUG_ARG="-vv --log-cli-level=INFO"
fi

# Run download tests first to ensure datasets are available
echo "Running dataset download tests to ensure datasets are available for performance tests..."
if ! pytest benchmarking/test_download_datasets.py -s $DEBUG_ARG; then
  echo "ERROR: Dataset download tests failed. Aborting performance tests."
  exit 1
fi

echo "Datasets downloaded successfully. Running performance tests..."
# Run performance tests
if [ "$NSYS_ENABLED" -eq 1 ]; then
  nsys_var="--gpu-metrics-devices=all -t cuda,nvtx,osrt,opengl,python-gil,nvvideo"
  nsys profile $nsys_var -o "profiler_nvtx_real_batch.nsys-$(date +'%Y-%m-%d_%H-%M-%S')-rep" pytest -s $DEBUG_ARG -k "test_decode_performance"
else
  pytest benchmarking/ -s $DEBUG_ARG -k "test_decode_performance"
fi

if [ -f latest_results_dir.txt ]; then
  RESULTS_DIR=$(cat latest_results_dir.txt)
  if [ -f "${RESULTS_DIR}/test_results.json" ]; then
    python benchmarking/plot_results.py "${RESULTS_DIR}/test_results.json" --output-dir "${RESULTS_DIR}" --plot
  fi
fi
