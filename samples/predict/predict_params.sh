#!/bin/sh
###############################################################################
# Run predict_params on all available tuning CSV files.
# Usage: ./predict_params.sh [path-to-libxstream]
###############################################################################

HERE="$(cd "$(dirname "$0")" && pwd)"
LIBXSTREAM_ROOT="${1:-${HOME}/libxstream}"
PARAMS_DIR="${LIBXSTREAM_ROOT}/samples/smm/params"

if [ ! -x "${HERE}/predict_params.x" ]; then
  echo "predict_params.x not found in ${HERE}, building..."
  make -C "${HERE}" GNU=1 predict_params.x 2>&1 | tail -3
fi

for csv in "${PARAMS_DIR}"/tune_multiply_*.csv; do
  if [ -f "${csv}" ]; then
    name="$(basename "${csv}" .csv | sed 's/tune_multiply_//')"
    echo "=== ${name} ==="
    "${HERE}/predict_params.x" "${csv}"
    echo
  fi
done
