#!/bin/sh
###############################################################################
# Run predict_params on all available tuning CSV files.
# Usage: ./predict_params.sh [predict_params args...]
###############################################################################

HERE="$(cd "$(dirname "$0")" && pwd)"
LIBXSTREAM_ROOT="${HOME}/libxstream"
PARAMS_DIR="${LIBXSTREAM_ROOT}/samples/smm/params"

if [ ! -x "${HERE}/predict_params.x" ]; then
  echo "predict_params.x not found in ${HERE}, building..."
  make -C "${HERE}" GNU=1 predict_params.x 2>&1 | tail -3
fi

ARGS="${@:-hknn compress}"

for csv in "${PARAMS_DIR}"/tune_multiply_*.csv; do
  if [ -f "${csv}" ]; then
    base="$(basename "${csv}" .csv)"
    echo "=== ${base} ==="
    "${HERE}/predict_params.x" ${ARGS} "${csv}" "${PARAMS_DIR}/${base}.bin"
    echo
  fi
done
