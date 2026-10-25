#!/bin/sh
###############################################################################
# Run predict_params on all available tuning CSV files.
# Usage: ./predict_params.sh [predict_params args...]
###############################################################################

HERE="$(cd "$(dirname "$0")" && pwd)"
# LIBXSTREAM_ROOT if given, else a sibling checkout, else HOME
if [ ! "${LIBXSTREAM_ROOT}" ]; then
  if [ -d "${HERE}/../../../libxstream/samples/smm/params" ]; then
    LIBXSTREAM_ROOT="${HERE}/../../../libxstream"
  else
    LIBXSTREAM_ROOT="${HOME}/libxstream"
  fi
fi
PARAMS_DIR="${LIBXSTREAM_ROOT}/samples/smm/params"

if [ ! -x "${HERE}/predict_params.x" ]; then
  echo "predict_params.x not found in ${HERE}, building..."
  make -C "${HERE}" predict_params.x 2>&1 | tail -3
fi

# Deployed models are embedded, so the default names the method rather than
# leaving it to the trial, which scores exact matches only. A forest of eight
# trees matches hKNN on shapes it was not built from, at under twice its size,
# and its gate reads a calibrated probability where hKNN's confidence overstates.
ARGS="${*:-rf trees8}"

for csv in "${PARAMS_DIR}"/tune_multiply_*.csv; do
  if [ -f "${csv}" ]; then
    base="$(basename "${csv}" .csv)"
    echo "=== ${base} ==="
    # shellcheck disable=SC2086
    "${HERE}/predict_params.x" ${ARGS} "${csv}" "${PARAMS_DIR}/${base}.bin"
    echo
  fi
done
