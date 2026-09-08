#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
# a prerequisite this build tree need not carry, see tests/test.sh
skip() {
  >&2 echo "$1"
  exit 0
}

MEMDIR="${HERE}/../samples/memory"
if [ ! -d "${MEMDIR}" ]; then skip "no samples/memory to test"; fi

# out-of-place transpose via Fortran binary (square only)
NMB=1

MATCPY="${MEMDIR}/matcpyf.x"
if [ -x "${MATCPY}" ]; then
  run_otrans() {
    local N=$1 LD=$2 NREPEAT=$3
    local OUTPUT
    OUTPUT=$(${MATCPY} ${N} ${N} ${LD} ${LD} ${NREPEAT} ${NMB} 2>&1) || {
      >&2 echo "FAILED: ${MATCPY} ${N} ${N} ${LD} ${LD} ${NREPEAT}"
      exit 1
    }
    if echo "${OUTPUT}" | grep -qi "error"; then
      >&2 echo "FAILED: ${MATCPY} ${N} ${N} ${LD} ${LD} ${NREPEAT}"
      >&2 echo "${OUTPUT}"
      exit 1
    fi
  }
  run_otrans 1 1 1
  run_otrans 2 2 1
  run_otrans 7 7 1
  run_otrans 64 64 1
  run_otrans 63 64 1
  run_otrans 100 128 1
  run_otrans 256 256 1
else
  >&2 echo "./matcpyf.x is not built (no Fortran compiler?), otrans not tested"
fi

# in-place transpose via C binary
ITRANS="${MEMDIR}/itrans.x"
if [ ! -x "${ITRANS}" ]; then
  skip "./itrans.x is not built"
fi

run_itrans() {
  local M=$1 N=$2 LDI=$3 LDO=$4
  ${ITRANS} ${M} ${N} ${LDI} ${LDO} || {
    >&2 echo "FAILED: ${ITRANS} ${M} ${N} ${LDI} ${LDO}"
    exit 1
  }
}

# degenerate / empty shapes
run_itrans 0 0 0 0
run_itrans 1 0 1 1
run_itrans 0 1 1 1
run_itrans 1 1 1 1

# square, tight
run_itrans 2 2 2 2
run_itrans 3 3 3 3
run_itrans 7 7 7 7
run_itrans 16 16 16 16
run_itrans 64 64 64 64
run_itrans 100 100 100 100
run_itrans 255 255 255 255

# square, padded leading dimensions
run_itrans 7 7 8 8
run_itrans 63 63 64 64
run_itrans 100 100 128 128

# non-square (requires scratch)
run_itrans 4 8 4 8
run_itrans 8 4 8 4
run_itrans 13 7 16 8
run_itrans 64 32 64 32
run_itrans 100 200 128 200
run_itrans 1 64 1 64
run_itrans 64 1 64 1
