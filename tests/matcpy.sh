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

cd "${HERE}/../samples/memory" 2>/dev/null \
  || skip "no samples/memory to test"

PROG=./matcpyf.x
if [ ! -x "${PROG}" ]; then
  skip "${PROG} is not built (no Fortran compiler?)"
fi

NMB=1

run_check() {
  local M=$1 N=$2 LDI=$3 LDO=$4 NREPEAT=$5
  local OUTPUT
  OUTPUT=$(${PROG} ${M} ${N} ${LDI} ${LDO} ${NREPEAT} ${NMB} 2>&1) || {
    >&2 echo "FAILED: ${PROG} ${M} ${N} ${LDI} ${LDO} ${NREPEAT}"
    exit 1
  }
  if echo "${OUTPUT}" | grep -qi "error"; then
    >&2 echo "FAILED: ${PROG} ${M} ${N} ${LDI} ${LDO} ${NREPEAT}"
    >&2 echo "${OUTPUT}"
    exit 1
  fi
}

# square, tight leading dimensions
run_check 64 64 64 64 1
run_check 256 256 256 256 1

# square, padded leading dimensions
run_check 63 63 64 64 1
run_check 100 100 128 128 1

# non-square (only matcopy/zero, no transpose)
run_check 64 32 64 64 1
run_check 100 200 128 128 1
run_check 13 7 16 16 1

# degenerate shapes (vectors)
run_check 1 1 1 1 1
run_check 1 64 1 1 1
run_check 64 1 64 64 1

# null shapes (m=0 or n=0)
run_check 0 0 256 256 1
run_check 0 64 256 256 1
run_check 64 0 256 256 1
