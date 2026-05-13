#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
cd "${HERE}/../samples/syrk" 2>/dev/null || exit 1

PROG=./syrk.x
if [ ! -x "${PROG}" ]; then
  echo "SKIPPED: ${PROG} not found (no Fortran compiler?)"
  exit 0
fi
TOL=1E-10

run_check() {
  local N=$1 K=$2 R=$3
  local OUTPUT
  OUTPUT=$(${PROG} ${N} ${K} ${R} 2>&1)
  local ERR
  for ERR in $(echo "${OUTPUT}" | sed -n 's/.*max error.*: *//p'); do
    if awk "BEGIN{exit(${ERR} <= ${TOL} ? 0 : 1)}" 2>/dev/null; then
      :
    else
      >&2 echo "FAILED: ${PROG} ${N} ${K} ${R} => error=${ERR} (tol=${TOL})"
      exit 1
    fi
  done
}

# small path (n <= block, k <= block)
run_check 4 3 1
run_check 32 16 1
run_check 64 64 1

# blocked path (n > block or k > block), non-aligned remainders
run_check 100 77 1
run_check 200 200 1
run_check 500 500 1

# non-square k
run_check 128 300 1
run_check 300 50 1
