#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
cd "${HERE}/../samples/gemm" 2>/dev/null || exit 1

export CHECK=1

run_check() {
  PROG=$1; shift
  GOLDEN=$1; shift
  L1=$(${PROG} "$@" 2>&1 | sed -n 's/.*checksum=//p')
  if [ "${L1}" != "${GOLDEN}" ]; then
    >&2 echo "FAILED: ${PROG} $* => checksum=${L1} (expected ${GOLDEN})"
    exit 1
  fi
}

# gemm_strided: default shape
run_check ./gemm_strided.x 73541782869087043584.000000
# gemm_strided: small shapes
run_check ./gemm_strided.x 120163.500000              1 1 1 100 1
run_check ./gemm_strided.x 196424170392.000000        4 4 4 1000 1
run_check ./gemm_strided.x 249194695776.000000        8 8 8 100 1
# gemm_strided: non-square
run_check ./gemm_strided.x 3099313327246.500000       7 13 5 500 1
# gemm_strided: medium/large
run_check ./gemm_strided.x 128208954089856.000000     16 16 16 200 1
run_check ./gemm_strided.x 404521753229923.500000     23 23 23 100 1
run_check ./gemm_strided.x 4082185644738048.000000    32 32 32 100 1
# gemm_strided: ld-padding (pad=1)
run_check ./gemm_strided.x 221816414566.000000        8 8 8 100 1 1.0 1
# gemm_strided: ld-padding + beta=0 (pad=2, non-square)
run_check ./gemm_strided.x 1208802289760.000000       7 13 5 500 1 0.0 2

# gemm_batch: same shapes (dup=0, no duplicates)
run_check ./gemm_batch.x 120163.500000                1 1 1 100 1 0
run_check ./gemm_batch.x 249194695776.000000          8 8 8 100 1 0
run_check ./gemm_batch.x 3099313327246.500000         7 13 5 500 1 0
run_check ./gemm_batch.x 404521753229923.500000       23 23 23 100 1 0
run_check ./gemm_batch.x 4082185644738048.000000      32 32 32 100 1 0
# gemm_batch: sorted duplicates (dup=1)
run_check ./gemm_batch.x 123378370656.000000          8 8 8 100 1 1
# gemm_batch: shuffled duplicates (dup=2)
run_check ./gemm_batch.x 127267276896.000000          8 8 8 100 1 2
# gemm_batch: ld-padding (pad=1)
run_check ./gemm_batch.x 221816414566.000000          8 8 8 100 1 0 1.0 1
# gemm_batch: ld-padding + beta=0 (pad=1)
run_check ./gemm_batch.x 110907933943.000000          8 8 8 100 1 0 0.0 1

# gemm_index: same layout as strided => identical checksums
run_check ./gemm_index.x 73541782869087043584.000000
run_check ./gemm_index.x 120163.500000                1 1 1 100 1
run_check ./gemm_index.x 196424170392.000000          4 4 4 1000 1
run_check ./gemm_index.x 249194695776.000000          8 8 8 100 1
# gemm_index: non-square
run_check ./gemm_index.x 3099313327246.500000         7 13 5 500 1
# gemm_index: medium/large
run_check ./gemm_index.x 404521753229923.500000       23 23 23 100 1
run_check ./gemm_index.x 4082185644738048.000000      32 32 32 100 1
# gemm_index: ld-padding (pad=1)
run_check ./gemm_index.x 221816414566.000000          8 8 8 100 1 1.0 1
# gemm_index: ld-padding + beta=0 (pad=2, non-square)
run_check ./gemm_index.x 1208802289760.000000         7 13 5 500 1 0.0 2

# gemm_groups
run_check ./gemm_groups.x 6750768.000000              1 100 1 8
run_check ./gemm_groups.x 118725276.000000            2 50 1
run_check ./gemm_groups.x 112034040.000000            3 100 1 4
run_check ./gemm_groups.x 41892221964.000000          4 50 1 16
# gemm_groups: ld-padding (pad=1)
run_check ./gemm_groups.x 101110935.000000            2 50 1 8 1.0 1
# gemm_groups: ld-padding + beta=0 (pad=1)
run_check ./gemm_groups.x 50547549.000000             2 50 1 8 0.0 1
