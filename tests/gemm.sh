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

cd "${HERE}/../samples/gemm" 2>/dev/null \
  || skip "no samples/gemm to test"
for PROG in gemm_strided gemm_batch gemm_groups gemm_index; do
  if [ ! -x "./${PROG}.x" ]; then skip "./${PROG}.x is not built"; fi
done

export CHECK=1

run_check() {
  PROG=$1; shift
  GOLDEN=$1; shift
  L1=$(${PROG} "$@" 2>&1 | sed -n 's/.*checksum=//p')
  if [ "${L1}" != "${GOLDEN}" ]; then
    >&2 echo "FAILED: ${PROG} $* (LIBXS_GEMM_BACKEND=${LIBXS_GEMM_BACKEND}) => checksum=${L1} (expected ${GOLDEN})"
    exit 1
  fi
}

for LIBXS_GEMM_BACKEND in 0 1 2 3 4; do
  export LIBXS_GEMM_BACKEND

  # gemm_strided: default shape
  run_check ./gemm_strided.x 7.323033e+04
  # gemm_strided: small shapes
  run_check ./gemm_strided.x 3.014610e+00              1 1 1 100 1
  run_check ./gemm_strided.x 1.969160e+02              4 4 4 1000 1
  run_check ./gemm_strided.x 1.986959e+02              8 8 8 100 1
  # gemm_strided: non-square
  run_check ./gemm_strided.x 8.655755e+02              7 13 5 500 1
  # gemm_strided: medium/large
  run_check ./gemm_strided.x 1.093476e+03              16 16 16 200 1
  run_check ./gemm_strided.x 2.220874e+03              23 23 23 100 1
  run_check ./gemm_strided.x 4.227260e+03              32 32 32 100 1
  # gemm_strided: ld-padding (pad=1)
  run_check ./gemm_strided.x 1.371294e+02              8 8 8 100 1 1.0 1
  # gemm_strided: ld-padding + beta=0 (pad=2, non-square)
  run_check ./gemm_strided.x 2.890614e+02              7 13 5 500 1 0.0 2

  # gemm_batch: same shapes (dup=0, no duplicates)
  run_check ./gemm_batch.x 3.014610e+00                1 1 1 100 1 0
  run_check ./gemm_batch.x 1.986959e+02                8 8 8 100 1 0
  run_check ./gemm_batch.x 8.655755e+02                7 13 5 500 1 0
  run_check ./gemm_batch.x 2.220874e+03                23 23 23 100 1 0
  run_check ./gemm_batch.x 4.227260e+03                32 32 32 100 1 0
  # gemm_batch: sorted duplicates (dup=1)
  run_check ./gemm_batch.x 3.703216e+02                8 8 8 100 1 1
  # gemm_batch: shuffled duplicates (dup=2)
  run_check ./gemm_batch.x 1.734845e+02                8 8 8 100 1 2
  # gemm_batch: ld-padding (pad=1)
  run_check ./gemm_batch.x 1.371294e+02                8 8 8 100 1 0 1.0 1
  # gemm_batch: ld-padding + beta=0 (pad=1)
  run_check ./gemm_batch.x 6.956720e+01                8 8 8 100 1 0 0.0 1

  # gemm_index: same layout as strided => identical checksums
  run_check ./gemm_index.x 7.323033e+04
  run_check ./gemm_index.x 3.014610e+00                1 1 1 100 1
  run_check ./gemm_index.x 1.969160e+02                4 4 4 1000 1
  run_check ./gemm_index.x 1.986959e+02                8 8 8 100 1
  # gemm_index: non-square
  run_check ./gemm_index.x 8.655755e+02                7 13 5 500 1
  # gemm_index: medium/large
  run_check ./gemm_index.x 2.220874e+03                23 23 23 100 1
  run_check ./gemm_index.x 4.227260e+03                32 32 32 100 1
  # gemm_index: ld-padding (pad=1)
  run_check ./gemm_index.x 1.371294e+02                8 8 8 100 1 1.0 1
  # gemm_index: ld-padding + beta=0 (pad=2, non-square)
  run_check ./gemm_index.x 2.890614e+02                7 13 5 500 1 0.0 2

  # gemm_groups: per-group dispatch + batch loop
  run_check ./gemm_groups.x 1.305436e+02               1 100 1 8
  run_check ./gemm_groups.x 5.783653e+02               2 50 1
  run_check ./gemm_groups.x 4.638711e+02               3 100 1 4
  run_check ./gemm_groups.x 6.717660e+03               4 50 1 16
  # gemm_groups: ld-padding (pad=1)
  run_check ./gemm_groups.x 5.330265e+02               2 50 1 8 1.0 1
  # gemm_groups: ld-padding + beta=0 (pad=1)
  run_check ./gemm_groups.x 2.470291e+02               2 50 1 8 0.0 1

done
