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

export OZAKI_THRESHOLD=0

# a prerequisite this build tree need not carry, see tests/test.sh
skip() {
  >&2 echo "$1"
  exit 0
}

cd "${HERE}/../samples/ozaki" 2>/dev/null \
  || skip "no samples/ozaki to test"
if [ ! -x ./test-check.sh ]; then skip "./test-check.sh is absent"; fi
./test-check.sh gemm
./test-check.sh gemm  16  20 350 1 0  1 0.0 350 350 1000
./test-check.sh gemm  23  21  32 0 1 -1 0.5  32  32 1000
./test-check.sh gemm 200 200 256 1 1  1 0.0 256 256 1000

# Test Scheme 2 i8 fallback (GPU runtime toggle; CPU compile-time)
OZAKI_I8=1 ./test-check.sh gemm
OZAKI_I8=1 ./test-check.sh gemm 200 200 256 1 1  1 0.0 256 256 1000

# Test complex GEMM (ZGEMM/CGEMM via 3M method)
./test-check.sh zgemm
./test-check.sh zgemm  16  20 350 1 0  1 0.0 350 350 1000
./test-check.sh zgemm  23  21  32 0 1 -1 0.5  32  32 1000
./test-check.sh zgemm 200 200 256 1 1  1 0.0 256 256 1000

./test-check.sh cgemm
./test-check.sh cgemm  16  20 350 1 0  1 0.0 350 350 1000
./test-check.sh cgemm  23  21  32 0 1 -1 0.5  32  32 1000
./test-check.sh cgemm 200 200 256 1 1  1 0.0 256 256 1000
