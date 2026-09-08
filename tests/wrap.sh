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

cd "${HERE}/../samples/ozaki" 2>/dev/null \
  || skip "no samples/ozaki to test"
if [ ! -x ./test-wrap.sh ]; then skip "./test-wrap.sh is absent"; fi
./test-wrap.sh dgemm
./test-wrap.sh dgemm  16  20 350 1 0  1 0.0 350 350 1000
./test-wrap.sh dgemm  23  21  32 0 1 -1 0.5  32  32 1000
./test-wrap.sh dgemm 200 200 256 1 1  1 0.0 256 256 1000
