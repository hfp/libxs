#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

cd "${HERE}/../samples/ozaki"
./test-check.sh gemm
./test-check.sh gemm  16  20 350 1 0  1 0.0 350 350 1000
./test-check.sh gemm  23  21  32 0 1 -1 0.5  32  32 1000
./test-check.sh gemm 200 200 256 1 1  1 0.0 256 256 1000
