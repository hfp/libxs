#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXS library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxs/                          #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

cd "${HERE}/../samples/ozaki"
./wrap-test.sh gemm
./wrap-test.sh gemm  16  20 350 1 0  1 0.0  35 350 1000
./wrap-test.sh gemm  23  21  32 0 1 -1 0.5  32  32 1000
./wrap-test.sh gemm 200 200 256 1 1  1 0.0 256 256 1000
