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

cd "${HERE}/../samples/scratch"    2>/dev/null || exit 1
CHECK=0 ./scratch.x                 >/dev/null
CHECK=1 ./scratch.x                 >/dev/null
CHECK=1 ./scratch.x 43 8 "$(nproc)" >/dev/null
