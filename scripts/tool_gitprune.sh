#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXS library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxs/                              #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

GIT=$(command -v git)

if [ "${GIT}" ]; then
  ${GIT} reflog expire --expire=now --all
  ${GIT} gc --prune=now
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi

