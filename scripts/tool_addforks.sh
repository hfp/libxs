#!/bin/bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXS library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxs/                              #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

CURL=$(command -v curl)
GREP=$(command -v grep)
CUT=$(command -v cut)
GIT=$(command -v git)

if [ "" != "${CURL}" ] && [ "" != "${GIT}" ] && \
   [ "" != "${GREP}" ] && [ "" != "${CUT}" ];
then
  for FORK in $(${CURL} -s https://api.github.com/repos/hfp/libxs/forks \
  | ${GREP} "\"html_url\"" | ${GREP} "libxs" | ${CUT} -d/ -f4);
  do
    ${GIT} remote add ${FORK} https://github.com/${FORK}/libxs.git
    ${GIT} fetch ${FORK}
  done
else
  echo "Error: missing prerequisites!"
  exit 1
fi

