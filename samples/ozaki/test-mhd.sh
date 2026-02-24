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
DIR=${1:-.}
WRAP=${HERE}/gemm-wrap.x

if [ ! -x "${WRAP}" ]; then
  >&2 echo "ERROR: ${WRAP} not found"
  exit 1
fi

NPAIRS=0
NFAIL=0

for A in "${DIR}"/gemm-*-a.mhd; do
  B="${A%-a.mhd}-b.mhd"
  if [ ! -f "${B}" ]; then
    >&2 echo "SKIP $(basename "${A}"): no matching B-file"
    continue
  fi
  ID=$(basename "${A}" | sed 's/gemm-\(.*\)-a\.mhd/\1/')
  NPAIRS=$((NPAIRS + 1))
  RESULT=0
  OUTPUT=$(${WRAP} "${A}" "${B}" 2>&1) || RESULT=$?
  if [ "0" != "${RESULT}" ]; then
    echo "FAIL  gemm-${ID} (exit ${RESULT})"
    NFAIL=$((NFAIL + 1))
  else
    echo "OK    gemm-${ID}"
  fi
done

echo "---"
echo "${NPAIRS} pair(s), ${NFAIL} failure(s)"
exit ${NFAIL}
