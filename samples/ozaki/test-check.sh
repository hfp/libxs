#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

GREP=$(command -v grep)
CAT=$(command -v cat)

if [ ! "${GREP}" ] || [ ! "${CAT}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

if [ "$1" ]; then
  TEST=$1
  shift
else
  TEST=gemm
fi

EXE="${HERE}/${TEST}-wrap.x"
if [ ! -e "${EXE}" ] || [ ! -e .state ] || \
   [ "$(${GREP} 'BLAS=0' .state 2>/dev/null)" ];
then
  echo "SKIPPED (${EXE} not available or BLAS=0)"
  exit 0
fi

TMPF=$(mktemp)
trap 'rm -f ${TMPF}' EXIT

RESULT=0

# Scheme 1 (mantissa slicing): exact with default nslices
echo "-----------------------------------"
echo "CHECK: Scheme 1 (default)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 GEMM_VERBOSE=1 GEMM_OZAKI=1 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
  exit ${RESULT}
fi
if ${GREP} -q "CHECK:" "${TMPF}"; then
  echo "OK $(${GREP} "CHECK:" "${TMPF}")"
else
  echo "FAILED (no CHECK output)"
  exit 1
fi
echo
