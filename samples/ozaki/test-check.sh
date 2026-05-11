#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# shellcheck disable=SC2011
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

GREP=$(command -v grep)
CAT=$(command -v cat)

if [ ! "${GREP}" ] || [ ! "${CAT}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

if [ "$1" ]; then
  TESTS=$1
else
  # Discover tests from built *-wrap.x executables
  TESTS="$(ls -1 "${HERE}"/*gemm-wrap.x 2>/dev/null \
    | xargs -I{} basename {} -wrap.x | sort -u)"
  if [ ! "${TESTS}" ]; then TESTS=dgemm; fi
fi
if [ $# -gt 0 ]; then shift; fi

if [ ! "${OZAKI_THRESHOLD}" ]; then
  export OZAKI_THRESHOLD=0
fi

TMPF=$(mktemp)
trap 'rm -f ${TMPF}' EXIT

for TEST in ${TESTS}; do

EXE="${HERE}/${TEST}-wrap.x"
if [ ! -e "${EXE}" ] || [ ! -e .state ] || \
   [ "$(${GREP} 'BLAS=0' .state 2>/dev/null)" ];
then
  echo "SKIPPED (${EXE} not available or BLAS=0)"
  continue
fi

RESULT=0

# Scheme 1 (mantissa slicing): exact with default nslices
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 1 (default)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 OZAKI_VERBOSE=1 OZAKI=1 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
if [ "0" != "${RESULT}" ]; then
  echo "FAILED[${RESULT}]: $(${CAT} "${TMPF}")"
  exit ${RESULT}
fi
if ${GREP} -q "CHECK:" "${TMPF}"; then
  echo "OK $(${GREP} "CHECK:" "${TMPF}")"
else
  echo "FAILED (no CHECK output)"
  exit 1
fi
echo

# Scheme 2 (CRT modular arithmetic): exact with default nprimes (u8)
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 2 (CRT u8)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 OZAKI_VERBOSE=1 OZAKI=2 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
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

# Scheme 2 i8 fallback: signed residues, moduli <= 128
# OZAKI_I8 is a runtime toggle for the GPU path (ozaki_opencl.c) and a
# compile-time toggle for the CPU path.  This test exercises the GPU i8
# path when a GPU is available, or harmlessly re-tests the u8 CPU path.
echo "-----------------------------------"
echo "CHECK [${TEST}]: Scheme 2 (CRT i8)"
if [ "$*" ]; then echo "args    $*"; fi
{ CHECK=-1 OZAKI_VERBOSE=1 OZAKI=2 OZAKI_I8=1 "${EXE}" "$@" 2>"${TMPF}"; } >/dev/null || RESULT=$?
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

done # TESTS