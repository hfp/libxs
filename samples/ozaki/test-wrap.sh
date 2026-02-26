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
DEPDIR=${HERE}/../../..

UNAME=$(command -v uname)
GREP=$(command -v grep)
CAT=$(command -v cat)
TR=$(command -v tr)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi
if [ "$1" ]; then
  TESTS=$1
  shift
else
  TESTS="$(ls -1 "${HERE}"/*.c | xargs -I{} basename {} .c)"
fi

TMPF=$(mktemp)
trap 'rm ${TMPF}' EXIT

# set verbosity to check for generated kernels
export GEMM_VERBOSE=${GEMM_VERBOSE:-1}

for TEST in ${TESTS}; do
  NAME=$(echo "${TEST}" | ${TR} [[:lower:]] [[:upper:]])

  if [ -e "${HERE}/${TEST}-blas.x" ]; then
    echo "-----------------------------------"
    echo "${NAME} (ORIGINAL BLAS)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval "${HERE}/${TEST}-blas.x $*" 2>"${TMPF}"; } 2>&1 \
      | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ! ${GREP} -q "GEMM:" "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi

  if [ -e "${HERE}/${TEST}-wrap.x" ] && [ -e .state ] && \
     [ ! "$(${GREP} 'BLAS=0' .state)" ];
  then
    echo "-----------------------------------"
    echo "${NAME} (STATIC WRAP)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval "${HERE}/${TEST}-wrap.x $*" 2>"${TMPF}"; } 2>&1 \
      | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ${GREP} -q "GEMM:" "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi

  if [ -e "${HERE}/${TEST}-blas.x" ] && \
     [ -e "${HERE}/libwrap.${LIBEXT}" ];
  then
    echo "-----------------------------------"
    echo "${NAME} (LD_PRELOAD)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval " \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${HERE}/libwrap.${LIBEXT} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxs.${LIBEXT} \
      ${HERE}/${TEST}-blas.x $*" 2>"${TMPF}"; } 2>&1 | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      echo "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ${GREP} -q "GEMM:" "${TMPF}"; then
      echo "OK"
    else
      echo "FAILED"
      exit 1
    fi
    echo
  fi
done
