#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# shellcheck disable=SC2086

HERE=$(cd "$(dirname "$0")" && pwd -P)
SORT=$(command -v sort)
GREP=$(command -v grep)
SED=$(command -v sed)
ENV=$(command -v env)
TR=$(command -v tr)
WC=$(command -v wc)

if [ ! "${GREP}" ] || [ ! "${SED}" ] || [ ! "${TR}" ] || [ ! "${WC}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

# disable interceptor to actually test against real LAPACK/BLAS
export LIBXS_GEMM_WRAP=${LIBXS_GEMM_WRAP:-0}

# disabled set of tests
#TESTS_DISABLED="headeronly"

# list of tests that produce "application shall be linked against LAPACK/BLAS" in case of BLAS=0
TESTS_NEEDBLAS="wrap.sh"
# grep pattern based on TESTS_NEEDBLAS
TESTS_NEEDBLAS_GREP=$(${SED} <<<"${TESTS_NEEDBLAS}" "s/[[:space:]][[:space:]]*/\\\\|/g" | ${SED} "s/\./\\\\./g")
# good-enough pattern to match main functions, and to include translation unit in test set
if [ ! "$*" ]; then
  TESTS="$(cd "${HERE}" && ${GREP} -l "main[[:space:]]*(.*)" ./*.c 2>/dev/null) \
    scratch.sh wrap.sh ozaki.sh"
  if [ "${SORT}" ]; then
    TESTS=$(${TR} <<<"${TESTS}" -s " " "\n" | ${SORT})
  fi
else
  TESTS="$*"
fi

if [ "${TESTS}" ] && [ "$(${GREP} 'BLAS=0' "${HERE}/../.state" 2>/dev/null)" ]; then
  TESTS=$(${GREP} <<<"${TESTS}" -v "${TESTS_NEEDBLAS_GREP}")
fi

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../lib:/usr/x86_64-w64-mingw32/sys-root/mingw/bin
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(command -v cygcheck)
  EXE=.exe
else
  if [ "$(command -v ldd)" ]; then
    LDD=ldd
  elif [ "$(command -v otool)" ]; then
    LDD="otool -L"
  else
    LDD="echo"
  fi
  EXE=.x
fi

echo "============="
echo "Running tests"
echo "============="

NTEST=1
NMAX=$(${WC} <<<"${TESTS}" -w | ${TR} -d " ")
for TEST in ${TESTS}; do
  NAME=$(${SED} <<<"${TEST}" 's/.*\///;s/\(.*\)\..*/\1/')
  printf "%02d of %02d: %-12s " "${NTEST}" "${NMAX}" "${NAME}"
  if [ "0" != "$(${GREP} <<<"${TESTS_DISABLED}" -q "${NAME}"; echo $?)" ]; then
    cd "${HERE}" || exit 1
    TESTX=$( \
      if [ -e "${HERE}/${NAME}.sh" ]; then \
        echo "${HERE}/${NAME}.sh"; \
      else \
        echo "${HERE}/${NAME}${EXE}"; \
      fi)
    if [ -e "${TESTX}" ]; then
      RESULT=0
      ERROR=$({ \
        ${ENV} LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HERE}/../lib" \
          DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${HERE}/../lib" \
          OMP_PROC_BIND=TRUE \
        ${TOOL_COMMAND} ${TESTX} ${TOOL_COMMAND_POST} \
        >/dev/null; } 2>&1) || RESULT=$?
    else
      ERROR="Test is missing"
      RESULT=1
    fi
  else
    ERROR="Test is disabled"
    RESULT=0
  fi
  if [ 0 != ${RESULT} ]; then
    echo "FAILED(${RESULT}) ${ERROR}"
    exit ${RESULT}
  else
    echo "OK ${ERROR}"
  fi
  NTEST=$((NTEST+1))
done
