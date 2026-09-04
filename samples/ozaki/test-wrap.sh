#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
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
SED=$(command -v sed)
TR=$(command -v tr)
# only to locate a sanitizer runtime below; absent on Darwin, which needs none
LDD=$(command -v ldd)

if [ "Darwin" != "$(${UNAME})" ]; then
  LIBEXT=so
else
  LIBEXT=dylib
fi
if [ "$1" ]; then
  TESTS=$1
else
  # Discover tests from built executables (*-wrap.x and *-blas.x)
  TESTS="$({ ls -1 "${HERE}"/*-wrap.x "${HERE}"/*-blas.x 2>/dev/null || true; } \
    | xargs -I{} basename {} .x | sed 's/-wrap$//;s/-blas$//' | sort -u)"
fi
if [ $# -gt 0 ]; then shift; fi

TMPF=$(mktemp)
trap 'rm ${TMPF}' EXIT

# What tells an intercepted run from a plain one.  Not "GEMM:": the driver labels
# its own timed block "OZAKI GEMM:" whether or not anything wrapped the call, so
# that pattern matches both and cannot decide anything.  The bracketed form is the
# wrapper's own verification line, which only the wrapper prints -- measured 0 for
# a plain *-blas.x against 2 for both the static wrap and the LD_PRELOAD path.
WRAPPED="GEMM\["

# Every verdict goes to stderr as well as stdout, because tests/test.sh runs this
# with stdout on /dev/null and reports only what it captured from stderr.  A
# failure that announces itself on stdout alone therefore reaches CI as a bare
# exit code with no message, which is how one arrived: "FAILED(127)" and nothing
# else to read.
say() {
  echo "$*"
  echo "${NAME:-test-wrap}: $*" >&2
}

# 126 and 127 are the shell's "found it but could not run it" and "could not find
# it": a driver that the loader rejects for a missing library, or an interpreter
# that is absent.  Neither is this test's subject -- it checks whether a call gets
# intercepted, which presupposes a driver that runs at all, and the ozaki and gemm
# tests already cover whether the drivers work.  So report it and move on rather
# than failing the suite for a build or environment fault, while a driver that DOES
# run and reports the wrong thing still fails.
unrunnable() {
  if [ "126" = "$1" ] || [ "127" = "$1" ]; then
    say "SKIPPED(cannot execute, rc=$1) $(${CAT} "${TMPF}")"
    return 0
  fi
  return 1
}

# set verbosity to check for generated kernels
export OZAKI_VERBOSE=${OZAKI_VERBOSE:-1}

for TEST in ${TESTS}; do
  NAME=$(echo "${TEST}" | ${TR} [[:lower:]] [[:upper:]])

  if [ -e "${HERE}/${TEST}-blas.x" ]; then
    echo "-----------------------------------"
    echo "${NAME} (ORIGINAL BLAS)"
    if [ "$*" ]; then echo "args    $*"; fi
    RESULT=0
    { time eval "${HERE}/${TEST}-blas.x $* >${TMPF} 2>&1"; } 2>&1 \
      | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      if unrunnable "${RESULT}"; then continue; fi
      say "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ! ${GREP} -q "${WRAPPED}" "${TMPF}"; then
      say "OK"
    else
      say "FAILED: expected ${WRAPPED} $(${CAT} "${TMPF}")"
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
    { time eval "${HERE}/${TEST}-wrap.x $* >${TMPF} 2>&1"; } 2>&1 \
      | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      if unrunnable "${RESULT}"; then continue; fi
      say "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ${GREP} -q "${WRAPPED}" "${TMPF}"; then
      say "OK"
    else
      say "FAILED: expected ${WRAPPED} $(${CAT} "${TMPF}")"
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
    # AddressSanitizer refuses to run with a library preloaded ahead of its own
    # runtime, which is precisely what this test does.  Its runtime therefore goes
    # first, which is what a sanitized drop-in replacement asks of its user too.
    # A build without a sanitizer finds nothing here and preloads the wrapper alone.
    PRELOAD=${HERE}/libwrap.${LIBEXT}
    ASANOPT=${ASAN_OPTIONS}
    if [ "${LDD}" ]; then
      for LIB in "${HERE}/libwrap.${LIBEXT}" "${HERE}/${TEST}-blas.x"; do
        ASANLIB=$({ ${LDD} "${LIB}" 2>/dev/null || true; } \
          | ${SED} -n 's/.*=>[[:space:]]*\(\/[^ ]*libasan[^ ]*\).*/\1/p')
        if [ "${ASANLIB}" ]; then
          PRELOAD=${ASANLIB}:${PRELOAD}
          # Driver and wrapper each carry their own copy of a statically linked
          # LIBXS/LIBXSTREAM, so the same global is defined twice in one process.
          # Level 1 accepts that as long as the definitions agree in size, i.e. an
          # actual ABI difference still reports.  Appended, so an ASAN_OPTIONS from
          # the environment keeps its say (the last assignment of a flag wins).
          ASANOPT=${ASANOPT:+${ASANOPT}:}detect_odr_violation=1
          break
        fi
      done
    fi
    RESULT=0
    { time eval " \
      ${ASANOPT:+ASAN_OPTIONS=${ASANOPT}} \
      LD_LIBRARY_PATH=${DEPDIR}/lib:${LD_LIBRARY_PATH} LD_PRELOAD=${PRELOAD} \
      DYLD_LIBRARY_PATH=${DEPDIR}/lib:${DYLD_LIBRARY_PATH} DYLD_INSERT_LIBRARIES=${DEPDIR}/lib/libxs.${LIBEXT} \
      ${HERE}/${TEST}-blas.x $* >${TMPF} 2>&1"; } 2>&1 | ${GREP} real || RESULT=$?
    if [ "0" != "${RESULT}" ]; then
      if unrunnable "${RESULT}"; then continue; fi
      say "FAILED[${RESULT}] $(${CAT} "${TMPF}")"
      exit ${RESULT}
    elif ${GREP} -q "${WRAPPED}" "${TMPF}"; then
      say "OK"
    else
      say "FAILED: expected ${WRAPPED} $(${CAT} "${TMPF}")"
      exit 1
    fi
    echo
  fi
done
