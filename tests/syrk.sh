#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
# a prerequisite this build tree need not carry, see tests/test.sh
skip() {
  >&2 echo "$1"
  exit 0
}

cd "${HERE}/../samples/syrk" 2>/dev/null \
  || skip "no samples/syrk to test"

PROG=./syrk.x
if [ ! -x "${PROG}" ]; then
  skip "${PROG} is not built (no Fortran compiler?)"
fi
TOL=1E-10

# launcher of the program (empty, or mpiexec -n NP)
RUN=()

# arguments: N K nrepeat [direct [tasks [block]]]
run_check() {
  local OUTPUT
  OUTPUT=$("${RUN[@]}" ${PROG} "$@" 2>&1)
  local ERR
  for ERR in $(echo "${OUTPUT}" | sed -n 's/.*max error.*: *//p'); do
    if awk "BEGIN{exit(${ERR} <= ${TOL} ? 0 : 1)}" 2>/dev/null; then
      :
    else
      >&2 echo "FAILED: ${RUN[*]} ${PROG} $* (LIBXS_GEMM_BACKEND=${LIBXS_GEMM_BACKEND}, LIBXS_SYRK_BLAS=${LIBXS_SYRK_BLAS}) => error=${ERR} (tol=${TOL})"
      exit 1
    fi
  done
}

for LIBXS_GEMM_BACKEND in 0 1 2 3 4; do
  export LIBXS_GEMM_BACKEND
  for LIBXS_SYRK_BLAS in 0 1; do
    export LIBXS_SYRK_BLAS

    # small path (n <= block, k <= block)
    run_check 4 3 1
    run_check 32 16 1
    run_check 64 64 1

    # blocked path (n > block or k > block), non-aligned remainders
    run_check 100 77 1
    run_check 200 200 1
    run_check 500 500 1

    # non-square k
    run_check 128 300 1
    run_check 300 50 1

    # one step, i.e., dispatch and call (the arguments in the order of BLAS)
    run_check 24 16 1 1
    run_check 200 77 1 1

  done
done

# distributed SYRK if the sample is built with MPI (make MPI=1), e.g.,
# MPIEXEC="mpiexec --oversubscribe" for Open MPI on fewer cores than ranks
if ${PROG} 4 3 1 2>&1 | grep -q "(distributed)"; then
  read -r -a LAUNCH <<<"${MPIEXEC:-mpiexec}"
  if command -v "${LAUNCH[0]}" >/dev/null; then
    export LIBXS_GEMM_BACKEND=0 LIBXS_SYRK_BLAS=1
    for NP in 2 3 4; do
      RUN=("${LAUNCH[@]}" -n "${NP}")
      # ranks times threads, hence few threads
      export OMP_NUM_THREADS=$((5 - NP))
      run_check 100 77 1 0 4 7
      run_check 200 200 1 0 4 32
      run_check 300 50 1 0 4 64
      # fewer columns of A than ranks (empty stripes), blocks of one
      run_check 64 3 1 0 4 5
      run_check 4 3 1 0 4 1
    done
  else
    >&2 echo "${LAUNCH[0]} not found: distributed SYRK not tested"
  fi
fi
