#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# The synchronization samples, run for their verdict rather than their numbers.
#
# samples/sync/barrier.x checks what it measures before it reports it: a
# rendezvous that lets a task through early is fast and wrong, and a timing that
# does not say which of the two it timed is worth nothing. So it is run here at
# more than one team size, and only its exit code is read. The numbers it prints
# belong to a machine that is not shared, which a test host is not.
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
SAMPLES="${HERE}/../samples/sync"

# a prerequisite this build tree need not carry, see tests/test.sh
skip() {
  >&2 echo "$1"
  exit 0
}

cd "${SAMPLES}" 2>/dev/null || skip "no samples/sync to test"
if [ ! -e ./barrier.x ]; then skip "barrier.x is not built"; fi

# few rounds: the property is checked every round, so more of them buys
# confidence in the scheduler's choices and nothing about the barrier
for NTHREADS in 1 2 3 4; do
  ./barrier.x "${NTHREADS}" 2000 >/dev/null || exit 1
done

# an over-subscribed team is the case a spin barrier is worst at, and it must
# still be correct: the sample asks the runtime how many tasks it actually got.
# Capped, because over-subscribing a many-core host means spinners in the
# hundreds and the point is made by a handful of them.
NCORES=$(nproc 2>/dev/null || echo 2)
NOVER=$((NCORES * 2))
if [ 24 -lt "${NOVER}" ]; then NOVER=24; fi
./barrier.x "${NOVER}" 200 >/dev/null || exit 1

if [ -e ./sync.x ]; then
  ./sync.x 2 5 100 1000 10000 100 >/dev/null || exit 1
fi

echo "OK"
