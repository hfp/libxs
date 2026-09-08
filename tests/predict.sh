#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2009-2026 Hans Pabst                                          #
# Copyright (c) 2009-2026 Intel Corporation                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Absent inputs, end to end through the sample rather than through a stub.
#
# A gap is a value the corpus does not have, not a value that happens to be
# zero, and the two are only distinguishable in what the model refuses to do
# with them. A mode that reads coordinates independently answers from the ones
# it has; a rotation mixes every coordinate into every output and a tree sorts
# on one at a time, so neither can express a gap and both refuse to build.
# Refusing is the property under test: the failure it prevents is a model that
# answers confidently from a coordinate it never had.
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)
SAMPLES="${HERE}/../samples/predict"

# What is missing is reported and the script yields rather than failing: the
# corpus is an 11 MB file that only the source tree carries, so a build tree
# that did not copy it has nothing to test rather than something broken. The
# reason goes to stderr because tests/test.sh discards stdout and prints
# stderr beside the verdict, which is what makes a skip visible as a skip.
skip() {
  >&2 echo "$1"
  exit 0
}

cd "${SAMPLES}" 2>/dev/null || skip "no ${SAMPLES} to test"
if [ ! -e ./predict_crystal.x ]; then skip "predict_crystal.x is not built"; fi
if [ ! -e ./predict_crystal.csv ]; then
  skip "predict_crystal.csv is not available"
fi

# a small training split and a capped corpus keep the run short; the contract
# does not depend on either. Scoring costs one pass over the test split per
# training point, so the cap is quadratic where the split is linear: it is what
# brings the whole script from minutes to seconds and keeps it affordable under
# a sanitizer. The rows are not ordered by label, hence the head keeps the mix.
NROWS=6000
CSV=$(mktemp)
trap 'rm -f ${CSV}' EXIT
head -n $((NROWS + 1)) ./predict_crystal.csv >"${CSV}"

RUN="./predict_crystal.x ${CSV} 0.2 2 0"

for MODE in none fisher setdiff; do
  OUT=$(${RUN} "${MODE}" gaps0.15 2>&1) || true
  if ! echo "${OUT}" | grep -q "^Accuracy:"; then
    echo "${MODE} did not answer over a corpus with gaps"
    echo "${OUT}" | tail -3
    exit 1
  fi
done

for MODE in pca rf hknn; do
  # a refusal is reported through the exit code as well as the text
  OUT=$(${RUN} "${MODE}" gaps0.15 2>&1) || true
  if ! echo "${OUT}" | grep -q "^Refused:"; then
    echo "${MODE} accepted a corpus with gaps instead of refusing it"
    echo "${OUT}" | tail -3
    exit 1
  fi
done

# the sentinel resolves to a real mode and the model reports which one, so a
# caller can tell what was chosen
OUT=$(${RUN} 2>&1) || true
if ! echo "${OUT}" | grep -q "^Decomposition: .* (selected at build)$"; then
  echo "the default did not select a decomposition"
  echo "${OUT}" | tail -3
  exit 1
fi
AUTO=$(echo "${OUT}" | sed -n 's|^Accuracy: [0-9]*/[0-9]* = \([0-9.]*\)%$|\1|p')

# every mode answers the same corpus without gaps, gaps being the only variable.
# Each run is read more than once rather than repeated: a mode named on the
# command line is reported as requested rather than selected, and pca is the
# baseline the selection is measured against below.
PCA=""
for MODE in none fisher setdiff pca rf hknn; do
  OUT=$(${RUN} "${MODE}" 2>&1) || true
  if ! echo "${OUT}" | grep -q "^Accuracy:"; then
    echo "${MODE} failed on a corpus with no gaps at all"
    echo "${OUT}" | tail -3
    exit 1
  fi
  if [ "none" = "${MODE}" ] &&
     ! echo "${OUT}" | grep -q "^Decomposition: RAW (requested)$"
  then
    echo "a requested decomposition was not reported as requested"
    echo "${OUT}" | tail -3
    exit 1
  fi
  if [ "pca" = "${MODE}" ]; then
    PCA=$(echo "${OUT}" | sed -n 's|^Accuracy: [0-9]*/[0-9]* = \([0-9.]*\)%$|\1|p')
  fi
done

# selecting is worth what it costs: the choice beats the mode this corpus
# punishes, which is the whole claim behind making it the default
if [ -z "${AUTO}" ] || [ -z "${PCA}" ]; then
  echo "could not read an accuracy back from the sample"
  exit 1
fi
if [ "$(echo "${AUTO} ${PCA}" | awk '{print ($1 >= $2)}')" != "1" ]; then
  echo "the selected mode (${AUTO}%) lost to a fixed one (${PCA}%)"
  exit 1
fi

# What a stored model can still be asked for. Where the mode was selected, the
# file carries the corpus and a loaded model rebuilds under another mode; where
# the caller named it, the file carries that mode alone and the switch is
# declined. Both are checked, because a switch that silently answers from a
# corpus that is not there is the failure this pair exists to catch - it
# segfaulted once, in exactly the second case.
OUT=$(TEST=1 ${RUN} 2>&1) || true
if ! echo "${OUT}" | grep -q "^Method switch: .* to .* over [0-9]* entries$"; then
  echo "a selected mode did not survive a round trip as another mode"
  echo "${OUT}" | grep -E "^Reloaded|^Method switch" || echo "${OUT}" | tail -3
  exit 1
fi
if echo "${OUT}" | grep -q "^Reloaded: 0 clusters"; then
  echo "a selected mode was stored without the corpus the switch needs"
  exit 1
fi

OUT=$(TEST=1 ${RUN} rf 2>&1) || true
if ! echo "${OUT}" | grep -q "^Method switch: declined"; then
  echo "a named mode was switched after storing, which its file cannot support"
  echo "${OUT}" | grep -E "^Reloaded|^Method switch" || echo "${OUT}" | tail -3
  exit 1
fi
if ! echo "${OUT}" | grep -q "^Reloaded: 0 clusters"; then
  echo "a named mode stored a partition it never reads"
  exit 1
fi

echo "OK"
