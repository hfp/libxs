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

OBJDUMP=$(command -v objdump || true)
GREP=$(command -v grep)
SORT=$(command -v sort)

# a prerequisite this build tree need not carry, see tests/test.sh
skip() {
  >&2 echo "$1"
  exit 0
}

# Each source of the ozaki sample is compiled once per precision (X-d.o, X-s.o)
# and both land in one library. A helper whose code depends on GEMM_REAL_TYPE but
# whose name does not is then one name for two functions, and whatever merges by
# name (C++ inline, a weak symbol, a non-static definition) runs one copy for
# both precisions. Such a helper needs a precision prefix (see ozaki.h).
cd "${HERE}/../samples/ozaki/obj" 2>/dev/null \
  || skip "no samples/ozaki objects to test"
if [ ! "${OBJDUMP}" ] || ! ${OBJDUMP} --version 2>/dev/null | ${GREP} -q "GNU objdump"; then
  skip "GNU objdump is absent"
fi
# optimized, each object inlines and clones differently, which is no difference in the source
if ! ${GREP} -q "OPTFLAG1=-O0" ../.state 2>/dev/null; then
  skip "needs an unoptimized build (DBG=1)"
fi

TMPD=$(mktemp -d)
trap 'rm -rf "${TMPD}"' EXIT

# One line per instruction, prefixed by its function, with addresses and the
# numbering of compiler-generated names removed. A call target keeps its name:
# a call to a static function in the same section carries no relocation.
code() {
  ${OBJDUMP} -dr --no-show-raw-insn "$1" | awk '
    /^[0-9a-f]+ <.*>:$/ { sym = $2; gsub(/^<|>:$/, "", sym); next }
    /^Disassembly of section/ { sym = ""; next }
    "" != sym && 0 < NF {
      line = $0
      sub(/^[ \t]*[0-9a-f]+:[ \t]*/, "", line)
      gsub(/[0-9a-f]+ </, "<", line)
      gsub(/R_[A-Z0-9_]+[ \t]+/, "", line)
      gsub(/\.(text|rodata|data|bss)[^ \t,]*/, ".SEC", line)
      gsub(/\.[0-9]+/, "", line)
      gsub(/\t/, " ", line)
      print sym "\t" line
    }'
}

# one line per initialized object: name and its bytes (function names excluded)
data() {
  local SECTIONS
  SECTIONS=$(${OBJDUMP} -h "$1" | awk '$2 ~ /^\.(rodata|data)/ { printf " -j %s", $2 }')
  if [ ! "${SECTIONS}" ]; then return; fi
  # shellcheck disable=SC2086
  { ${OBJDUMP} -t "$1"; echo "@@"; ${OBJDUMP} -s ${SECTIONS} "$1"; } | awk '
    function hex2dec(x, i, n) {
      for (i = 1; i <= length(x); ++i) n = 16 * n + index("0123456789abcdef", substr(x, i, 1)) - 1
      return n
    }
    "@@" == $0 { syms = 1; next }
    0 == syms {
      if (5 <= NF && $(NF-2) ~ /^\.(rodata|data)/ && $(NF-1) !~ /^0+$/ && $NF !~ /^(__PRETTY_FUNCTION__|__func__)/) {
        sec[$NF] = $(NF-2); val[$NF] = $1; size[$NF] = $(NF-1)
      }
      next
    }
    /^Contents of section / { cur = $4; sub(/:$/, "", cur); next }
    "" != cur && /^ [0-9a-f]+ / {
      hex = substr($0, length($1) + 3, 35)
      gsub(/ /, "", hex)
      bytes[cur] = bytes[cur] hex
    }
    END {
      for (sym in sec) print sym "\t" substr(bytes[sec[sym]], 2 * hex2dec(val[sym]) + 1, 2 * hex2dec(size[sym]))
    }'
}

NPAIRS=0
FOUND=""
for D in *-d.o; do
  S=${D%-d.o}-s.o
  if [ ! -e "${D}" ] || [ ! -e "${S}" ]; then continue; fi
  for KIND in code data; do
    "${KIND}" "${D}" >"${TMPD}/d"
    "${KIND}" "${S}" >"${TMPD}/s"
    # names defined in both objects whose lines differ in content or count
    FOUND="${FOUND}$(awk -F '\t' -v obj="${D%-d.o}" '
      NR == FNR { n[$1]++; line[$1, n[$1]] = $2; next }
      !($1 in n) { next }
      { m[$1]++; if (m[$1] > n[$1] || line[$1, m[$1]] != $2) bad[$1] = 1 }
      END { for (sym in m) if ((sym in bad) || m[sym] != n[sym]) print obj ":" sym }' \
      "${TMPD}/d" "${TMPD}/s")
"
  done
  NPAIRS=$((NPAIRS + 1))
done
if [ "0" = "${NPAIRS}" ]; then skip "no double/float object pairs to compare"; fi

FOUND=$(${SORT} -u <<<"${FOUND}" | awk 'NF { printf " %s", $0 }')
if [ "${FOUND}" ]; then
  >&2 echo "precision-dependent but not precision-prefixed:${FOUND}"
  exit 1
fi
