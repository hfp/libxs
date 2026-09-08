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
EXEC=${HERE}/../scripts/tool_pexec.sh
SIZE=1000

export CHECK=1

# a prerequisite this build tree need not carry, see tests/test.sh
skip() {
  >&2 echo "$1"
  exit 0
}

if [ ! -x "${EXEC}" ]; then skip "${EXEC##*/} is absent"; fi
cd "${HERE}/../samples/memory" 2>/dev/null \
  || skip "no samples/memory to test"
if [ ! -x ./memcmp.x ]; then skip "./memcmp.x is not built"; fi

cat <<EOM | ${EXEC} -o /dev/null 2>/dev/null "$@"
./memcmp.x 0 0 $((SIZE*1)) 0
./memcmp.x 0 0 $((SIZE*2)) 0
./memcmp.x 0 0 $((SIZE*3)) 0
./memcmp.x 0 0 $((SIZE*1)) 1
./memcmp.x 0 0 $((SIZE*2)) 1
./memcmp.x 0 0 $((SIZE*3)) 1
./memcmp.x 0 0 $((SIZE*1)) 2
./memcmp.x 0 0 $((SIZE*2)) 2
./memcmp.x 0 0 $((SIZE*3)) 2
./memcmp.x 0 0 $((SIZE*1)) 4
./memcmp.x 0 0 $((SIZE*2)) 4
./memcmp.x 0 0 $((SIZE*3)) 4
./memcmp.x 0 0 $((SIZE*1)) 8
./memcmp.x 0 0 $((SIZE*2)) 8
./memcmp.x 0 0 $((SIZE*3)) 8
./memcmp.x 0 0 $((SIZE*1)) 16
./memcmp.x 0 0 $((SIZE*2)) 16
./memcmp.x 0 0 $((SIZE*3)) 16
./memcmp.x 0 0 $((SIZE*1)) 17
./memcmp.x 0 0 $((SIZE*2)) 17
./memcmp.x 0 0 $((SIZE*3)) 17
./memcmp.x 0 0 $((SIZE*1)) 23
./memcmp.x 0 0 $((SIZE*2)) 23
./memcmp.x 0 0 $((SIZE*3)) 23
./memcmp.x 0 0 $((SIZE*1)) 32
./memcmp.x 0 0 $((SIZE*2)) 32
./memcmp.x 0 0 $((SIZE*3)) 32
./memcmp.x 1 0 $((SIZE*1)) 1
./memcmp.x 2 0 $((SIZE*1)) 1
./memcmp.x 3 0 $((SIZE*1)) 1
./memcmp.x 4 0 $((SIZE*1)) 1
./memcmp.x 5 0 $((SIZE*1)) 1
./memcmp.x 7 0 $((SIZE*1)) 1
./memcmp.x 8 0 $((SIZE*1)) 1
./memcmp.x 9 0 $((SIZE*1)) 1
./memcmp.x 15 0 $((SIZE*1)) 1
./memcmp.x 16 0 $((SIZE*1)) 1
./memcmp.x 31 0 $((SIZE*1)) 1
./memcmp.x 32 0 $((SIZE*1)) 1
./memcmp.x 33 0 $((SIZE*1)) 1
./memcmp.x 63 0 $((SIZE*1)) 1
./memcmp.x 64 0 $((SIZE*1)) 1
./memcmp.x 65 0 $((SIZE*1)) 1
./memcmp.x 127 0 $((SIZE*1)) 1
./memcmp.x 128 0 $((SIZE*1)) 1
EOM
