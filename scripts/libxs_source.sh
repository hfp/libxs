#!/usr/bin/env sh

SRCDIR=../src
HERE=$(cd "$(dirname "$0")" && pwd -P)
GREP=$(command -v grep)
GIT=$(command -v git)

if [ ! "${GREP}" ]; then
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
if [ "${GIT}" ] && [ ! "$(${GIT} ls-files "${HERE}/${SRCDIR}/libxs_main.c" 2>/dev/null)" ]; then
  GIT=""
fi
cat << EOM
/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXS_SOURCE_H
#define LIBXS_SOURCE_H

#if defined(LIBXS_MACROS_H)
# error Please do not include any LIBXS header other than libxs_source.h!
#endif
#if defined(LIBXS_BUILD)
# error LIBXS_BUILD cannot be defined for the header-only LIBXS!
#endif

/**
 * This header is intentionally called "libxs_source.h" since the followings block
 * includes *internal* files, and thereby exposes LIBXS's implementation.
 * The so-called "header-only" usage model gives up the clearly defined binary interface
 * (including support for hot-fixes after deployment), and requires to rebuild client
 * code for every (internal) change of LIBXS. Please make sure to only rely on the
 * public interface as the internal implementation may change without notice.
 */
EOM

if [ "$1" ]; then
  DSTDIR=$1
else
  DSTDIR=${SRCDIR}
fi

# determine order of filenames in directory list
export LC_ALL=C

# good-enough pattern to match a main function, and to exclude this translation unit
for FILE in $(cd "${HERE}/${SRCDIR}" && ${GREP} -L "main[[:space:]]*(.*)" ./*.c); do
  BASENAME=$(basename "${FILE}")
  if [ ! "${GIT}" ] || [ "$(${GIT} ls-files "${HERE}/${SRCDIR}/${BASENAME}" 2>/dev/null)" ]; then
    echo "#include \"${DSTDIR}/${BASENAME}\""
  fi
done

cat << EOM

#endif /*LIBXS_SOURCE_H*/
EOM

