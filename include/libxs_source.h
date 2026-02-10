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
#include "../src/libxs_cpuid_arm.c"
#include "../src/libxs_cpuid_rv64.c"
#include "../src/libxs_cpuid_x86.c"
#include "../src/libxs_hash.c"
#include "../src/libxs_hist.c"
#include "../src/libxs_main.c"
#include "../src/libxs_malloc.c"
#include "../src/libxs_math.c"
#include "../src/libxs_mem.c"
#include "../src/libxs_mhd.c"
#include "../src/libxs_reg.c"
#include "../src/libxs_rng.c"
#include "../src/libxs_sync.c"
#include "../src/libxs_timer.c"
#include "../src/libxs_utils.c"

#endif /*LIBXS_SOURCE_H*/
