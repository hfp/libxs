/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_SORT_H
#define LIBXS_SORT_H

#include "libxs.h"

#define LIBXS_SORT_NONE     0
#define LIBXS_SORT_IDENTITY 1
#define LIBXS_SORT_NORM     2
#define LIBXS_SORT_MEAN     3
#define LIBXS_SORT_GREEDY   4

/**
 * Compute a row permutation that reorders rows of an m-by-n
 * column-major matrix for smoothness (decaying forward differences).
 *
 * method:   LIBXS_SORT_NONE..LIBXS_SORT_GREEDY
 * m, n:     matrix dimensions (m rows, n columns)
 * mat:      column-major matrix data (const, not modified)
 * ld:       leading dimension (ld >= m)
 * datatype: element type (F64, F32, I32, U32, I16, U16, I8, U8)
 * perm:     output array of m ints; perm[i] = source row for position i
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_sort_smooth(int method, int m, int n,
  const void* mat, int ld, libxs_data_t datatype, int* perm);

#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_MATH_H) && !defined(LIBXS_CPUID_H) && !defined(LIBXS_GEMM_H) \
 && !defined(LIBXS_MHD_H) && !defined(LIBXS_TIMER_H) && !defined(LIBXS_MEM_H) \
 && !defined(LIBXS_SYNC_H) && !defined(LIBXS_UTILS_H) && !defined(LIBXS_RNG_H) \
 && !defined(LIBXS_HIST_H) && !defined(LIBXS_MALLOC_H) && !defined(LIBXS_REG_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_SORT_H*/
