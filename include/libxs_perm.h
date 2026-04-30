/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_PERM_H
#define LIBXS_PERM_H

#include "libxs.h"

/** Sorting method for libxs_sort_smooth. */
typedef enum libxs_sort_t {
  LIBXS_SORT_NONE     = 0,
  LIBXS_SORT_IDENTITY = 1,
  LIBXS_SORT_NORM     = 2,
  LIBXS_SORT_MEAN     = 3,
  LIBXS_SORT_GREEDY   = 4
} libxs_sort_t;

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
LIBXS_API int libxs_sort_smooth(libxs_sort_t method, int m, int n,
  const void* mat, int ld, libxs_data_t datatype, int* perm);

/** Out-of-place shuffling of data given by elemsize and count. */
LIBXS_API int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle,
  /** If NULL, the default value is one. */
  const size_t* nrepeat);

/** Out-of-place shuffling of data given by elemsize and count. */
LIBXS_API int libxs_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle,
  /** If NULL, the default value is one. If zero, an ordinary copy is performed. */
  const size_t* nrepeat);

/** Determines the number of calls to restore the original data (libxs_shuffle2). */
LIBXS_API size_t libxs_unshuffle(
  /** The number of elements to be unshuffled. */
  size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PERM_H*/
