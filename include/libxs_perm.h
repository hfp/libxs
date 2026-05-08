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

/** Comparator type for libxs_sort (tristate: negative, zero, positive). */
LIBXS_EXTERN_C typedef int (*libxs_sort_cmp_t)(
  const void* a, const void* b, void* ctx);

/** Built-in comparators (enable fast paths when recognized). */
LIBXS_API int libxs_cmp_f64(const void* a, const void* b, void* ctx);
LIBXS_API int libxs_cmp_f32(const void* a, const void* b, void* ctx);
LIBXS_API int libxs_cmp_i32(const void* a, const void* b, void* ctx);
LIBXS_API int libxs_cmp_u32(const void* a, const void* b, void* ctx);

/**
 * Sort n elements of given size using comparator with context.
 * Built-in comparators (libxs_cmp_f64, etc.) trigger a fast path:
 *   ctx=NULL sorts base in-place; ctx!=NULL reads source from ctx
 *   and writes sorted result to base (out-of-place, no memcpy).
 * Unknown comparators use generic in-place heap sort.
 */
LIBXS_API void libxs_sort(void* base, int n, size_t size,
  libxs_sort_cmp_t cmp, void* ctx);

/**
 * 2D Hilbert curve index. Maps (x, y) grid coordinates to a
 * locality-preserving 1D index. Order is bits per axis (1..16),
 * producing a 2*order-bit key. Coordinates must be in [0, 2^order).
 */
LIBXS_API unsigned int libxs_hilbert2d(
  unsigned int x, unsigned int y, int order);

/**
 * Build a 2D k-d tree in-place. Points are n interleaved (x, y) pairs
 * in pts[0..2n-1]. The index array idx[0..n-1] is rearranged to
 * reflect the implicit tree structure (median splits alternating
 * on x and y). Initialize idx to identity {0, 1, ..., n-1} before
 * calling. After build, pts is unchanged; idx encodes the tree.
 */
LIBXS_API void libxs_kdtree2d_build(double* pts, int* idx, int n);

/**
 * Find the nearest point to (x, y) within squared Euclidean distance
 * max_dist2. Returns the point index (into original pts layout) or
 * -1 if no point is within range. The used array (may be NULL) marks
 * points to skip (non-zero = skip). This enables repeated queries
 * with consumption (mark returned index as used after each hit).
 */
LIBXS_API int libxs_kdtree2d_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n, double x, double y, double max_dist2);

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
