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

/** Forward index map: original position I -> shuffled position. */
#define LIBXS_SHUFFLE_INDEX(I, N, C, B) \
  ((size_t)(N) - 1 - ((size_t)(C) * (size_t)(I) + (size_t)(B)) % (size_t)(N))
/** Inverse index map: shuffled position J -> original position. */
#define LIBXS_UNSHUFFLE_INDEX(J, N, CINV, B) \
  ((size_t)(CINV) * (((size_t)(N) - 1 - (size_t)(B) % (size_t)(N) \
    + (size_t)(N) - (size_t)(J)) % (size_t)(N)) % (size_t)(N))


/** Sorting method for libxs_sort_smooth. */
typedef enum libxs_sort_t {
  LIBXS_SORT_NONE     = 0,
  LIBXS_SORT_IDENTITY = 1,
  LIBXS_SORT_NORM     = 2,
  LIBXS_SORT_MEAN     = 3,
  LIBXS_SORT_GREEDY   = 4,
  LIBXS_SORT_MORTON   = 5,
  LIBXS_SORT_HILBERT  = 6
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

/**
 * Split callback for custom k-d tree construction.
 * Receives points, index array (node slice), node count, depth,
 * and the current number of leaves already assigned.
 * Writes chosen dimension into *dim and split position into *pos
 * (number of elements going left, 1..count-1).
 * The callback may reorder idx[0..count-1] so that idx[0..pos-1]
 * are the left entries and idx[pos..count-1] are the right entries.
 * If the callback reorders idx, the partition function skips its
 * internal quickselect.
 * Return 0 for a valid split, nonzero to force a leaf.
 */
LIBXS_EXTERN_C typedef int (*libxs_kdtree_split_t)(
  int* dim, int* pos, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx);

/** Configuration for k-d tree build and partition. */
typedef struct libxs_kdtree_config_t {
  int min_leaf;
  libxs_kdtree_split_t split;
  void* ctx;
} libxs_kdtree_config_t;

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
 * N-dimensional Hilbert curve index. Maps ndims coordinates to a
 * 64-bit key with strong locality guarantees (no quadrant-boundary
 * jumps). Each coordinate is quantized to floor(64/ndims) bits.
 * coords[k] must be in [0, 2^bits_per_dim).
 */
LIBXS_API uint64_t libxs_hilbert(const unsigned int coords[], int ndims);

/**
 * Finite-bit N-dimensional Hilbert curve rank. Maps ndims coordinates
 * to a key using bits_per_dim bits per coordinate.
 */
LIBXS_API uint64_t libxs_hilbert_bits(
  const unsigned int coords[], int ndims, int bits_per_dim);

/**
 * Inverse N-dimensional Hilbert curve index. Maps a 64-bit key back
 * to ndims coordinates using floor(64/ndims) bits per coordinate.
 */
LIBXS_API void libxs_hilbert_decode(
  uint64_t code, unsigned int coords[], int ndims);

/**
 * Finite-bit inverse N-dimensional Hilbert curve rank.
 */
LIBXS_API void libxs_hilbert_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim);

/**
 * N-dimensional Morton code (Z-order curve). Bit-interleaves ndims
 * coordinates into a 64-bit key. Each coordinate is quantized to
 * floor(64/ndims) bits. coords[k] must be in [0, 2^bits_per_dim).
 */
LIBXS_API uint64_t libxs_morton(const unsigned int coords[], int ndims);

/**
 * Finite-bit N-dimensional Morton code. Maps ndims coordinates to a key
 * using bits_per_dim bits per coordinate.
 */
LIBXS_API uint64_t libxs_morton_bits(
  const unsigned int coords[], int ndims, int bits_per_dim);

/**
 * Inverse N-dimensional Morton code. Deinterleaves a 64-bit key back
 * to ndims coordinates using floor(64/ndims) bits per coordinate.
 */
LIBXS_API void libxs_morton_decode(
  uint64_t code, unsigned int coords[], int ndims);

/**
 * Finite-bit inverse N-dimensional Morton code.
 */
LIBXS_API void libxs_morton_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim);

/**
 * Stratify higher-dimensional coordinates into a lower-dimensional
 * Morton layout. Encodes src_coords as a source-dimensional Morton rank
 * and decodes that rank into the smallest dst_ndims Morton layout that
 * preserves the source rank bits.
 */
LIBXS_API int libxs_stratify_morton(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims);

/**
 * Finite-bit Morton stratification. Encodes src_coords with src_bits
 * bits per source coordinate and decodes into dst_ndims coordinates with
 * dst_bits bits per destination coordinate. Requires
 * dst_ndims * dst_bits >= src_ndims * src_bits.
 */
LIBXS_API int libxs_stratify_morton_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits);

/**
 * Stratify higher-dimensional coordinates into a lower-dimensional
 * Hilbert layout. Encodes src_coords as a source-dimensional Hilbert rank
 * and decodes that rank into the smallest dst_ndims Hilbert layout that
 * preserves the source rank bits.
 */
LIBXS_API int libxs_stratify_hilbert(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims);

/**
 * Finite-bit Hilbert stratification. Encodes src_coords with src_bits
 * bits per source coordinate and decodes into dst_ndims coordinates with
 * dst_bits bits per destination coordinate. Requires
 * dst_ndims * dst_bits >= src_ndims * src_bits.
 */
LIBXS_API int libxs_stratify_hilbert_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits);

/**
 * Build a k-d tree in-place for n points in ndims dimensions.
 * Points are stored row-major: pts[i*stride + k] is coordinate k of point i.
 * stride >= ndims (allows for padding or interleaved auxiliary data).
 * The index array idx[0..n-1] is rearranged to encode the implicit tree.
 * Initialize idx to identity {0, 1, ..., n-1} before calling.
 * config may be NULL (round-robin median, leaf at 1 element).
 */
LIBXS_API void libxs_kdtree_build(
  const double* pts, int* idx, int n, int ndims, int stride,
  const libxs_kdtree_config_t* config);

/**
 * Partition n points into leaves and write leaf IDs into assignments[0..n-1].
 * Returns the number of leaves. Same tree logic as build, but produces
 * cluster assignments instead of a query-tree index.
 * config may be NULL (round-robin median, leaf at 1 element).
 */
LIBXS_API int libxs_kdtree_partition(
  const double* pts, int* idx, int n, int ndims, int stride,
  int* assignments, const libxs_kdtree_config_t* config);

/**
 * Find the nearest point to query[0..ndims-1] within squared Euclidean
 * distance max_dist2. Returns the point index (into original pts layout)
 * or -1 if no point is within range. The used array (may be NULL) marks
 * points to skip (non-zero = skip).
 */
LIBXS_API int libxs_kdtree_nearest(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2);

/** Convenience wrappers for 2D (interleaved x,y layout). */
LIBXS_API_INLINE void libxs_kdtree2d_build(double* pts, int* idx, int n) {
  libxs_kdtree_build(pts, idx, n, 2, 2, NULL);
}
LIBXS_API_INLINE int libxs_kdtree2d_nearest(const double* pts, const int* idx,
  const unsigned char* used, int n, double x, double y, double max_dist2)
{
  double q[2];
  q[0] = x; q[1] = y;
  return libxs_kdtree_nearest(pts, idx, used, n, 2, 2, q, max_dist2);
}

/** In-place shuffle: element i goes to (N-1) - ((C*i + offset) mod N). */
LIBXS_API int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle, size_t offset,
  /** If NULL, the default value is one. */
  const size_t* nrepeat);

/** Out-of-place shuffle: dst[k] = src[(N-1) - ((C*k + offset) mod N)]. */
LIBXS_API int libxs_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle, size_t offset,
  /** If NULL, the default value is one. If zero, an ordinary copy is performed. */
  const size_t* nrepeat);

/** Determines the number of calls to restore the original data (libxs_shuffle2). */
LIBXS_API size_t libxs_unshuffle(
  /** The number of elements to be unshuffled. */
  size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle);

/** Single-pass inverse of libxs_shuffle2 (uses modular inverse of coprime). */
LIBXS_API int libxs_unshuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  /** Shall be co-prime to count-argument; uses libxs_coprime2(count) if shuffle=NULL. */
  const size_t* shuffle, size_t offset,
  /** If NULL, the default value is one. If zero, an ordinary copy is performed. */
  const size_t* nrepeat);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PERM_H*/
