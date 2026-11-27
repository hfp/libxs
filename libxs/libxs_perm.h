/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_PERM_H
#define LIBXS_PERM_H

#include "libxs_reg.h"

/** Shuffle: position I of the shuffled sequence reads this original position. */
#define LIBXS_SHUFFLE_INDEX(I, N, C, B) \
  ((size_t)(N) - 1 - ((size_t)(C) * (size_t)(I) + (size_t)(B)) % (size_t)(N))
/** Unshuffle: position J of the restored sequence reads this shuffled position. */
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

/** Comparator type for libxs_sort (tristate: negative, zero, positive). */
LIBXS_EXTERN_C typedef int (*libxs_sort_cmp_t)(
  const void* a, const void* b, void* ctx);

/**
 * Split callback for custom k-d tree construction.
 * Receives points, index array (node slice), node count, depth,
 * and the current number of leaves already assigned (in preorder).
 * Writes chosen dimension into *dim and split position into *pos
 * (number of elements going left, 1..count-1), and reorders
 * idx[0..count-1] so that idx[0..pos-1] are the left entries and
 * idx[pos..count-1] the right ones. *threshold arrives as HUGE_VAL:
 * left so, the tree places it midway between the two sides along
 * *dim (in pts); a callback splitting by other coordinates writes
 * its own (entries below it go left).
 * Return 0 for a valid split, nonzero to force a leaf.
 */
LIBXS_EXTERN_C typedef int (*libxs_kdtree_split_t)(
  int* dim, int* pos, double* threshold, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx);

/**
 * A node of a tree from libxs_kdtree_partition, in preorder. A split
 * (dim >= 0) has its left child next in the array and its right child
 * at index next; entries below threshold along dim went left. A leaf
 * (dim < 0) has its leaf id in next.
 */
typedef struct libxs_kdtree_node_t {
  double threshold;
  int dim, next;
} libxs_kdtree_node_t;

/**
 * Configuration for k-d tree build and partition; zero-initialize it
 * (libxs_kdtree_config_t config = { 0 }) so that unset fields stay off.
 * split:    NULL (median), libxs_kdtree_split_middle, or custom.
 * nodes:    partition only: receives the tree in preorder, 2*nleaves-1
 *           nodes, at most 2n-1 (NULL: no tree).
 * min_leaf: nodes of at most min_leaf points are leaves (0: down to 1).
 */
typedef struct libxs_kdtree_config_t {
  libxs_kdtree_split_t split;
  void* ctx;
  libxs_kdtree_node_t* nodes;
  int min_leaf;
} libxs_kdtree_config_t;

/** Sorted Hilbert-code spatial index over registry entries. */
LIBXS_EXTERN_C typedef struct libxs_spatial_t {
  uint64_t* codes;
  void** values;
  int n;
} libxs_spatial_t;


/**
 * Compute a row permutation that reorders rows of an m-by-n
 * column-major matrix for smoothness (decaying forward differences).
 *
 * method:   LIBXS_SORT_NONE..LIBXS_SORT_HILBERT
 * m, n:     matrix dimensions (m rows, n columns)
 * mat:      column-major matrix data (const, not modified)
 * ld:       leading dimension (ld >= m)
 * datatype: element type (F64, F32, and all integer types)
 * perm:     output array of m ints; perm[i] = source row for position i
 *
 * Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_sort_smooth(libxs_sort_t method, int m, int n,
  const void* mat, int ld, libxs_data_t datatype, int* perm);

/**
 * Indirect comparator: base is an array of indices and ctx the double keys they
 * index. libxs_sort then orders the indices and leaves the keys untouched,
 * which is what a caller sorting a payload alongside a value needs; passing it
 * to libxs_sort takes the same radix path the value comparators take.
 */
LIBXS_API int libxs_cmp_f64_idx(const void* a, const void* b, void* ctx);
/** Like libxs_cmp_f64_idx, but ctx holds int keys (e.g., offsets to group by). */
LIBXS_API int libxs_cmp_i32_idx(const void* a, const void* b, void* ctx);

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
 * Split rules of the tree itself, for libxs_kdtree_config_t.split. Both are
 * markers that build and partition recognize; called directly, they only
 * return nonzero. libxs_kdtree_split_median is the default (split NULL): the
 * median along the axes in turn, the tree that libxs_kdtree_nearest and
 * libxs_kdtree_knearest search. libxs_kdtree_split_middle splits at the middle
 * of the range of the axis with the largest extent (e.g., grid coordinates):
 * points below the middle go left, and a node without extent stays a leaf.
 */
LIBXS_API int libxs_kdtree_split_median(int* dim, int* pos, double* threshold,
  const double* pts, int* idx, int count, int depth, int nleaves, void* ctx);
LIBXS_API int libxs_kdtree_split_middle(int* dim, int* pos, double* threshold,
  const double* pts, int* idx, int count, int depth, int nleaves, void* ctx);

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
 * The leaf id a point descends to in the nodes libxs_kdtree_partition wrote
 * (config.nodes): below a split's threshold left, otherwise right. The point
 * is in the coordinates of the thresholds (those of pts, or of the split
 * callback that wrote them). Returns -1 for NULL arguments, or where a node
 * does not lead forward (no tree of partition).
 */
LIBXS_API int libxs_kdtree_descend(const libxs_kdtree_node_t nodes[], const double point[]);

/**
 * Find the nearest point to query[0..ndims-1] within squared Euclidean
 * distance max_dist2. Returns the point index (into original pts layout)
 * or -1 if no point is within range. The used array (may be NULL) marks
 * points to skip (non-zero = skip). Same tree requirement as for
 * libxs_kdtree_knearest (split NULL, min_leaf <= 1).
 */
LIBXS_API int libxs_kdtree_nearest(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2);

/**
 * Find the k nearest points to query[0..ndims-1] within squared Euclidean
 * distance max_dist2. Writes point indices to out_idx and their squared
 * distances to out_dist2, both SORTED ASCENDING by distance, and returns how
 * many were written (0..k; fewer than k when n or max_dist2 bounds it).
 * The used array (may be NULL) marks points to skip (non-zero = skip).
 *
 * The result is EXACT: the same k points a linear scan would pick, in the same
 * order. The tree only avoids visiting subtrees that cannot contain a closer
 * point, so this replaces a scan without changing what the scan would answer -
 * the distinction that matters when the neighbours feed an estimate rather than
 * only a ranking.
 *
 * A kd-tree stops paying as dimension rises: it beats a scan while n is large
 * against 2^ndims. Measured with k = 8, uniformly random points need n of about
 * 64 * 2^ndims before the tree is faster. Structured data (a lower intrinsic
 * dimension) pays earlier: 1.26x faster at 37 dimensions and 60,000 points,
 * whereas data without such structure was 3.4x slower than a scan at 28
 * dimensions and 100,000 points. Whether a scan is the better choice depends on
 * the data rather than on ndims alone: time a few queries both ways.
 *
 * Requires a tree built by libxs_kdtree_build with config == NULL (or a config
 * whose split is NULL or libxs_kdtree_split_median, and min_leaf <= 1), because
 * the query recomputes the split dimension as depth % ndims and treats every
 * node as split at its middle element - as libxs_kdtree_nearest does. A tree
 * built with another split function or larger leaves does not record that and
 * cannot be searched this way.
 */
LIBXS_API int libxs_kdtree_knearest(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2,
  int out_idx[], double out_dist2[], int k);

/**
 * As libxs_kdtree_knearest (same tree requirement and output), but the tree is
 * searched best bin first: the waiting subtree with the smallest lower bound of
 * its distance next, until none can hold a nearer point or budget points have
 * been examined (one distance each). With budget <= 0 there is no limit and the
 * result is exact (the same distances as libxs_kdtree_knearest); with a budget
 * it is the k nearest among the points examined, still sorted ascending. This is
 * the search for dimensions where the exact one visits most of the tree.
 */
LIBXS_API int libxs_kdtree_knearest_budget(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2,
  int out_idx[], double out_dist2[], int k, int budget);

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

/** In-place shuffle: element k receives element (N-1) - ((C*k + offset) mod N). */
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

/** Build spatial index from registry (extracts first 8 key bytes as Hilbert code). */
LIBXS_API int libxs_spatial_build(libxs_spatial_t* sp,
  const libxs_registry_t* registry);

/**
 * Build the index from parallel arrays instead of from registry keys. Use this
 * when the registry key is an identity (a content hash, say) rather than a
 * locality-preserving code: the caller supplies the codes, so the key is free to
 * be unique and the index still orders by locality. Arrays are copied.
 */
LIBXS_API int libxs_spatial_build_codes(libxs_spatial_t* sp,
  const uint64_t codes[], void* const values[], int n);

/** Find up to k entries nearest to query_code. Returns count found. */
LIBXS_API int libxs_spatial_nearest(const libxs_spatial_t* sp,
  uint64_t query_code, int k, void** out_values);

/** Release spatial index memory. */
LIBXS_API void libxs_spatial_destroy(libxs_spatial_t* sp);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_PERM_H*/
