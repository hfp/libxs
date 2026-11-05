/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_perm.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#define KNN_MAXN 512
#define KNN_MAXK 32
#define KNN_MAXD 8


static int check_sorted_f64(const double* data, int n)
{
  int i, result = 1;
  for (i = 1; i < n && 0 != result; ++i) {
    if (data[i - 1] > data[i]) result = 0;
  }
  return result;
}


static int check_sorted_f32(const float* data, int n)
{
  int i, result = 1;
  for (i = 1; i < n && 0 != result; ++i) {
    if (data[i - 1] > data[i]) result = 0;
  }
  return result;
}


static int check_sorted_i32(const int* data, int n)
{
  int i, result = 1;
  for (i = 1; i < n && 0 != result; ++i) {
    if (data[i - 1] > data[i]) result = 0;
  }
  return result;
}


static int cmp_by_abs(const void* a, const void* b, void* ctx)
{
  const double va = *(const double*)a, vb = *(const double*)b;
  const double aa = (va < 0 ? -va : va), ab = (vb < 0 ? -vb : vb);
  LIBXS_UNUSED(ctx);
  return (aa > ab) - (aa < ab);
}


static int cmp_indirect_f64(const void* a, const void* b, void* ctx)
{
  const double* keys = (const double*)ctx;
  const int ia = *(const int*)a, ib = *(const int*)b;
  return (keys[ia] > keys[ib]) - (keys[ia] < keys[ib]);
}


static uint64_t encode_morton_bits(
  const unsigned int coords[], int ndims, int bits_per_dim)
{
  uint64_t code = 0;
  int bit, dim;
  for (bit = bits_per_dim - 1; 0 <= bit; --bit) {
    for (dim = ndims - 1; 0 <= dim; --dim) {
      code = (code << 1) | ((coords[dim] >> bit) & 1u);
    }
  }
  return code;
}


static uint64_t encode_hilbert_bits(
  const unsigned int coords[], int ndims, int bits_per_dim)
{
  unsigned int transposed[64];
  uint64_t code = 0;
  int index, level;
  for (index = 0; index < ndims; ++index) transposed[index] = coords[index];
  { const unsigned int top = 1u << (bits_per_dim - 1);
    unsigned int side, mask, swap;
    for (side = top; side > 1; side >>= 1) {
      mask = side - 1;
      for (index = 0; index < ndims; ++index) {
        if (0 != (transposed[index] & side)) {
          transposed[0] ^= mask;
        }
        else {
          swap = (transposed[0] ^ transposed[index]) & mask;
          transposed[0] ^= swap; transposed[index] ^= swap;
        }
      }
    }
    for (index = 1; index < ndims; ++index) {
      transposed[index] ^= transposed[index - 1];
    }
    swap = 0;
    for (side = top; side > 1; side >>= 1) {
      if (0 != (transposed[ndims - 1] & side)) swap ^= (side - 1);
    }
    for (index = 0; index < ndims; ++index) transposed[index] ^= swap;
  }
  for (level = bits_per_dim - 1; 0 <= level; --level) {
    for (index = 0; index < ndims; ++index) {
      code = (code << 1) | ((transposed[index] >> level) & 1u);
    }
  }
  return code;
}


static void decode_morton_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim)
{
  int bit, dim;
  for (dim = 0; dim < ndims; ++dim) coords[dim] = 0;
  for (bit = 0; bit < bits_per_dim; ++bit) {
    for (dim = 0; dim < ndims; ++dim) {
      coords[dim] |= (unsigned int)(code & 1u) << bit;
      code >>= 1;
    }
  }
}


static void decode_hilbert_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim)
{
  int index, level;
  for (index = 0; index < ndims; ++index) coords[index] = 0;
  for (level = 0; level < bits_per_dim; ++level) {
    const int shift = level;
    int dim;
    for (dim = ndims - 1; 0 <= dim; --dim) {
      coords[dim] |= (unsigned int)(code & 1u) << shift;
      code >>= 1;
    }
  }
  if (1 < ndims) {
    const unsigned int top = 1u << (bits_per_dim - 1);
    unsigned int side;
    unsigned int swap = coords[ndims - 1] >> 1;
    for (index = ndims - 1; 0 < index; --index) {
      coords[index] ^= coords[index - 1];
    }
    coords[0] ^= swap;
    for (side = 2; 0 != side && side <= top; side <<= 1) {
      const unsigned int mask = side - 1;
      for (index = ndims - 1; 0 <= index; --index) {
        if (0 != (coords[index] & side)) {
          coords[0] ^= mask;
        }
        else {
          swap = (coords[0] ^ coords[index]) & mask;
          coords[0] ^= swap; coords[index] ^= swap;
        }
      }
    }
  }
}


/**
 * Linear-scan oracle: the k nearest by brute force, sorted ascending. The kd-tree
 * search must agree with this exactly - it is meant to avoid visiting subtrees
 * that cannot hold a closer point, not to approximate the answer, and only a
 * reference computed a different way can show the pruning has not cut too much.
 */
static int knn_oracle(const double* pts, int n, int ndims, int stride,
  const double* query, double max_dist2, int* out_idx, double* out_d2, int k)
{
  int count = 0, i;
  for (i = 0; i < n; ++i) {
    double d2 = 0;
    int dd;
    for (dd = 0; dd < ndims; ++dd) {
      const double d = pts[(size_t)i * stride + dd] - query[dd];
      d2 += d * d;
    }
    if (d2 <= max_dist2 && (count < k || d2 < out_d2[count - 1])) {
      int at = (count < k) ? count++ : (k - 1);
      while (0 < at && out_d2[at - 1] > d2) {
        out_d2[at] = out_d2[at - 1];
        out_idx[at] = out_idx[at - 1];
        --at;
      }
      out_d2[at] = d2;
      out_idx[at] = i;
    }
  }
  return count;
}


/**
 * Compare the tree against the oracle. Distances are compared rather than
 * indices, because ties at equal distance may legitimately resolve to different
 * points - with duplicate coordinates several points ARE the same answer, and
 * demanding one particular index would fail the test for a correct result.
 */
static int knn_check(const double* pts, int* idx, int n, int ndims,
  const double* query, double max_dist2, int k, int line)
{
  int tidx[KNN_MAXK], oidx[KNN_MAXK], i;
  double td2[KNN_MAXK], od2[KNN_MAXK];
  const int nt = libxs_kdtree_knearest(pts, idx, NULL, n, ndims, ndims,
    query, max_dist2, tidx, td2, k);
  const int no = knn_oracle(pts, n, ndims, ndims, query, max_dist2,
    oidx, od2, k);
  int result = 1;
  if (nt != no) {
    FPRINTF(stderr, "ERROR line #%i: knn found %i, oracle %i\n", line, nt, no);
    result = 0;
  }
  for (i = 0; i < nt && 0 != result; ++i) {
    if (td2[i] != od2[i]) {
      FPRINTF(stderr, "ERROR line #%i: knn d2[%i]=%g, oracle %g\n",
        line, i, td2[i], od2[i]);
      result = 0;
    }
    else if (0 < i && td2[i - 1] > td2[i]) {
      FPRINTF(stderr, "ERROR line #%i: knn result not sorted\n", line);
      result = 0;
    }
  }
  return result;
}


/* The budgeted search: exact without a budget (or with n), else sorted true distances, each no
 * nearer than the exact one of its rank, and at least min(k, budget) of them (unbounded radius). */
static int knn_check_budget(const double* pts, int* idx, int n, int ndims,
  const double* query, double max_dist2, int k, int line)
{
  static const int budgets[] = { 0, -1, 4, 16 };
  int tidx[KNN_MAXK], oidx[KNN_MAXK], i, b, result = 1;
  double td2[KNN_MAXK], od2[KNN_MAXK];
  const int no = knn_oracle(pts, n, ndims, ndims, query, max_dist2, oidx, od2, k);
  for (b = 0; b < 4 && 0 != result; ++b) {
    const int budget = (0 <= budgets[b] ? budgets[b] : n);
    const int nt = libxs_kdtree_knearest_budget(pts, idx, NULL, n, ndims, ndims,
      query, max_dist2, tidx, td2, k, budget);
    const int exact = (0 >= budget || n <= budget);
    if ((0 != exact && nt != no) || nt > no || (1e300 <= max_dist2 && nt < LIBXS_MIN(no, budget))) {
      FPRINTF(stderr, "ERROR line #%i: budget %i found %i, oracle %i\n", line, budget, nt, no);
      result = 0;
    }
    for (i = 0; i < nt && 0 != result; ++i) {
      double d2 = 0;
      int dd;
      for (dd = 0; dd < ndims; ++dd) {
        const double d = pts[(size_t)tidx[i] * ndims + dd] - query[dd];
        d2 += d * d;
      }
      if (d2 != td2[i] || td2[i] < od2[i] || (0 != exact && td2[i] != od2[i])
        || (0 < i && td2[i - 1] > td2[i]))
      {
        FPRINTF(stderr, "ERROR line #%i: budget %i d2[%i]=%g, oracle %g\n", line, budget, i, td2[i], od2[i]);
        result = 0;
      }
    }
  }
  return result;
}


static void knn_fill(double* pts, int n, int ndims, unsigned int seed, int mod)
{
  int i, dd;
  for (i = 0; i < n; ++i) {
    for (dd = 0; dd < ndims; ++dd) {
      seed = seed * 1103515245u + 12347u;
      pts[(size_t)i * ndims + dd] = (double)((seed >> 8) % (unsigned int)mod);
    }
  }
}


/* Splits at the middle position along axis 0 (no reordering) and records the leaves passed in:
 * seen[0] calls, seen[1] the last count, seen[2] decreases. */
static int split_count(int* dim, int* pos, double* threshold, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx)
{
  int *const seen = (int*)ctx;
  if (nleaves < seen[1]) ++seen[2];
  seen[1] = nleaves;
  ++seen[0];
  *dim = 0;
  *pos = count / 2;
  LIBXS_UNUSED(threshold);
  LIBXS_UNUSED(pts);
  LIBXS_UNUSED(idx);
  LIBXS_UNUSED(depth);
  return 0;
}


/* Median split along the axes in turn, in coordinates twice those of pts: the threshold is the
 * callback's own (as for a split by other coordinates than the tree's). */
static int split_scaled(int* dim, int* pos, double* threshold, const double* pts, int* idx,
  int count, int depth, int nleaves, void* ctx)
{
  const int d = depth % 3;
  int i, j;
  for (i = 1; i < count; ++i) { /* insertion sort along d */
    const int t = idx[i];
    for (j = i; 0 < j && pts[3 * t + d] < pts[3 * idx[j - 1] + d]; --j) idx[j] = idx[j - 1];
    idx[j] = t;
  }
  *dim = d;
  *pos = count / 2;
  *threshold = pts[3 * idx[*pos - 1] + d] + pts[3 * idx[*pos] + d]; /* 2 * midway */
  LIBXS_UNUSED(nleaves);
  LIBXS_UNUSED(ctx);
  return 0;
}


int main(void)
{
  /* kd-tree k-nearest must equal a linear scan, over a spread of shapes */
  { static double pts[KNN_MAXN * KNN_MAXD];
    static int idx[KNN_MAXN];
    const int dims[] = { 1, 2, 3, 8 };
    const int counts[] = { 1, 2, 7, 64, 512 };
    int di, ci, ki, t;
    for (di = 0; di < 4; ++di) {
      const int ndims = dims[di];
      for (ci = 0; ci < 5; ++ci) {
        const int n = counts[ci];
        /* mod 5 forces many duplicate coordinates and hence distance ties */
        for (t = 0; t < 2; ++t) {
          const int mod = (0 == t) ? 1000 : 5;
          int i;
          knn_fill(pts, n, ndims, (unsigned int)(1 + di * 97 + ci * 13 + t), mod);
          for (i = 0; i < n; ++i) idx[i] = i;
          libxs_kdtree_build(pts, idx, n, ndims, ndims, NULL);
          for (ki = 0; ki < 4; ++ki) {
            const int k = (0 == ki) ? 1 : ((1 == ki) ? 3 : ((2 == ki) ? 32 : 2));
            double query[KNN_MAXD];
            int dd;
            /* a query AT a stored point exercises the exact-match case */
            for (dd = 0; dd < ndims; ++dd) query[dd] = pts[dd];
            if (!knn_check(pts, idx, n, ndims, query, 1e300, k, __LINE__)
              || !knn_check_budget(pts, idx, n, ndims, query, 1e300, k, __LINE__))
            {
              exit(EXIT_FAILURE);
            }
            for (dd = 0; dd < ndims; ++dd) query[dd] = 1.5 + 3.0 * dd;
            if (!knn_check(pts, idx, n, ndims, query, 1e300, k, __LINE__)
              || !knn_check_budget(pts, idx, n, ndims, query, 1e300, k, __LINE__))
            {
              exit(EXIT_FAILURE);
            }
            /* a radius bound must drop the far members, not reorder the near */
            if (!knn_check(pts, idx, n, ndims, query, 50.0, k, __LINE__)
              || !knn_check_budget(pts, idx, n, ndims, query, 50.0, k, __LINE__))
            {
              exit(EXIT_FAILURE);
            }
          }
        }
      }
    }
  }
  /* k larger than the point count reports what exists, not k */
  { double pts[3 * 2];
    int idx[3], out_idx[KNN_MAXK], got;
    double out_d2[KNN_MAXK], query[2];
    int i;
    pts[0] = 0; pts[1] = 0; pts[2] = 1; pts[3] = 0; pts[4] = 0; pts[5] = 2;
    for (i = 0; i < 3; ++i) idx[i] = i;
    libxs_kdtree_build(pts, idx, 3, 2, 2, NULL);
    query[0] = 0; query[1] = 0;
    got = libxs_kdtree_knearest(pts, idx, NULL, 3, 2, 2, query, 1e300,
      out_idx, out_d2, 16);
    if (3 != got || 0.0 != out_d2[0]) {
      FPRINTF(stderr, "ERROR line #%i: k>n gave %i\n", __LINE__, got);
      exit(EXIT_FAILURE);
    }
  }

  /* direct in-place sort of doubles */
  { double data[] = {5.0, -1.0, 3.0, 0.0, 2.0, -4.0, 1.0};
    libxs_sort(data, 7, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 7)) {
      FPRINTF(stderr, "ERROR line #%i: f64 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* out-of-place sort of doubles (ctx = source) */
  { const double src[] = {9.0, 1.0, 5.0, 3.0, 7.0};
    double dst[5];
    libxs_sort(dst, 5, sizeof(double), libxs_cmp_f64, (void*)(uintptr_t)src);
    if (!check_sorted_f64(dst, 5)) {
      FPRINTF(stderr, "ERROR line #%i: f64 out-of-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (9.0 != src[0] || 1.0 != src[1]) {
      FPRINTF(stderr, "ERROR line #%i: f64 source modified\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* f32 in-place */
  { float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    libxs_sort(data, 8, sizeof(float), libxs_cmp_f32, NULL);
    if (!check_sorted_f32(data, 8)) {
      FPRINTF(stderr, "ERROR line #%i: f32 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* i32 in-place */
  { int data[] = {42, -7, 0, 100, -100, 3, 3};
    libxs_sort(data, 7, sizeof(int), libxs_cmp_i32, NULL);
    if (!check_sorted_i32(data, 7)) {
      FPRINTF(stderr, "ERROR line #%i: i32 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* u32 in-place */
  { unsigned int data[] = {300, 100, 200, 0, 400};
    libxs_sort(data, 5, sizeof(unsigned int), libxs_cmp_u32, NULL);
    if (data[0] != 0 || data[1] != 100 || data[2] != 200
      || data[3] != 300 || data[4] != 400)
    {
      FPRINTF(stderr, "ERROR line #%i: u32 in-place\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* custom comparator: sort by absolute value */
  { double data[] = {-5.0, 1.0, -3.0, 2.0, -4.0};
    libxs_sort(data, 5, sizeof(double), cmp_by_abs, NULL);
    if (!(1.0 == data[0] && 2.0 == data[1] && -3.0 == data[2]
      && -4.0 == data[3] && -5.0 == data[4]))
    {
      FPRINTF(stderr, "ERROR line #%i: custom comparator\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* indirect sort (argsort): sort index array by key values */
  { const double keys[] = {3.0, 1.0, 4.0, 1.5, 2.0};
    int perm[] = {0, 1, 2, 3, 4};
    libxs_sort(perm, 5, sizeof(int), cmp_indirect_f64, (void*)(uintptr_t)keys);
    if (perm[0] != 1 || perm[1] != 3 || perm[2] != 4
      || perm[3] != 0 || perm[4] != 2)
    {
      FPRINTF(stderr, "ERROR line #%i: indirect sort\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* single element */
  { double data[] = {42.0};
    libxs_sort(data, 1, sizeof(double), libxs_cmp_f64, NULL);
    if (42.0 != data[0]) {
      FPRINTF(stderr, "ERROR line #%i: single element\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* already sorted */
  { double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    libxs_sort(data, 5, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 5)) {
      FPRINTF(stderr, "ERROR line #%i: already sorted\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* reverse sorted */
  { double data[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    libxs_sort(data, 5, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 5)) {
      FPRINTF(stderr, "ERROR line #%i: reverse sorted\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* duplicates */
  { double data[] = {2.0, 2.0, 1.0, 1.0, 3.0, 3.0};
    libxs_sort(data, 6, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 6)) {
      FPRINTF(stderr, "ERROR line #%i: duplicates\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* negative values */
  { double data[] = {-1.0, -5.0, -2.0, -4.0, -3.0};
    libxs_sort(data, 5, sizeof(double), libxs_cmp_f64, NULL);
    if (!check_sorted_f64(data, 5)) {
      FPRINTF(stderr, "ERROR line #%i: negatives\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* hilbert (ndims=2): verify locality on 4x4 grid */
  { const unsigned int n = 4;
    uint64_t codes[16];
    unsigned int order[16];
    unsigned int coords[2], x, y, idx;
    for (y = 0; y < n; ++y) {
      for (x = 0; x < n; ++x) {
        coords[0] = x; coords[1] = y;
        codes[y * n + x] = libxs_hilbert(coords, 2);
      }
    }
    /* sort by code to get curve order */
    for (idx = 0; idx < n * n; ++idx) order[idx] = idx;
    { unsigned int i, j;
      for (i = 0; i < n * n - 1; ++i) {
        for (j = i + 1; j < n * n; ++j) {
          if (codes[order[j]] < codes[order[i]]) {
            unsigned int t = order[i]; order[i] = order[j]; order[j] = t;
          }
        }
      }
    }
    /* check all distinct */
    { unsigned int i;
      for (i = 1; i < n * n; ++i) {
        if (codes[order[i]] == codes[order[i - 1]]) {
          FPRINTF(stderr, "ERROR line #%i: hilbert 2D not bijective\n", __LINE__);
          exit(EXIT_FAILURE);
        }
      }
    }
    /* check locality: consecutive curve positions are Manhattan-adjacent */
    { unsigned int i;
      for (i = 1; i < n * n; ++i) {
        const unsigned int px = order[i - 1] % n, py = order[i - 1] / n;
        const unsigned int cx = order[i] % n, cy = order[i] / n;
        const unsigned int dx = (cx > px) ? cx - px : px - cx;
        const unsigned int dy = (cy > py) ? cy - py : py - cy;
        if (dx + dy != 1) {
          FPRINTF(stderr, "ERROR line #%i: hilbert 2D locality "
            "i=%u dx=%u dy=%u\n", __LINE__, i, dx, dy);
          exit(EXIT_FAILURE);
        }
      }
    }
  }
  /* hilbert (ndims=3): verify bijectivity on 4x4x4 grid */
  { const unsigned int n = 4;
    const unsigned int total = n * n * n;
    uint64_t codes[64];
    unsigned int x, y, z, i, j, collisions = 0;
    for (z = 0; z < n; ++z) {
      for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
          unsigned int coords[3];
          coords[0] = x; coords[1] = y; coords[2] = z;
          codes[z * n * n + y * n + x] = libxs_hilbert(coords, 3);
        }
      }
    }
    for (i = 0; i < total; ++i) {
      for (j = i + 1; j < total; ++j) {
        if (codes[i] == codes[j]) ++collisions;
      }
    }
    if (0 != collisions) {
      FPRINTF(stderr, "ERROR line #%i: hilbert 3D collisions=%u\n",
        __LINE__, collisions);
      exit(EXIT_FAILURE);
    }
  }
  /* morton/hilbert decode: verify small-grid round trips */
  { const unsigned int n = 4;
    unsigned int coords[3], decoded[3], x, y, z;
    for (z = 0; z < n; ++z) {
      for (y = 0; y < n; ++y) {
        for (x = 0; x < n; ++x) {
          coords[0] = x; coords[1] = y; coords[2] = z;
          libxs_morton_decode(libxs_morton(coords, 3), decoded, 3);
          if (decoded[0] != x || decoded[1] != y || decoded[2] != z) {
            FPRINTF(stderr, "ERROR line #%i: morton round trip\n", __LINE__);
            exit(EXIT_FAILURE);
          }
          libxs_hilbert_decode(libxs_hilbert(coords, 3), decoded, 3);
          if (decoded[0] != x || decoded[1] != y || decoded[2] != z) {
            FPRINTF(stderr, "ERROR line #%i: hilbert round trip\n", __LINE__);
            exit(EXIT_FAILURE);
          }
        }
      }
    }
  }
  /* stratify: verify 3D to 2D composition */
  { unsigned int src[3], dst[2], ref[2];
    src[0] = 1; src[1] = 2; src[2] = 3;
    if (EXIT_SUCCESS != libxs_stratify_morton(src, 3, dst, 2)) {
      FPRINTF(stderr, "ERROR line #%i: morton stratify failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    libxs_morton_decode(libxs_morton(src, 3), ref, 2);
    if (dst[0] != ref[0] || dst[1] != ref[1]) {
      FPRINTF(stderr, "ERROR line #%i: morton stratify mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (EXIT_SUCCESS != libxs_stratify_hilbert(src, 3, dst, 2)) {
      FPRINTF(stderr, "ERROR line #%i: hilbert stratify failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    libxs_hilbert_decode(libxs_hilbert(src, 3), ref, 2);
    if (dst[0] != ref[0] || dst[1] != ref[1]) {
      FPRINTF(stderr, "ERROR line #%i: hilbert stratify mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (EXIT_FAILURE != libxs_stratify_hilbert(src, 2, dst, 3)) {
      FPRINTF(stderr, "ERROR line #%i: invalid stratify accepted\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* stratify: verify finite-bit canonical rank-preserving layout */
  { unsigned int src[3], dst[2], ref[2];
    const int src_ndims = 3, dst_ndims = 2, src_bits = 2, dst_bits = 3;
    src[0] = 1; src[1] = 2; src[2] = 3;
    if (EXIT_SUCCESS != libxs_stratify_morton_bits(
      src, src_ndims, src_bits, dst, dst_ndims, dst_bits))
    {
      FPRINTF(stderr, "ERROR line #%i: finite morton stratify failed\n",
        __LINE__);
      exit(EXIT_FAILURE);
    }
    decode_morton_bits(encode_morton_bits(src, src_ndims, src_bits),
      ref, dst_ndims, dst_bits);
    if (dst[0] != ref[0] || dst[1] != ref[1]) {
      FPRINTF(stderr, "ERROR line #%i: finite morton mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (EXIT_SUCCESS != libxs_stratify_hilbert_bits(
      src, src_ndims, src_bits, dst, dst_ndims, dst_bits))
    {
      FPRINTF(stderr, "ERROR line #%i: finite hilbert stratify failed\n",
        __LINE__);
      exit(EXIT_FAILURE);
    }
    decode_hilbert_bits(encode_hilbert_bits(src, src_ndims, src_bits),
      ref, dst_ndims, dst_bits);
    if (dst[0] != ref[0] || dst[1] != ref[1]) {
      FPRINTF(stderr, "ERROR line #%i: finite hilbert mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (EXIT_FAILURE != libxs_stratify_hilbert_bits(
      src, src_ndims, src_bits, dst, dst_ndims, 2))
    {
      FPRINTF(stderr, "ERROR line #%i: invalid finite stratify accepted\n",
        __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* split rules of the tree: median equals the default, middle partitions a grid into boxes */
  { static double pts[KNN_MAXN * 3];
    static int idx[KNN_MAXN], ref[KNN_MAXN], leaf[KNN_MAXN], leaf2[KNN_MAXN];
    libxs_kdtree_config_t config;
    double lo_thr = 0;
    int i, l, nleaves, nleaves2, dd;
    knn_fill(pts, 300, 3, 7u, 1000);
    for (i = 0; i < 300; ++i) idx[i] = ref[i] = i;
    libxs_kdtree_build(pts, ref, 300, 3, 3, NULL);
    config.min_leaf = 0;
    config.split = libxs_kdtree_split_median;
    config.ctx = NULL;
    config.nodes = NULL;
    libxs_kdtree_build(pts, idx, 300, 3, 3, &config);
    if (0 != memcmp(idx, ref, sizeof(int) * 300)) {
      FPRINTF(stderr, "ERROR line #%i: median split differs from the default\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    for (i = 0; i < 300; ++i) idx[i] = ref[i] = i;
    config.min_leaf = 4;
    nleaves = libxs_kdtree_partition(pts, idx, 300, 3, 3, leaf, &config);
    config.split = NULL;
    nleaves2 = libxs_kdtree_partition(pts, ref, 300, 3, 3, leaf2, &config);
    if (nleaves != nleaves2 || 0 != memcmp(leaf, leaf2, sizeof(int) * 300)) {
      FPRINTF(stderr, "ERROR line #%i: median partition differs from the default\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    /* a 4x6x5 grid: every leaf is the full set of grid points in its bounding box */
    for (i = 0; i < 120; ++i) {
      pts[3 * i + 0] = (double)(i % 4);
      pts[3 * i + 1] = (double)((i / 4) % 6);
      pts[3 * i + 2] = (double)(i / 24);
      idx[i] = i;
    }
    config.min_leaf = 8;
    config.split = libxs_kdtree_split_middle;
    nleaves = libxs_kdtree_partition(pts, idx, 120, 3, 3, leaf, &config);
    for (l = 0; l < nleaves; ++l) {
      double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };
      int count = 0, inbox = 0;
      for (i = 0; i < 120; ++i) {
        if (l == leaf[i]) {
          for (dd = 0; dd < 3; ++dd) {
            lo[dd] = LIBXS_MIN(lo[dd], pts[3 * i + dd]);
            hi[dd] = LIBXS_MAX(hi[dd], pts[3 * i + dd]);
          }
          ++count;
        }
      }
      for (i = 0; i < 120; ++i) {
        int in = 1;
        for (dd = 0; dd < 3; ++dd) in &= (lo[dd] <= pts[3 * i + dd] && pts[3 * i + dd] <= hi[dd]);
        inbox += in;
      }
      if (0 == count || 8 < count || inbox != count) {
        FPRINTF(stderr, "ERROR line #%i: middle leaf %i has %i points, %i in its box\n",
          __LINE__, l, count, inbox);
        exit(EXIT_FAILURE);
      }
    }
    for (i = 0; i < 120; ++i) {
      if (0 > leaf[i] || nleaves <= leaf[i]) {
        FPRINTF(stderr, "ERROR line #%i: point %i not assigned\n", __LINE__, i);
        exit(EXIT_FAILURE);
      }
    }
    /* the tree in preorder: every point descends to its leaf */
    { static libxs_kdtree_node_t nodes[2 * KNN_MAXN];
      double scaled[3];
      int s, nleaf_nodes = 0;
      config.nodes = nodes;
      for (i = 0; i < 120; ++i) idx[i] = i;
      nleaves = libxs_kdtree_partition(pts, idx, 120, 3, 3, leaf, &config);
      for (l = 0; l < 2 * nleaves - 1; ++l) nleaf_nodes += (0 > nodes[l].dim ? 1 : 0);
      for (i = 0; i < 120 && nleaf_nodes == nleaves; ++i) {
        if (leaf[i] != libxs_kdtree_descend(nodes, pts + 3 * i)) nleaf_nodes = -1;
      }
      if (nleaf_nodes != nleaves) {
        FPRINTF(stderr, "ERROR line #%i: middle tree does not route to its leaves\n", __LINE__);
        exit(EXIT_FAILURE);
      }
      for (s = 0; s < 2; ++s) { /* distinct coordinates per axis: median, then own coordinates */
        for (i = 0; i < 300; ++i) {
          pts[3 * i + 0] = (double)((i * 7) % 300);
          pts[3 * i + 1] = (double)((i * 11) % 300);
          pts[3 * i + 2] = (double)((i * 13) % 300);
          idx[i] = i;
        }
        config.min_leaf = 4;
        config.split = (0 == s ? NULL : split_scaled);
        nleaves = libxs_kdtree_partition(pts, idx, 300, 3, 3, leaf, &config);
        for (i = 0; i < 300; ++i) {
          for (dd = 0; dd < 3; ++dd) scaled[dd] = (0 == s ? 1.0 : 2.0) * pts[3 * i + dd];
          if (leaf[i] != libxs_kdtree_descend(nodes, scaled)) {
            FPRINTF(stderr, "ERROR line #%i: point %i descends to %i, not leaf %i (%s)\n", __LINE__,
              i, libxs_kdtree_descend(nodes, scaled), leaf[i], 0 == s ? "median" : "own coordinates");
            exit(EXIT_FAILURE);
          }
        }
      }
      config.split = libxs_kdtree_split_middle;
      config.nodes = NULL;
    }
    /* identical points have no extent: one leaf */
    for (i = 0; i < 16; ++i) {
      pts[3 * i + 0] = pts[3 * i + 1] = pts[3 * i + 2] = 1.0;
      idx[i] = i;
    }
    config.min_leaf = 1;
    if (1 != libxs_kdtree_partition(pts, idx, 16, 3, 3, leaf, &config)) {
      FPRINTF(stderr, "ERROR line #%i: identical points split\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    /* build passes the leaves so far to the split function, as partition does */
    { int seen[3] = { 0, 0, 0 };
      knn_fill(pts, 64, 3, 11u, 1000);
      for (i = 0; i < 64; ++i) idx[i] = i;
      config.min_leaf = 4;
      config.split = split_count;
      config.ctx = seen;
      libxs_kdtree_build(pts, idx, 64, 3, 3, &config);
      if (2 > seen[0] || 0 >= seen[1] || 0 != seen[2]) {
        FPRINTF(stderr, "ERROR line #%i: build passed nleaves %i after %i calls (%i decreases)\n",
          __LINE__, seen[1], seen[0], seen[2]);
        exit(EXIT_FAILURE);
      }
      config.ctx = NULL;
    }
    /* called directly, the markers do not split */
    if (0 == libxs_kdtree_split_median(&i, &l, &lo_thr, pts, idx, 16, 0, 0, NULL)
      || 0 == libxs_kdtree_split_middle(&i, &l, &lo_thr, pts, idx, 16, 0, 0, NULL))
    {
      FPRINTF(stderr, "ERROR line #%i: a marker split\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* kdtree2d: basic nearest neighbor */
  { double pts[] = {0.0,0.0, 1.0,0.0, 0.0,1.0, 1.0,1.0};
    int idx[] = {0, 1, 2, 3};
    int hit;
    libxs_kdtree2d_build(pts, idx, 4);
    hit = libxs_kdtree2d_nearest(pts, idx, NULL, 4, 0.1, 0.1, 1.0);
    if (hit != 0) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d nearest=%d\n", __LINE__, hit);
      exit(EXIT_FAILURE);
    }
    hit = libxs_kdtree2d_nearest(pts, idx, NULL, 4, 0.9, 0.9, 1.0);
    if (hit != 3) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d nearest=%d\n", __LINE__, hit);
      exit(EXIT_FAILURE);
    }
  }
  /* kdtree2d: used-flag consumption */
  { double pts[] = {0.0,0.0, 0.1,0.1, 5.0,5.0};
    int idx[] = {0, 1, 2};
    unsigned char used[] = {0, 0, 0};
    int h1, h2;
    libxs_kdtree2d_build(pts, idx, 3);
    h1 = libxs_kdtree2d_nearest(pts, idx, used, 3, 0.0, 0.0, 1.0);
    if (h1 != 0) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d used h1=%d\n", __LINE__, h1);
      exit(EXIT_FAILURE);
    }
    used[h1] = 1;
    h2 = libxs_kdtree2d_nearest(pts, idx, used, 3, 0.0, 0.0, 1.0);
    if (h2 != 1) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d used h2=%d\n", __LINE__, h2);
      exit(EXIT_FAILURE);
    }
  }
  /* kdtree2d: no match within radius */
  { double pts[] = {10.0,10.0, 20.0,20.0};
    int idx[] = {0, 1};
    int hit;
    libxs_kdtree2d_build(pts, idx, 2);
    hit = libxs_kdtree2d_nearest(pts, idx, NULL, 2, 0.0, 0.0, 1.0);
    if (hit != -1) {
      FPRINTF(stderr, "ERROR line #%i: kdtree2d no-match=%d\n", __LINE__, hit);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
