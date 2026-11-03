/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_pack.h>
#include <libxs/libxs_mhd.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_timer.h>
#include <libxs/libxs_utils.h>
#include <float.h>
#include <math.h>
#if defined(__AVX512F__) /* k-d prototype (also without LIBXS intrinsics) */
# include <immintrin.h>
#endif

#define BLOCK 512
#define NGROUPS 5
#define MAXBITS 64
#define WIDTH_HEADER 6 /* bits per block that record its width (CP2K: maxval stream) */
#define NMODELS 4 /* group of 8: width byte + 8*w, zero mask, exceptions, both */
#define NAXES 3
#define TREE 64 /* values per dyadic tree (leaves of 4 to TREE values) */
#define NTREES 2 /* dyadic tree: one width per leaf, or with exceptions at leaves of 8 */
#define NKD 4 /* k-d tree: all axes, all axes with exceptions, axes 1+2 (whole rows), axis 2 (ranges) */
#define NORDERS 6 /* traversal orders of a block's axes (innermost first) */
#define RICE_K 2
#define RICE_ESC 24
#define RICE_MAXLANES 8


typedef struct stream_t {
  int64_t* q;
  size_t n, cap;
} stream_t;

typedef struct stats_t {
  stream_t plain[MAXBITS + 1], delta[MAXBITS + 1], twice[MAXBITS + 1];
  stream_t blocks, shapes; /* values of the blocks with a shape (in order), and their shapes */
  size_t nblocks, nvalues, nzeros, nskipped;
  double bits_fixed, bits_twice, bits_needed, eps;
  double bits_axis[1 + NAXES], bits_axis_best; /* per block: as is, delta along an axis, best */
  double bits_kd[NKD], nleaves_kd[NKD]; /* per block: k-d tree of sub-boxes */
  double bits_order[NORDERS][2], bits_order_best[2]; /* per block: groups of 8 in an axis order, w/o and with exceptions */
  int has_twice;
} stats_t;


#if defined(LIBXS_INTRINSICS_AVX2)
/* Decoder state of four lanes. */
typedef struct rice_avx2_t {
  __m256i bit, p1, p2;
} rice_avx2_t;
#endif

/* Output stream of the prototype. */
typedef struct rice_writer_t {
  uint64_t* out;
  uint64_t acc;
  int fill;
} rice_writer_t;

typedef struct header_t {
  double payload, byte, seven, base3, base2, nwidths, hist[MAXBITS + 1];
} header_t;

typedef size_t (*vpack_fn)(void*, const void*, libxs_data_t, size_t, int, int);
typedef size_t (*vunpack_fn)(void*, const void*, size_t, libxs_data_t, size_t);

#if defined(PACK_FORTRAN)
/* vpack.f: a group of 8 in the format libxs_vpack writes under LIBXS_PACK_SW */
size_t vpack_f(int64_t* packed, const int64_t* values, size_t n, int maxbits);
size_t vunpack_f(int64_t* values, const int64_t* packed, size_t n);
#endif


static const int groups[NGROUPS] = { 4, 8, 16, 32, 64 };
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static size_t record; /* values per record (env. RECORD), 0: a stream at once */
static int nrepeat = 1; /* timed passes per measurement, the fastest counts */
static size_t kd_minleaf; /* k-d prototype: values a leaf has at least (env. KDLEAF) */
static double kd_runs, kd_leaves; /* k-d prototype: contiguous runs and leaves decoded */
static int kd_exc = 1; /* k-d prototype: exceptions (env. KDEXC) */


static uint64_t rng_next(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}


static int stream_push(stream_t* s, int64_t v)
{
  int result = EXIT_SUCCESS;
  if (s->n == s->cap) {
    const size_t cap = (0 != s->cap ? 2 * s->cap : 1024);
    int64_t *const q = (int64_t*)realloc(s->q, cap * sizeof(int64_t));
    if (NULL != q) {
      s->q = q;
      s->cap = cap;
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) s->q[s->n++] = v;
  return result;
}


/* Width CP2K derives for a block (qs_tensors.F: compress_tensor) at the given step. */
static int cp2k_nbits(double maxabs, double eps)
{
  int e, r;
  frexp(0 < maxabs ? maxabs : DBL_MIN, &e);
  if (-15 > e) e = -15;
  frexp(round(ldexp(1.0, e) / eps), &r);
  return r + 1;
}


static int64_t cp2k_quantize(double x, double eps, double step)
{
  return (fabs(x) > eps) ? (int64_t)llround(x / step) : 0;
}


static int needed_bits(int64_t q)
{
  const uint64_t z = ((uint64_t)q << 1) ^ (0 - ((uint64_t)q >> 63));
  return (int)LIBXS_NBITS(z);
}


/* Zeroes fresh memory outside a timing. Not inlined: next to its malloc, a memset becomes calloc and
 * the page faults stay in the timed code. */
static LIBXS_INLINE_NEVER void first_touch(void* p, size_t size)
{
  memset(p, 0, size);
}


static uint64_t zigzag(int64_t v)
{
  return ((uint64_t)v << 1) ^ (0 - ((uint64_t)v >> 63));
}


/**
 * Bits of a group of 8 (zig-zag) under cost models, with a flag bit spare in the width byte:
 * [0] width byte and 8*w; [1] a zero mask (byte) and the nonzeros only, where it saves;
 * [2] exceptions: a regular width v, the values above it patched (bitmap byte, width byte of
 * the high parts, w-v bits each) at the best v; [3] zero mask and exceptions.
 * For [2], the exceptions at the best v and the width of their high parts.
 */
static void group_costs(const uint64_t z[8], double cost[NMODELS], int* nexc, int* hbits)
{
  int nb[8], w = 0, nz = 0, i, v;
  for (i = 0; i < 8; ++i) {
    nb[i] = (int)LIBXS_NBITS(z[i]);
    w = LIBXS_MAX(w, nb[i]);
    nz += (0 != nb[i] ? 1 : 0);
  }
  cost[0] = 8 + 8.0 * w;
  cost[1] = 8 + (0 < w ? LIBXS_MIN(8.0 * w, 8 + (double)nz * w) : 0);
  cost[2] = cost[0];
  cost[3] = cost[1];
  *nexc = *hbits = 0;
  for (v = 0; v < w; ++v) {
    double patch, regular;
    int e = 0;
    for (i = 0; i < 8; ++i) e += (v < nb[i] ? 1 : 0);
    patch = 16 + (double)e * (w - v);
    regular = LIBXS_MIN(8.0 * v, 8 + (double)nz * v);
    if (8 + 8.0 * v + patch < cost[2]) {
      cost[2] = 8 + 8.0 * v + patch;
      *nexc = e;
      *hbits = w - v;
    }
    cost[3] = LIBXS_MIN(cost[3], 8 + regular + patch);
  }
}


/**
 * Cost models over the streams of every width (groups of 8 as vpack forms them), and for the
 * exceptions model: groups, groups with exceptions, exceptions, and the sum of high-part widths.
 */
static void stream_costs(const stream_t* s, double bits[NMODELS], double exc[4])
{
  int nbits, k;
  for (k = 0; k < NMODELS; ++k) bits[k] = 0;
  for (k = 0; k < 4; ++k) exc[k] = 0;
  for (nbits = 1; nbits <= MAXBITS; ++nbits) {
    size_t i, j;
    for (i = 0; i < s[nbits].n; i += 8) {
      uint64_t z[8];
      double cost[NMODELS];
      int nexc, hbits;
      for (j = 0; j < 8; ++j) z[j] = (i + j < s[nbits].n ? zigzag(s[nbits].q[i + j]) : 0);
      group_costs(z, cost, &nexc, &hbits);
      for (k = 0; k < NMODELS; ++k) bits[k] += cost[k];
      exc[0] += 1;
      if (0 < nexc) {
        exc[1] += 1;
        exc[2] += nexc;
        exc[3] += hbits;
      }
    }
  }
}


/**
 * Bits of the best dyadic tree over z[0..n) (n = 4 * 2^k): one width for all values (a width byte
 * and n*w), for n = 8 optionally with exceptions, or both halves; 1 bit per node above size 4.
 */
static double tree_cost(const uint64_t* z, int n, int exc)
{
  uint64_t any = 0;
  double leaf;
  int i;
  for (i = 0; i < n; ++i) any |= z[i];
  leaf = 8 + (double)n * (double)LIBXS_NBITS(any);
  if (8 == n && 0 != exc) {
    double cost[NMODELS];
    int nexc, hbits;
    group_costs(z, cost, &nexc, &hbits);
    leaf = LIBXS_MIN(leaf, cost[2]);
  }
  if (4 < n) {
    const double split = tree_cost(z, n / 2, exc) + tree_cost(z + n / 2, n / 2, exc);
    leaf = 1 + LIBXS_MIN(leaf, split);
  }
  return leaf;
}


/** Dyadic trees of TREE values over the streams of every width. */
static void tree_costs(const stream_t* s, double bits[NTREES])
{
  int nbits, k;
  for (k = 0; k < NTREES; ++k) bits[k] = 0;
  for (nbits = 1; nbits <= MAXBITS; ++nbits) {
    size_t i, j;
    for (i = 0; i < s[nbits].n; i += TREE) {
      uint64_t z[TREE];
      for (j = 0; j < TREE; ++j) z[j] = (i + j < s[nbits].n ? zigzag(s[nbits].q[i + j]) : 0);
      for (k = 0; k < NTREES; ++k) bits[k] += tree_cost(z, TREE, k);
    }
  }
}


/**
 * Bits of a box of count values (z[idx[i]]) with one width (a width byte and count*w), or with
 * exceptions: a regular width v and the values above it at w bits (positions as a bitmap or as
 * indexes, whichever is smaller, and a width byte), at the best v.
 */
static double leaf_cost(const uint64_t* z, const int* idx, int count, int exc)
{
  int hist[MAXBITS + 1], w = 0, above = 0, lg = 0, i, v;
  double result;
  for (i = 0; i <= MAXBITS; ++i) hist[i] = 0;
  for (i = 0; i < count; ++i) {
    const int b = (int)LIBXS_NBITS(z[idx[i]]);
    ++hist[b];
    w = LIBXS_MAX(w, b);
  }
  result = 8 + (double)count * w;
  if (0 != exc) {
    while ((1 << lg) < count) ++lg;
    for (v = w - 1; 0 <= v; --v) {
      above += hist[v + 1]; /* values wider than v */
      result = LIBXS_MIN(result, 16 + (double)count * v + (double)LIBXS_MIN(count, above * lg)
        + (double)above * (w - v));
    }
  }
  return result;
}


typedef struct kd_ctx_t {
  const uint64_t* z;
  int *left, *right; /* scratch */
  int axes, dbits, exc; /* axes that may split (bit mask), bits to name one, exceptions */
} kd_ctx_t;

/* Splits a box at the middle of the allowed axis that lowers the bits most (both halves as leaves,
 * the axis and their flags counted), else a leaf. */
static int kd_split(int* dim, int* pos, double* threshold, const double* pts, int* idx, int count,
  int depth, int nleaves, void* ctx)
{
  const kd_ctx_t *const kd = (const kd_ctx_t*)ctx;
  double best = leaf_cost(kd->z, idx, count, kd->exc);
  int result = 1, i, d, bestd = -1, bestl = 0;
  for (d = 0; d < NAXES; ++d) {
    if (0 != (kd->axes & (1 << d))) {
      double lo = pts[3 * idx[0] + d], hi = lo, mid, cost;
      int nl = 0, nr = 0;
      for (i = 1; i < count; ++i) {
        lo = LIBXS_MIN(lo, pts[3 * idx[i] + d]);
        hi = LIBXS_MAX(hi, pts[3 * idx[i] + d]);
      }
      if (lo < hi) {
        mid = lo + floor((hi - lo + 1) / 2);
        for (i = 0; i < count; ++i) {
          if (pts[3 * idx[i] + d] < mid) kd->left[nl++] = idx[i];
          else kd->right[nr++] = idx[i];
        }
        cost = kd->dbits + 2 + leaf_cost(kd->z, kd->left, nl, kd->exc) + leaf_cost(kd->z, kd->right, nr, kd->exc);
        if (cost < best) {
          best = cost;
          bestd = d;
          bestl = nl;
        }
      }
    }
  }
  if (0 <= bestd) { /* the left box first */
    double lo = pts[3 * idx[0] + bestd], hi = lo, mid;
    int nl = 0, nr = 0;
    for (i = 1; i < count; ++i) {
      lo = LIBXS_MIN(lo, pts[3 * idx[i] + bestd]);
      hi = LIBXS_MAX(hi, pts[3 * idx[i] + bestd]);
    }
    mid = lo + floor((hi - lo + 1) / 2);
    for (i = 0; i < count; ++i) {
      if (pts[3 * idx[i] + bestd] < mid) kd->left[nl++] = idx[i];
      else kd->right[nr++] = idx[i];
    }
    memcpy(idx, kd->left, nl * sizeof(int));
    memcpy(idx + nl, kd->right, nr * sizeof(int));
    *dim = bestd;
    *pos = bestl;
    result = 0;
  }
  LIBXS_UNUSED(threshold); /* the tree places it in pts */
  LIBXS_UNUSED(depth);
  LIBXS_UNUSED(nleaves);
  return result;
}


/**
 * A block as sub-boxes (k-d tree, libxs_kdtree_partition) of one width each or with exceptions:
 * a flag per node and the axis per split; variants as NKD lists them.
 */
static int kd_costs(stats_t* st, const int64_t* q, size_t n, const int32_t* shape)
{
  static const int axes[NKD] = { 7, 7, 6, 4 }, dbits[NKD] = { 2, 2, 1, 0 }, exc[NKD] = { 0, 1, 1, 1 };
  int result = EXIT_SUCCESS;
  if (NULL != shape && 0 < n) {
    double *const pts = (double*)malloc(3 * n * sizeof(double));
    uint64_t *const z = (uint64_t*)malloc(n * sizeof(uint64_t));
    int *const idx = (int*)malloc(n * sizeof(int)), *const leaf = (int*)malloc(n * sizeof(int));
    int *const left = (int*)malloc(n * sizeof(int)), *const right = (int*)malloc(n * sizeof(int));
    int *const start = (int*)malloc((n + 1) * sizeof(int)), *const list = (int*)malloc(n * sizeof(int));
    if (NULL != pts && NULL != z && NULL != idx && NULL != leaf && NULL != left && NULL != right
      && NULL != start && NULL != list)
    {
      const size_t s0 = (size_t)shape[0], s1 = (size_t)shape[1];
      libxs_kdtree_config_t config;
      kd_ctx_t ctx;
      size_t i;
      int k;
      for (i = 0; i < n; ++i) {
        pts[3 * i + 0] = (double)(i % s0);
        pts[3 * i + 1] = (double)((i / s0) % s1);
        pts[3 * i + 2] = (double)(i / (s0 * s1));
        z[i] = zigzag(q[i]);
      }
      ctx.z = z;
      ctx.left = left;
      ctx.right = right;
      config.min_leaf = 1;
      config.split = kd_split;
      config.ctx = &ctx;
      config.nodes = NULL;
      for (k = 0; k < NKD; ++k) {
        int nleaves, l;
        for (i = 0; i < n; ++i) idx[i] = (int)i;
        ctx.axes = axes[k];
        ctx.dbits = dbits[k];
        ctx.exc = exc[k];
        nleaves = libxs_kdtree_partition(pts, idx, (int)n, NAXES, NAXES, leaf, &config);
        for (l = 0; l <= nleaves; ++l) start[l] = 0; /* values per leaf, then lists */
        for (i = 0; i < n; ++i) ++start[leaf[i] + 1];
        for (l = 0; l < nleaves; ++l) start[l + 1] += start[l];
        for (i = 0; i < n; ++i) list[start[leaf[i]]++] = (int)i;
        for (l = nleaves; 0 < l; --l) start[l] = start[l - 1];
        start[0] = 0;
        for (l = 0; l < nleaves; ++l) {
          st->bits_kd[k] += leaf_cost(z, list + start[l], start[l + 1] - start[l], exc[k]);
        }
        st->bits_kd[k] += (double)(2 * nleaves - 1) + (double)dbits[k] * (nleaves - 1);
        st->nleaves_kd[k] += nleaves;
      }
    }
    else result = EXIT_FAILURE;
    free(pts);
    free(z);
    free(idx);
    free(leaf);
    free(left);
    free(right);
    free(start);
    free(list);
  }
  return result;
}


/** Groups of 8 of a block traversed in each order of its axes (innermost first), and the best order. */
static int order_costs(stats_t* st, const int64_t* q, size_t n, const int32_t* shape)
{
  static const int perm[NORDERS][3] = { { 0, 1, 2 }, { 0, 2, 1 }, { 1, 0, 2 }, { 1, 2, 0 }, { 2, 0, 1 }, { 2, 1, 0 } };
  int result = EXIT_SUCCESS;
  if (NULL != shape && 0 < n) {
    uint64_t *const z = (uint64_t*)malloc((n + 8) * sizeof(uint64_t));
    if (NULL != z) {
      size_t ext[3], stride[3];
      double best[2] = { 0, 0 };
      int o, x;
      for (x = 0; x < 3; ++x) ext[x] = (size_t)shape[x];
      stride[0] = 1;
      stride[1] = ext[0];
      stride[2] = ext[0] * ext[1];
      for (o = 0; o < NORDERS; ++o) {
        const int a = perm[o][0], b = perm[o][1], c = perm[o][2];
        double bits[2] = { 0, 0 };
        size_t ia, ib, ic, k = 0, i;
        for (ic = 0; ic < ext[c]; ++ic) {
          for (ib = 0; ib < ext[b]; ++ib) {
            for (ia = 0; ia < ext[a]; ++ia) z[k++] = zigzag(q[ia * stride[a] + ib * stride[b] + ic * stride[c]]);
          }
        }
        for (; 0 != (k & 7); ++k) z[k] = 0;
        for (i = 0; i < k; i += 8) {
          double cost[NMODELS];
          int nexc, hbits;
          group_costs(z + i, cost, &nexc, &hbits);
          bits[0] += cost[0];
          bits[1] += cost[2];
        }
        for (x = 0; x < 2; ++x) {
          st->bits_order[o][x] += bits[x];
          best[x] = (0 < o ? LIBXS_MIN(best[x], bits[x]) : bits[x]);
        }
      }
      for (x = 0; x < 2; ++x) st->bits_order_best[x] += best[x] + 3; /* the order per block */
    }
    else result = EXIT_FAILURE;
    free(z);
  }
  return result;
}


/** Groups of 8 of a block in memory order: as is, as deltas along each axis, and the best. */
static void axis_costs(stats_t* st, const int64_t* q, size_t n, const int32_t* shape)
{
  const size_t ext[NAXES] = { NULL != shape ? (size_t)shape[0] : n,
    NULL != shape ? (size_t)shape[1] : 1, NULL != shape ? (size_t)shape[2] : 1 };
  size_t stride[NAXES], i, j;
  double bits[1 + NAXES], best = 0;
  int a;
  stride[0] = 1;
  stride[1] = ext[0];
  stride[2] = ext[0] * ext[1];
  for (a = 0; a <= NAXES; ++a) {
    bits[a] = 0;
    for (i = 0; i < n; i += 8) {
      uint64_t z[8];
      double cost[NMODELS];
      int nexc, hbits;
      for (j = 0; j < 8; ++j) {
        const size_t k = i + j;
        int64_t d = 0;
        if (k < n) {
          d = q[k];
          if (0 < a && 0 < (k / stride[a - 1]) % ext[a - 1]) d -= q[k - stride[a - 1]];
        }
        z[j] = zigzag(d);
      }
      group_costs(z, cost, &nexc, &hbits);
      bits[a] += cost[0];
    }
    st->bits_axis[a] += bits[a];
    best = (0 < a ? LIBXS_MIN(best, bits[a]) : bits[a]);
  }
  st->bits_axis_best += best + 2; /* the choice per block */
}


/** Keeps a block in order with its shape (k-d prototype). */
static int keep_block(stats_t* st, const int64_t* q, size_t n, const int32_t* shape)
{
  int result = EXIT_SUCCESS, k;
  size_t i;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) result = stream_push(&st->blocks, q[i]);
  for (k = 0; k < 3 && EXIT_SUCCESS == result; ++k) result = stream_push(&st->shapes, shape[k]);
  return result;
}


/** Quantizes a block as CP2K does and appends it to the streams of its width. */
static int add_block(stats_t* st, const double* x, size_t n, double eps, const int32_t* shape)
{
  int64_t *const qb = (int64_t*)malloc(LIBXS_MAX(n, 1) * sizeof(int64_t));
  double maxabs = 0;
  int result = (NULL != qb ? EXIT_SUCCESS : EXIT_FAILURE), nbits, nbits2;
  int64_t prev = 0;
  size_t i;
  for (i = 0; i < n; ++i) maxabs = LIBXS_MAX(maxabs, fabs(x[i]));
  nbits = cp2k_nbits(maxabs, eps);
  nbits2 = cp2k_nbits(maxabs, 2 * eps);
  if (MAXBITS <= nbits) ++st->nskipped; /* CP2K aborts on this */
  else {
    ++st->nblocks;
    st->nvalues += n;
    st->bits_fixed += (double)n * nbits + WIDTH_HEADER;
    st->bits_twice += (double)n * nbits2 + WIDTH_HEADER;
    for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
      const int64_t q = cp2k_quantize(x[i], eps, eps);
      if (0 == q) ++st->nzeros;
      st->bits_needed += needed_bits(q);
      result = stream_push(&st->plain[nbits], q);
      if (EXIT_SUCCESS == result) result = stream_push(&st->delta[nbits + 1], q - prev);
      if (EXIT_SUCCESS == result) result = stream_push(&st->twice[nbits2], cp2k_quantize(x[i], eps, 2 * eps));
      qb[i] = q;
      prev = q;
    }
    if (EXIT_SUCCESS == result) axis_costs(st, qb, n, shape);
    if (EXIT_SUCCESS == result) result = kd_costs(st, qb, n, shape);
    if (EXIT_SUCCESS == result && NULL != shape) result = keep_block(st, qb, n, shape);
    if (EXIT_SUCCESS == result) result = order_costs(st, qb, n, shape);
  }
  free(qb);
  return result;
}


/** Integer values as they are: the fixed width is what the largest magnitude needs. */
static int add_block_int(stats_t* st, const int64_t* q, size_t n, const int32_t* shape)
{
  int result = EXIT_SUCCESS, nbits = 1;
  int64_t prev = 0;
  size_t i;
  for (i = 0; i < n; ++i) nbits = LIBXS_MAX(nbits, needed_bits(q[i]));
  ++st->nblocks;
  st->nvalues += n;
  st->bits_fixed += (double)n * nbits + WIDTH_HEADER;
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    if (0 == q[i]) ++st->nzeros;
    st->bits_needed += needed_bits(q[i]);
    result = stream_push(&st->plain[nbits], q[i]);
    if (EXIT_SUCCESS == result) result = stream_push(&st->delta[nbits + 1], q[i] - prev);
    prev = q[i];
  }
  axis_costs(st, q, n, shape);
  if (EXIT_SUCCESS == result) result = kd_costs(st, q, n, shape);
  if (EXIT_SUCCESS == result && NULL != shape) result = keep_block(st, q, n, shape);
  if (EXIT_SUCCESS == result) result = order_costs(st, q, n, shape);
  return result;
}


/** Adds an MHD file: CP2K's block table and eps from the extension, otherwise blocks of BLOCK. */
static int add_file(stats_t* st, const char* filename, double eps)
{
  char datafile[4096];
  size_t size[8], extsize = 0, n = 1, i;
  libxs_mhd_info_t info;
  char* ext = NULL;
  void* data = NULL;
  int result;
  info.ndims = sizeof(size) / sizeof(*size);
  result = libxs_mhd_read_header(filename, sizeof(datafile), datafile, &info, size, NULL, &extsize);
  if (EXIT_SUCCESS == result && 0 != extsize) {
    ext = (char*)malloc(extsize);
    info.ndims = sizeof(size) / sizeof(*size);
    result = (NULL != ext ? libxs_mhd_read_header(filename, sizeof(datafile), datafile,
      &info, size, ext, &extsize) : EXIT_FAILURE);
  }
  for (i = 0; i < info.ndims; ++i) n *= size[i];
  n *= info.ncomponents;
  if (EXIT_SUCCESS == result) {
    data = malloc(n * LIBXS_TYPESIZE(info.type));
    result = (NULL != data ? libxs_mhd_read(datafile, NULL, size, NULL, &info, data,
      NULL, NULL, NULL) : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) {
    const int is_float = (LIBXS_DATATYPE_F64 == info.type || LIBXS_DATATYPE_F32 == info.type);
    int32_t nblk = 0;
    double* x = NULL;
    int64_t* q = NULL;
    if (12 <= extsize) { /* CP2K: REAL(8) eps, INT32 number of blocks, INT32 shape(3) per block */
      memcpy(&nblk, ext + 8, sizeof(nblk));
      if (0 < nblk && (size_t)extsize == 12 + 12 * (size_t)nblk) memcpy(&eps, ext, sizeof(eps));
      else nblk = 0;
    }
    if (0 != is_float) {
      x = (double*)malloc(n * sizeof(double));
      if (NULL != x) {
        for (i = 0; i < n; ++i) {
          x[i] = (LIBXS_DATATYPE_F64 == info.type ? ((const double*)data)[i] : ((const float*)data)[i]);
        }
      }
      if (0 >= eps) {
        fprintf(stderr, "ERROR: %s holds floating-point data, an eps is needed\n", filename);
        result = EXIT_FAILURE;
      }
      st->has_twice = 1;
      st->eps = eps;
    }
    else {
      q = (int64_t*)malloc(n * sizeof(int64_t));
      if (NULL != q) {
        for (i = 0; i < n; ++i) {
          switch (info.type) {
            case LIBXS_DATATYPE_I8: q[i] = ((const int8_t*)data)[i]; break;
            case LIBXS_DATATYPE_U8: q[i] = ((const uint8_t*)data)[i]; break;
            case LIBXS_DATATYPE_I16: q[i] = ((const int16_t*)data)[i]; break;
            case LIBXS_DATATYPE_U16: q[i] = ((const uint16_t*)data)[i]; break;
            case LIBXS_DATATYPE_I32: q[i] = ((const int32_t*)data)[i]; break;
            case LIBXS_DATATYPE_U32: q[i] = ((const uint32_t*)data)[i]; break;
            default: q[i] = ((const int64_t*)data)[i];
          }
        }
      }
    }
    if (EXIT_SUCCESS == result && (NULL != x || NULL != q)) {
      size_t offset = 0, iblk = 0;
      while (offset < n && EXIT_SUCCESS == result) {
        size_t m = LIBXS_MIN((size_t)BLOCK, n - offset);
        int32_t shape[3];
        const int32_t* bshape = NULL;
        if (0 < nblk && iblk < (size_t)nblk) {
          memcpy(shape, ext + 12 + 12 * iblk, sizeof(shape));
          m = LIBXS_MIN((size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2], n - offset);
          if ((size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2] == m) bshape = shape;
        }
        result = (NULL != x ? add_block(st, x + offset, m, eps, bshape)
                            : add_block_int(st, q + offset, m, bshape));
        offset += m;
        ++iblk;
      }
    }
    else if (EXIT_SUCCESS == result) result = EXIT_FAILURE;
    free(x);
    free(q);
  }
  else fprintf(stderr, "ERROR: cannot read %s as MHD file\n", filename);
  free(ext);
  free(data);
  return result;
}


/* Header costs of the widths: bytes, 7 bits, or a base per 8 groups with 3- or 2-bit offsets. */
static void header_costs(header_t* h, const unsigned char* widths, size_t ngroups)
{
  size_t i, j;
  for (i = 0; i < ngroups; i += 8) {
    const size_t m = LIBXS_MIN((size_t)8, ngroups - i);
    int wmin = MAXBITS, wmax = 0;
    for (j = 0; j < m; ++j) {
      const int w = widths[i + j] & 0x7F; /* the regular width (group 8: 0x80 marks exceptions) */
      wmin = LIBXS_MIN(wmin, w);
      wmax = LIBXS_MAX(wmax, w);
      h->hist[w] += 1;
    }
    h->base3 += 1.0 + (7 >= wmax - wmin ? 7.0 + 3.0 * m : 7.0 * m); /* flag selects the scheme */
    h->base2 += 1.0 + (3 >= wmax - wmin ? 7.0 + 2.0 * m : 7.0 * m);
  }
  h->byte += 8.0 * (1 + ngroups);
  h->seven += 7.0 * ngroups;
  h->nwidths += (double)ngroups;
}


/**
 * Packs every stream at its fixed width (CP2K: offset binary, a group of 64 values per nbits words),
 * per record if given: each record decodes into a buffer that is reused (cache-resident).
 */
static void fixed_seconds(const stream_t* s, double* seconds, double* useconds, int* ok)
{
  int nbits;
  *seconds = *useconds = 0;
  for (nbits = 1; nbits <= MAXBITS; ++nbits) {
    if (0 != s[nbits].n) {
      const size_t n = s[nbits].n, r = (0 != record ? LIBXS_MIN(record, n) : n), nrec = (n + r - 1) / r;
      const size_t rwords = (r * (size_t)nbits + 63) / 64, nwords = nrec * rwords;
      const uint64_t offset = (64 > nbits ? ((uint64_t)1) << (nbits - 1) : 0);
      uint64_t *const packed = (uint64_t*)malloc(LIBXS_MAX(nwords, 1) * sizeof(uint64_t));
      uint64_t *const back = (uint64_t*)malloc(n * sizeof(uint64_t));
      uint64_t *const tmp = (uint64_t*)malloc(r * sizeof(uint64_t));
      if (NULL != packed && NULL != back && NULL != tmp) {
        double best = DBL_MAX, ubest = DBL_MAX;
        size_t i, k;
        int rep;
        first_touch(packed, LIBXS_MAX(nwords, 1) * sizeof(uint64_t));
        first_touch(tmp, r * sizeof(uint64_t));
        for (i = 0; i < n; ++i) back[i] = (uint64_t)s[nbits].q[i] + offset;
        for (rep = 0; rep < nrepeat; ++rep) {
          const libxs_timer_tick_t t0 = libxs_timer_tick();
          double d;
          for (k = 0; k < nrec; ++k) {
            libxs_pack(packed + k * rwords, back + k * r, LIBXS_DATATYPE_U64, LIBXS_MIN(r, n - k * r), nbits);
          }
          d = libxs_timer_duration(t0, libxs_timer_tick());
          best = LIBXS_MIN(best, d);
        }
        *seconds += best;
        for (rep = 0; rep < nrepeat; ++rep) {
          double d = 0;
          for (k = 0; k < nrec; ++k) {
            const size_t m = LIBXS_MIN(r, n - k * r);
            uint64_t *const dst = (0 != record ? tmp : back);
            const libxs_timer_tick_t t1 = libxs_timer_tick();
            libxs_unpack(dst, packed + k * rwords, LIBXS_DATATYPE_U64, m, nbits);
            d += libxs_timer_duration(t1, libxs_timer_tick());
            for (i = 0; i < m; ++i) {
              if (dst[i] != (uint64_t)s[nbits].q[k * r + i] + offset) *ok = 0;
            }
          }
          ubest = LIBXS_MIN(ubest, d);
        }
        *useconds += ubest;
      }
      else *ok = 0;
      free(packed);
      free(back);
      free(tmp);
    }
  }
}


#if defined(PACK_FORTRAN)
static size_t vpack_fortran(void* packed, const void* values, libxs_data_t type, size_t n,
  int maxbits, int group)
{
  LIBXS_UNUSED(type);
  LIBXS_UNUSED(group);
  return vpack_f((int64_t*)packed, (const int64_t*)values, n, maxbits);
}


static size_t vunpack_fortran(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n)
{
  LIBXS_UNUSED(npacked);
  LIBXS_UNUSED(type);
  return vunpack_f((int64_t*)values, (const int64_t*)packed, n);
}
#endif


/**
 * Packs every stream at its width (one stream per width, as CP2K's containers), per record if
 * given: a length word each (as CP2K stores it), and every record decodes into a reused buffer.
 */
static double vpack_bits(const stream_t* s, int group, vpack_fn pack, vunpack_fn unpack,
  double* seconds, double* useconds, int* ok, header_t* h)
{
  double bits = 0;
  int nbits;
  *seconds = *useconds = 0;
  for (nbits = 1; nbits <= MAXBITS; ++nbits) {
    if (0 != s[nbits].n) {
      const size_t n = s[nbits].n, r = (0 != record ? LIBXS_MIN(record, n) : n), nrec = (n + r - 1) / r;
      /* 8 interleaved streams (vpack.f) pad each stream to a word: up to 7 words more */
      const size_t bound = nrec * (libxs_vpack_bound(r, nbits, group) + 7);
      uint64_t *const packed = (uint64_t*)malloc(bound * sizeof(uint64_t));
      int64_t *const back = (int64_t*)malloc(n * sizeof(int64_t));
      int64_t *const tmp = (int64_t*)malloc(r * sizeof(int64_t));
      size_t *const len = (size_t*)malloc(nrec * sizeof(size_t));
      if (NULL != packed && NULL != back && NULL != tmp && NULL != len) {
        double best = DBL_MAX, ubest = DBL_MAX;
        size_t k, off = 0;
        int rep;
        first_touch(packed, bound * sizeof(uint64_t));
        first_touch(back, n * sizeof(int64_t));
        first_touch(tmp, r * sizeof(int64_t));
        for (rep = 0; rep < nrepeat; ++rep) {
          const libxs_timer_tick_t t0 = libxs_timer_tick();
          double d;
          for (k = 0, off = 0; k < nrec; ++k) {
            len[k] = pack(packed + off, s[nbits].q + k * r, LIBXS_DATATYPE_I64, LIBXS_MIN(r, n - k * r), nbits, group);
            off += len[k];
          }
          d = libxs_timer_duration(t0, libxs_timer_tick());
          best = LIBXS_MIN(best, d);
        }
        *seconds += best;
        for (rep = 0; rep < nrepeat; ++rep) {
          double d = 0;
          for (k = 0, off = 0; k < nrec; ++k) {
            const size_t m = LIBXS_MIN(r, n - k * r);
            int64_t *const dst = (0 != record ? tmp : back);
            const libxs_timer_tick_t t1 = libxs_timer_tick();
            const size_t nread = unpack(dst, packed + off, len[k], LIBXS_DATATYPE_I64, m);
            d += libxs_timer_duration(t1, libxs_timer_tick());
            if (nread != len[k]) *ok = 0;
            if (dst != back) memcpy(back + k * r, dst, m * sizeof(int64_t));
            if (NULL != h && 0 == rep) {
              const size_t ngroups = (m + group - 1) / group;
              h->payload += 64.0 * ((double)len[k] - (double)((1 + ngroups + 7) / 8));
              header_costs(h, (const unsigned char*)(packed + off) + 1, ngroups);
            }
            off += len[k];
          }
          ubest = LIBXS_MIN(ubest, d);
        }
        *useconds += ubest;
        if (0 != memcmp(back, s[nbits].q, n * sizeof(int64_t))) *ok = 0;
        bits += 64.0 * (double)(off + (0 != record ? nrec : 0));
      }
      else *ok = 0;
      free(packed);
      free(back);
      free(tmp);
      free(len);
    }
  }
  return bits;
}


#define KD_MASK(N) (64 > (N) ? ((((uint64_t)1) << (N)) - 1) : ~((uint64_t)0))

typedef struct kd_bits_t {
  uint64_t* out; /* zeroed */
  size_t nbits;
} kd_bits_t;

/* A box of whole rows (s0 values) over [a1,b1) x [a2,b2) of axes 1 and 2. */
typedef struct kd_box_t {
  size_t s0, s1;
  int a1, b1, a2, b2;
} kd_box_t;

typedef struct kd_leaf_t {
  size_t count;
  double bits;
  int w, v, e, list, lg;
} kd_leaf_t;


/* Appends the n low bits of v (no higher bits set). */
static void kd_put(kd_bits_t* bw, uint64_t v, int n)
{
  if (0 < n) {
    const size_t k = bw->nbits >> 6;
    const int s = (int)(bw->nbits & 63);
    bw->out[k] |= v << s;
    if (64 < s + n) bw->out[k + 1] |= v >> (64 - s);
    bw->nbits += (size_t)n;
  }
}


#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
/* n values of v bits from bit position *bit: 8 per step from a window of 16 words (reads 16 beyond). */
static void kd_get_n(uint64_t* dst, size_t n, const uint64_t* in, size_t* bit, int v)
{
  const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), one = _mm512_set1_epi64(1);
  const __m512i step = _mm512_mullo_epi64(lane, _mm512_set1_epi64(v)), word = _mm512_set1_epi64(63);
  const __m512i sixtyfour = _mm512_set1_epi64(64), mask = _mm512_set1_epi64((long long)KD_MASK(v));
  size_t b = *bit, k;
  for (k = 0; k < n; k += 8, b += 8 * (size_t)v) {
    const uint64_t *const p = in + (b >> 6);
    const __m512i bits = _mm512_add_epi64(_mm512_set1_epi64((long long)(b & 63)), step);
    const __m512i lo = _mm512_srli_epi64(bits, 6), hi = _mm512_add_epi64(lo, one);
    const __m512i shift = _mm512_and_si512(bits, word), back = _mm512_sub_epi64(sixtyfour, shift);
    const __m512i w0 = _mm512_loadu_si512((const void*)p), w1 = _mm512_loadu_si512((const void*)(p + 8));
    __m512i x = _mm512_or_si512( /* a shift by 64 yields zero */
      _mm512_srlv_epi64(_mm512_permutex2var_epi64(w0, lo, w1), shift),
      _mm512_sllv_epi64(_mm512_permutex2var_epi64(w0, hi, w1), back));
    x = _mm512_and_si512(x, mask);
    if (k + 8 <= n) _mm512_storeu_si512((void*)(dst + k), x);
    else _mm512_mask_storeu_epi64(dst + k, (__mmask8)((1u << (n - k)) - 1), x);
  }
  *bit += n * (size_t)v;
}


/* As kd_get_n, the values stored with the sign unfolded (zig-zag). */
static void kd_get_zz(int64_t* dst, size_t n, const uint64_t* in, size_t* bit, int v)
{
  const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0), one = _mm512_set1_epi64(1);
  const __m512i step = _mm512_mullo_epi64(lane, _mm512_set1_epi64(v)), word = _mm512_set1_epi64(63);
  const __m512i sixtyfour = _mm512_set1_epi64(64), mask = _mm512_set1_epi64((long long)KD_MASK(v));
  const __m512i zero = _mm512_setzero_si512();
  size_t b = *bit, k;
  for (k = 0; k < n; k += 8, b += 8 * (size_t)v) {
    const uint64_t *const p = in + (b >> 6);
    const __m512i bits = _mm512_add_epi64(_mm512_set1_epi64((long long)(b & 63)), step);
    const __m512i lo = _mm512_srli_epi64(bits, 6), hi = _mm512_add_epi64(lo, one);
    const __m512i shift = _mm512_and_si512(bits, word), back = _mm512_sub_epi64(sixtyfour, shift);
    const __m512i w0 = _mm512_loadu_si512((const void*)p), w1 = _mm512_loadu_si512((const void*)(p + 8));
    __m512i x = _mm512_or_si512( /* a shift by 64 yields zero */
      _mm512_srlv_epi64(_mm512_permutex2var_epi64(w0, lo, w1), shift),
      _mm512_sllv_epi64(_mm512_permutex2var_epi64(w0, hi, w1), back));
    x = _mm512_and_si512(x, mask);
    x = _mm512_xor_si512(_mm512_srli_epi64(x, 1), _mm512_sub_epi64(zero, _mm512_and_si512(x, one)));
    if (k + 8 <= n) _mm512_storeu_si512((void*)(dst + k), x);
    else _mm512_mask_storeu_epi64(dst + k, (__mmask8)((1u << (n - k)) - 1), x);
  }
  *bit += n * (size_t)v;
}
#else
/* n values of v bits from bit position *bit, no branch (reads one word beyond). */
static void kd_get_n(uint64_t* dst, size_t n, const uint64_t* in, size_t* bit, int v)
{
  const uint64_t mask = KD_MASK(v);
  size_t b = *bit, k;
  for (k = 0; k < n; ++k, b += (size_t)v) {
    const uint64_t* const p = in + (b >> 6);
    const int s = (int)(b & 63);
    dst[k] = ((p[0] >> s) | ((p[1] << 1) << (63 - s))) & mask;
  }
  *bit = b;
}


static void kd_get_zz(int64_t* dst, size_t n, const uint64_t* in, size_t* bit, int v)
{
  const uint64_t mask = KD_MASK(v);
  size_t b = *bit, k;
  for (k = 0; k < n; ++k, b += (size_t)v) {
    const uint64_t* const p = in + (b >> 6);
    const int s = (int)(b & 63);
    const uint64_t x = ((p[0] >> s) | ((p[1] << 1) << (63 - s))) & mask;
    dst[k] = (int64_t)((x >> 1) ^ (0 - (x & 1)));
  }
  *bit = b;
}
#endif


#if defined(__GNUC__)
# define PACK_POPCOUNT64(X) __builtin_popcountll(X)
#else
# define PACK_POPCOUNT64(X) pack_popcount64(X)
static int pack_popcount64(uint64_t x)
{
  int result = 0;
  for (; 0 != x; x &= x - 1) ++result;
  return result;
}
#endif

#if defined(__AVX512F__)
/* 8 bits of a bitmap from bit position k (the word after the last one is zero). */
static unsigned int kd_byte(const uint64_t* bitmap, size_t k)
{
  const uint64_t* const p = bitmap + (k >> 6);
  const int s = (int)(k & 63);
  return (unsigned int)(((p[0] >> s) | ((p[1] << 1) << (63 - s))) & 0xFF);
}


/* Patches a run of n decoded values (sign unfolded) from bitmap position k: XOR with the high parts
 * shifted by v-1 (v > 0), or the high parts as the values (v = 0); returns the high parts used. */
static size_t kd_patch_run(int64_t* dst, size_t n, const uint64_t* bitmap, size_t k, const uint64_t* high, int v)
{
  const __m128i sv = _mm_cvtsi32_si128(0 < v ? v - 1 : 0);
  const __m512i one = _mm512_set1_epi64(1), zero = _mm512_setzero_si512();
  size_t i, j = 0;
  for (i = 0; i < n; i += 8) {
    const __mmask8 store = (__mmask8)(i + 8 <= n ? 0xFF : ((1u << (n - i)) - 1));
    const __mmask8 m = (__mmask8)(kd_byte(bitmap, k + i) & store);
    const __m512i h = _mm512_maskz_expandloadu_epi64(m, high + j);
    __m512i x = _mm512_maskz_loadu_epi64(store, dst + i);
    if (0 < v) x = _mm512_xor_si512(x, _mm512_sll_epi64(h, sv));
    else x = _mm512_mask_mov_epi64(x, m, _mm512_xor_si512(_mm512_srli_epi64(h, 1),
      _mm512_sub_epi64(zero, _mm512_and_si512(h, one))));
    _mm512_mask_storeu_epi64(dst + i, store, x);
    j += (size_t)PACK_POPCOUNT64(m);
  }
  return j;
}
#else
static size_t kd_patch_run(int64_t* dst, size_t n, const uint64_t* bitmap, size_t k, const uint64_t* high, int v)
{
  size_t i, j = 0;
  for (i = 0; i < n; ++i) {
    if (0 != ((bitmap[(k + i) >> 6] >> ((k + i) & 63)) & 1)) {
      const uint64_t h = high[j++];
      dst[i] = (0 < v ? (int64_t)((uint64_t)dst[i] ^ (h << (v - 1))) : (int64_t)((h >> 1) ^ (0 - (h & 1))));
    }
  }
  return j;
}
#endif


/* Position in the block of the k-th value of a box (axis 2 outer, axis 1 inner, then the row). */
static size_t kd_pos(const kd_box_t* b, size_t k)
{
  const size_t r = k / b->s0, len1 = (size_t)(b->b1 - b->a1);
  return b->s0 * ((size_t)b->a1 + r % len1 + b->s1 * ((size_t)b->a2 + r / len1)) + k % b->s0;
}


static int kd_flag(const kd_box_t* b)
{
  return (1 < b->b1 - b->a1 || 1 < b->b2 - b->a2) ? 1 : 0;
}


/* Bits of a box as a leaf: a byte (width and a flag), or with exceptions: the regular width v, a byte
 * (w-v and a flag for the positions: bitmap, or a 16-bit count and indexes), and the high parts. */
static void kd_leaf(kd_leaf_t* lf, const uint64_t* z, const kd_box_t* b)
{
  int hist[MAXBITS + 1], above = 0, i, v;
  size_t k;
  lf->count = b->s0 * (size_t)(b->b1 - b->a1) * (size_t)(b->b2 - b->a2);
  for (i = 0; i <= MAXBITS; ++i) hist[i] = 0;
  lf->w = 0;
  for (k = 0; k < lf->count; ++k) {
    const int nb = (int)LIBXS_NBITS(z[kd_pos(b, k)]);
    ++hist[nb];
    lf->w = LIBXS_MAX(lf->w, nb);
  }
  for (lf->lg = 0; ((size_t)1 << lf->lg) < lf->count; ++lf->lg);
  lf->v = lf->w;
  lf->e = lf->list = 0;
  lf->bits = 8 + (double)lf->count * lf->w;
  for (v = (0 != kd_exc ? lf->w - 1 : -1); 0 <= v; --v) {
    const double list = (65536 > lf->count ? 16 + (double)(above + hist[v + 1]) * lf->lg : (double)lf->count + 1);
    double bits;
    above += hist[v + 1]; /* values wider than v */
    bits = 16 + LIBXS_MIN((double)lf->count, list) + (double)lf->count * v + (double)above * (lf->w - v);
    if (bits < lf->bits) {
      lf->bits = bits;
      lf->v = v;
      lf->e = above;
      lf->list = (list < (double)lf->count ? 1 : 0);
    }
  }
}


/* A leaf as phase 1 of the decoder finds it: box, widths, positions (0: none, 1: list, 2: bitmap). */
typedef struct kd_node_t {
  int a1, b1, a2, b2, v, w, mode;
  size_t e;
} kd_node_t;


/* The header bytes of a leaf go to bh, its data (positions, values, high parts) to bp. */
static void kd_emit(kd_bits_t* bh, kd_bits_t* bp, const kd_leaf_t* lf, const uint64_t* z, const kd_box_t* b)
{
  const int exc = (lf->v < lf->w ? 1 : 0);
  size_t k;
  kd_put(bh, (uint64_t)lf->v | ((uint64_t)exc << 7), 8);
  if (0 != exc) {
    kd_put(bh, (uint64_t)(lf->w - lf->v) | ((uint64_t)lf->list << 7), 8);
    if (0 != lf->list) {
      kd_put(bh, (uint64_t)lf->e, 16);
      for (k = 0; k < lf->count; ++k) {
        if (lf->v < (int)LIBXS_NBITS(z[kd_pos(b, k)])) kd_put(bp, (uint64_t)k, lf->lg);
      }
    }
    else {
      for (k = 0; k < lf->count; ++k) kd_put(bp, lf->v < (int)LIBXS_NBITS(z[kd_pos(b, k)]) ? 1 : 0, 1);
    }
  }
  for (k = 0; k < lf->count; ++k) kd_put(bp, z[kd_pos(b, k)] & KD_MASK(lf->v), lf->v);
  if (0 != exc) {
    for (k = 0; k < lf->count; ++k) {
      const uint64_t x = z[kd_pos(b, k)];
      if (lf->v < (int)LIBXS_NBITS(x)) kd_put(bp, x >> lf->v, lf->w - lf->v);
    }
  }
}


/* Halves a box at the middle of axis d (1 or 2). */
static void kd_halves(const kd_box_t* b, int d, kd_box_t* lo, kd_box_t* hi)
{
  *lo = *hi = *b;
  if (1 == d) lo->b1 = hi->a1 = b->a1 + (b->b1 - b->a1) / 2;
  else lo->b2 = hi->a2 = b->a2 + (b->b2 - b->a2) / 2;
}


/* Greedy: a box is split at the middle of axis 1 or 2 if both halves as leaves take fewer bits;
 * the tree bits go to bt, the leaf headers to bh, the data to bp. */
static void kd_encode(kd_bits_t* bt, kd_bits_t* bh, kd_bits_t* bp, const uint64_t* z, const kd_box_t* b)
{
  const int can1 = (1 < b->b1 - b->a1 ? 1 : 0), can2 = (1 < b->b2 - b->a2 ? 1 : 0);
  kd_box_t lo, hi;
  kd_leaf_t lf;
  int axis = 0;
  kd_leaf(&lf, z, b);
  if (0 != can1 || 0 != can2) {
    const int dbits = can1 & can2;
    double best = 1 + lf.bits;
    int d;
    for (d = 1; d <= 2; ++d) {
      if (0 != (1 == d ? can1 : can2)) {
        kd_box_t l, h;
        kd_leaf_t ll, lh;
        double cost;
        kd_halves(b, d, &l, &h);
        kd_leaf(&ll, z, &l);
        kd_leaf(&lh, z, &h);
        cost = 1 + dbits + ll.bits + kd_flag(&l) + lh.bits + kd_flag(&h);
        if (cost < best && kd_minleaf <= ll.count && kd_minleaf <= lh.count) {
          best = cost;
          axis = d;
          lo = l;
          hi = h;
        }
      }
    }
    kd_put(bt, 0 != axis ? 1 : 0, 1);
    if (0 != axis && 0 != dbits) kd_put(bt, (uint64_t)(axis - 1), 1);
  }
  if (0 != axis) {
    kd_encode(bt, bh, bp, z, &lo);
    kd_encode(bt, bh, bp, z, &hi);
  }
  else kd_emit(bh, bp, &lf, z, b);
}


static int kd_bit(const uint64_t* in, size_t* bit)
{
  const int result = (int)((in[*bit >> 6] >> (*bit & 63)) & 1);
  ++*bit;
  return result;
}


/* Decoder phase 1: the boxes of the leaves in order from the tree bits. */
static void kd_scan(kd_node_t* leaves, int* nleaves, const uint64_t* in, size_t* bit, const kd_box_t* b)
{
  const int can1 = (1 < b->b1 - b->a1 ? 1 : 0), can2 = (1 < b->b2 - b->a2 ? 1 : 0);
  const int split = ((0 != can1 || 0 != can2) ? kd_bit(in, bit) : 0);
  if (0 != split) {
    const int axis = ((0 != can1 && 0 != can2) ? 1 + kd_bit(in, bit) : (0 != can1 ? 1 : 2));
    kd_box_t lo, hi;
    kd_halves(b, axis, &lo, &hi);
    kd_scan(leaves, nleaves, in, bit, &lo);
    kd_scan(leaves, nleaves, in, bit, &hi);
  }
  else {
    kd_node_t *const lf = leaves + (*nleaves)++;
    lf->a1 = b->a1;
    lf->b1 = b->b1;
    lf->a2 = b->a2;
    lf->b2 = b->b2;
  }
}


/* Decoder phase 1, the leaf headers (bytes); returns the bytes read. */
static size_t kd_heads(kd_node_t* leaves, int nleaves, const unsigned char* in)
{
  size_t result = 0;
  int l;
  for (l = 0; l < nleaves; ++l) {
    kd_node_t *const lf = leaves + l;
    const unsigned int h0 = in[result++];
    lf->v = lf->w = (int)(h0 & 0x7F);
    lf->mode = 0;
    lf->e = 0;
    if (0 != (h0 & 0x80)) {
      const unsigned int h1 = in[result++];
      lf->w += (int)(h1 & 0x7F);
      lf->mode = (0 != (h1 & 0x80) ? 1 : 2);
      if (1 == lf->mode) {
        lf->e = (size_t)in[result] | ((size_t)in[result + 1] << 8);
        result += 2;
      }
    }
  }
  return result;
}


/* Decoder phase 2: a leaf's data straight into its rows of the block (sign unfolded), then the
 * exceptions patched in place; pos holds positions (bitmap words or indexes), high a box. */
static size_t kd_fill(int64_t* out, uint64_t* high, uint64_t* pos, const kd_node_t* lf, size_t s0, size_t s1,
  const uint64_t* in, size_t* bit)
{
  const size_t count = s0 * (size_t)(lf->b1 - lf->a1) * (size_t)(lf->b2 - lf->a2);
  const int full = (0 == lf->a1 && (int)s1 == lf->b1 ? 1 : 0); /* all of axis 1: one contiguous run */
  const size_t run = (0 != full ? count : s0 * (size_t)(lf->b1 - lf->a1));
  const int b2 = (0 != full ? lf->a2 + 1 : lf->b2);
  size_t k, e = lf->e;
  int i2;
  if (2 == lf->mode) { /* bitmap: words, the bits beyond count cleared, a zero word after */
    const size_t nwords = (count + 63) / 64, start = *bit;
    kd_get_n(pos, nwords, in, bit, 64);
    if (0 != (count & 63)) pos[nwords - 1] &= KD_MASK((int)(count & 63));
    pos[nwords] = 0;
    *bit = start + count;
    for (k = 0; k < nwords; ++k) e += (size_t)PACK_POPCOUNT64(pos[k]);
  }
  else if (1 == lf->mode) {
    int lg = 0;
    for (; ((size_t)1 << lg) < count; ++lg);
    kd_get_n(pos, e, in, bit, lg);
  }
  for (i2 = lf->a2; i2 < b2; ++i2) { /* a contiguous run of rows per index of axis 2 */
    kd_get_zz(out + s0 * ((size_t)lf->a1 + s1 * (size_t)i2), run, in, bit, lf->v);
  }
  if (0 != e) { /* high parts at w-v bits */
    kd_get_n(high, e, in, bit, lf->w - lf->v);
    if (2 == lf->mode) {
      size_t j = 0;
      for (i2 = lf->a2, k = 0; i2 < b2; ++i2, k += run) {
        j += kd_patch_run(out + s0 * ((size_t)lf->a1 + s1 * (size_t)i2), run, pos, k, high + j, lf->v);
      }
    }
    else {
      for (k = 0; k < e; ++k) {
        const size_t r = (size_t)pos[k] / run, c = (size_t)pos[k] % run;
        int64_t *const x = out + s0 * ((size_t)lf->a1 + s1 * ((size_t)lf->a2 + r)) + c;
        const uint64_t h = high[k];
        *x = (0 < lf->v ? (int64_t)((uint64_t)*x ^ (h << (lf->v - 1))) : (int64_t)((h >> 1) ^ (0 - (h & 1))));
      }
    }
  }
  return (size_t)(b2 - lf->a2); /* runs */
}


/**
 * Prototype: every block as k-d boxes of whole rows with exceptions, a length word each; the headers
 * of a block first, then its data. Decoding in two phases: the leaves from the headers, then each.
 */
static double kd_rows_bits(const stats_t* st, double* seconds, double* useconds, int* ok)
{
  const size_t nblk = st->shapes.n / 3;
  size_t *const woff = (size_t*)malloc((nblk + 1) * sizeof(size_t)), *const nw = (size_t*)malloc((nblk + 1) * sizeof(size_t));
  size_t b, off, maxn = 0, total = 0;
  uint64_t *packed = NULL, *z = NULL, *tmp = NULL, *pos = NULL, *hbuf = NULL, *pbuf = NULL, *tbuf = NULL;
  kd_node_t* leaves = NULL;
  int64_t* out = NULL;
  double bits = 0;
  int k;
  *seconds = *useconds = 0;
  if (NULL != woff && NULL != nw) {
    for (b = 0, off = 0; b < nblk; ++b) {
      const size_t s0 = (size_t)st->shapes.q[3 * b], s1 = (size_t)st->shapes.q[3 * b + 1];
      const size_t s2 = (size_t)st->shapes.q[3 * b + 2], n = s0 * s1 * s2;
      woff[b] = total;
      total += n + s1 * s2 + 4; /* bound: 64 bits per value, a leaf header per row, slack */
      if (b + 1 == nblk) total += 16; /* read beyond the last block */
      maxn = LIBXS_MAX(maxn, n);
      off += n;
    }
    packed = (uint64_t*)calloc(LIBXS_MAX(total, 1), sizeof(uint64_t));
    z = (uint64_t*)malloc(LIBXS_MAX(maxn, 1) * sizeof(uint64_t));
    tmp = (uint64_t*)malloc(2 * LIBXS_MAX(maxn, 1) * sizeof(uint64_t)); /* values and high parts */
    pos = (uint64_t*)malloc((LIBXS_MAX(maxn, 1) + 1) * sizeof(uint64_t));
    hbuf = (uint64_t*)malloc((LIBXS_MAX(maxn, 1) + 4) * sizeof(uint64_t));
    pbuf = (uint64_t*)malloc((LIBXS_MAX(maxn, 1) + 4) * sizeof(uint64_t));
    tbuf = (uint64_t*)malloc((LIBXS_MAX(maxn, 1) + 4) * sizeof(uint64_t));
    leaves = (kd_node_t*)malloc(LIBXS_MAX(maxn, 1) * sizeof(kd_node_t));
    out = (int64_t*)malloc(LIBXS_MAX(maxn, 1) * sizeof(int64_t));
  }
  if (NULL != packed && NULL != z && NULL != tmp && NULL != pos && NULL != hbuf && NULL != pbuf
    && NULL != tbuf && NULL != leaves && NULL != out)
  {
    libxs_timer_tick_t t0 = libxs_timer_tick();
    for (b = 0, off = 0; b < nblk; ++b) {
      const size_t s0 = (size_t)st->shapes.q[3 * b], s1 = (size_t)st->shapes.q[3 * b + 1];
      const size_t s2 = (size_t)st->shapes.q[3 * b + 2], n = s0 * s1 * s2;
      kd_bits_t bt, bh, bp, bw;
      kd_box_t box;
      size_t i;
      for (i = 0; i < n; ++i) z[i] = zigzag(st->blocks.q[off + i]);
      box.s0 = s0;
      box.s1 = s1;
      box.a1 = box.a2 = 0;
      box.b1 = (int)s1;
      box.b2 = (int)s2;
      memset(hbuf, 0, (n + 4) * sizeof(uint64_t));
      memset(pbuf, 0, (n + 4) * sizeof(uint64_t));
      memset(tbuf, 0, (n + 4) * sizeof(uint64_t));
      bt.out = tbuf;
      bh.out = hbuf;
      bp.out = pbuf;
      bt.nbits = bh.nbits = bp.nbits = 0;
      kd_encode(&bt, &bh, &bp, z, &box);
      bw.out = packed + woff[b]; /* the tree (to a byte), the leaf headers (bytes), then the data */
      bw.nbits = 0;
      for (i = 0; 64 * i < bt.nbits; ++i) kd_put(&bw, tbuf[i], (int)LIBXS_MIN((size_t)64, bt.nbits - 64 * i));
      bw.nbits = (bw.nbits + 7) & ~(size_t)7;
      for (i = 0; 64 * i < bh.nbits; ++i) kd_put(&bw, hbuf[i], (int)LIBXS_MIN((size_t)64, bh.nbits - 64 * i));
      for (i = 0; 64 * i < bp.nbits; ++i) kd_put(&bw, pbuf[i], (int)LIBXS_MIN((size_t)64, bp.nbits - 64 * i));
      nw[b] = (bw.nbits + 63) / 64;
      off += n;
    }
    *seconds = libxs_timer_duration(t0, libxs_timer_tick());
    *useconds = DBL_MAX;
    for (k = 0; k <= nrepeat; ++k) { /* timed, then checked */
      t0 = libxs_timer_tick();
      for (b = 0, off = 0; b < nblk; ++b) {
        const size_t s0 = (size_t)st->shapes.q[3 * b], s1 = (size_t)st->shapes.q[3 * b + 1];
        const size_t s2 = (size_t)st->shapes.q[3 * b + 2], n = s0 * s1 * s2;
        const uint64_t *const in = packed + woff[b];
        kd_box_t box;
        size_t bit = 0;
        size_t nruns = 0;
        int nleaves = 0, l;
        box.s0 = s0;
        box.s1 = s1;
        box.a1 = box.a2 = 0;
        box.b1 = (int)s1;
        box.b2 = (int)s2;
        kd_scan(leaves, &nleaves, in, &bit, &box);
        bit = (bit + 7) & ~(size_t)7;
        bit += 8 * kd_heads(leaves, nleaves, (const unsigned char*)in + bit / 8);
        for (l = 0; l < nleaves; ++l) nruns += kd_fill(out, tmp, pos, leaves + l, s0, s1, in, &bit);
        if (nrepeat == k) {
          if ((bit + 63) / 64 != nw[b] || 0 != memcmp(out, st->blocks.q + off, n * sizeof(int64_t))) *ok = 0;
          kd_runs += (double)nruns;
          kd_leaves += (double)nleaves;
          bits += 64.0 * (double)(nw[b] + 1);
        }
        off += n;
      }
      if (nrepeat > k) {
        const double d = libxs_timer_duration(t0, libxs_timer_tick());
        *useconds = LIBXS_MIN(*useconds, d);
      }
    }
  }
  else *ok = 0;
  free(woff);
  free(nw);
  free(packed);
  free(z);
  free(tmp);
  free(pos);
  free(hbuf);
  free(pbuf);
  free(tbuf);
  free(leaves);
  free(out);
  return bits;
}


/* Appends n bits (at most 64) to a word stream, no branch on the data. */
LIBXS_INLINE_ALWAYS void rice_put(rice_writer_t* bw, uint64_t v, int n)
{
  const int next = bw->fill + n;
  const uint64_t acc = bw->acc | (v << bw->fill), hi = (v >> 1) >> (63 - bw->fill);
  *bw->out = acc;
  bw->out += next >> 6;
  bw->acc = (64 <= next ? hi : acc);
  bw->fill = next & 63;
}


/* Appends a code and its value. */
LIBXS_INLINE_ALWAYS void rice_value_put(rice_writer_t* bw, unsigned int s, uint64_t x, int nb)
{
  const unsigned int m = s >> RICE_K;
  const int nc = (int)m + 1 + RICE_K;
  if (m < RICE_ESC && nc + nb <= 64) {
    const uint64_t code = ((((uint64_t)1) << m) - 1) | (((uint64_t)(s & ((1U << RICE_K) - 1))) << (m + 1));
    rice_put(bw, code | (x << nc), nc + nb);
  }
  else {
    if (m < RICE_ESC) {
      rice_put(bw, (((uint64_t)1) << m) - 1, (int)m + 1);
      rice_put(bw, s & ((1U << RICE_K) - 1), RICE_K);
    }
    else {
      rice_put(bw, (((uint64_t)1) << RICE_ESC) - 1, RICE_ESC + 1);
      rice_put(bw, s, 8);
    }
    rice_put(bw, x, nb);
  }
}


/* Appends a run length. */
LIBXS_INLINE_ALWAYS void rice_run_put(rice_writer_t* bw, size_t run)
{
  const int b = (int)LIBXS_NBITS(run + 1);
  rice_put(bw, (((uint64_t)1) << (b - 1)) - 1, b);
  rice_put(bw, (run + 1) & KD_MASK(b - 1), b - 1);
}


/* Copies a finished stream from the words at tmp to out; returns the bytes it takes. */
static size_t rice_flush(unsigned char* out, const uint64_t* tmp, const rice_writer_t* bw)
{
  const size_t nwords = (size_t)(bw->out - tmp), nrest = (size_t)((bw->fill + 7) / 8);
  memcpy(out, tmp, 8 * nwords);
  memcpy(out + 8 * nwords, &bw->acc, nrest);
  return 8 * nwords + nrest;
}


/* Encodes n values in w lanes, cap words of tmp per stream. Returns the bytes written, offs[l] is the
 * byte position of stream l. */
static size_t rice_encode(unsigned char* out, const int64_t* q, size_t n, int w, size_t* offs,
  uint64_t* tmp, size_t cap)
{
  const size_t ng = (n + (size_t)w - 1) / (size_t)w;
  rice_writer_t bw[RICE_MAXLANES + 1];
  int p1[RICE_MAXLANES], p2[RICE_MAXLANES];
  int64_t pad[RICE_MAXLANES];
  size_t pos = 0, g, run = 0;
  int l, zero = 1;
  for (l = 0; l <= w; ++l) {
    bw[l].out = tmp + (size_t)l * cap;
    bw[l].acc = 0;
    bw[l].fill = 0;
    if (l < w) p1[l] = p2[l] = 0;
  }
  for (g = 0; g < ng; ++g) {
    const size_t i = g * (size_t)w;
    const int64_t* v = q + i;
    int64_t any = 0;
    if (n < i + (size_t)w) {
      memset(pad, 0, sizeof(pad));
      memcpy(pad, v, (n - i) * sizeof(int64_t));
      v = pad;
    }
    for (l = 0; l < w; ++l) any |= v[l];
    if ((0 == any) != (0 != zero)) {
      rice_run_put(bw + w, run);
      zero = !zero;
      run = 0;
    }
    ++run;
    if (0 != any) {
      for (l = 0; l < w; ++l) {
        const uint64_t z = zigzag(v[l]);
        if (0 != z) {
          const int b = (int)LIBXS_NBITS(z), d = b - ((p1[l] + p2[l] + 1) >> 1);
          rice_value_put(bw + l, (unsigned int)(0 <= d ? 2 * d : -2 * d - 1) + 1, z ^ (((uint64_t)1) << (b - 1)), b - 1);
          p2[l] = p1[l];
          p1[l] = b;
        }
        else rice_put(bw + l, 0, 1 + RICE_K);
      }
    }
  }
  rice_run_put(bw + w, run);
  for (l = 0; l <= w; ++l) {
    offs[l] = pos;
    pos += rice_flush(out + pos, tmp + (size_t)l * cap, bw + l);
  }
  return pos;
}


/* Next run length. */
LIBXS_INLINE_ALWAYS size_t rice_run(const unsigned char* in, size_t* bit)
{
  uint64_t v;
  int g;
  memcpy(&v, in + (*bit >> 3), sizeof(v));
  v >>= *bit & 7;
  g = (int)LIBXS_INTRINSICS_BITSCANFWD64(~v) + 1;
  *bit += (size_t)(2 * g - 1);
  return (size_t)((((uint64_t)1) << (g - 1)) | ((v >> g) & KD_MASK(g - 1))) - 1;
}


/* Next value of a lane, on the general path. */
LIBXS_INLINE_ALWAYS int64_t rice_value(const unsigned char* in, size_t* bit, uint64_t* p1, uint64_t* p2)
{
  int64_t result = 0;
  uint64_t v;
  unsigned int s;
  int m;
  memcpy(&v, in + (*bit >> 3), sizeof(v));
  v >>= *bit & 7;
  m = (int)LIBXS_INTRINSICS_BITSCANFWD64(~v);
  if (m < RICE_ESC) {
    s = ((unsigned int)m << RICE_K) | ((unsigned int)(v >> (m + 1)) & ((1U << RICE_K) - 1));
    *bit += (size_t)m + 1 + RICE_K;
  }
  else {
    s = (unsigned int)(v >> (RICE_ESC + 1)) & 0xFF;
    *bit += RICE_ESC + 9;
  }
  if (0 != s) {
    const unsigned int u = s - 1;
    const int d = (int)(u >> 1) ^ -(int)(u & 1), b = (int)((*p1 + *p2 + 1) >> 1) + d, nb = b - 1;
    const int lo = LIBXS_MIN(nb, 32);
    uint64_t hi = 0;
    memcpy(&v, in + (*bit >> 3), sizeof(v));
    v = (v >> (*bit & 7)) & KD_MASK(lo);
    *bit += (size_t)lo;
    if (lo < nb) {
      memcpy(&hi, in + (*bit >> 3), sizeof(hi));
      hi = (hi >> (*bit & 7)) & KD_MASK(nb - lo);
      *bit += (size_t)(nb - lo);
    }
    v |= (hi << lo) | (((uint64_t)1) << nb);
    result = (int64_t)(v >> 1) ^ -(int64_t)(v & 1);
    *p2 = *p1;
    *p1 = (uint64_t)b;
  }
  return result;
}


/* Decodes n values (padded to a group): q holds room for a whole group more. */
static void rice_decode(int64_t* q, size_t n, const unsigned char* in, int w, const size_t* offs)
{
  const size_t ng = (n + (size_t)w - 1) / (size_t)w;
  uint64_t p1[RICE_MAXLANES], p2[RICE_MAXLANES];
  size_t bit[RICE_MAXLANES], cbit = 8 * offs[w], g = 0;
  int l, zero = 1;
  for (l = 0; l < w; ++l) {
    bit[l] = 8 * offs[l];
    p1[l] = p2[l] = 0;
  }
  while (g < ng) {
    const size_t run = rice_run(in, &cbit);
    if (0 != zero) memset(q + g * (size_t)w, 0, run * (size_t)w * sizeof(*q));
    else {
      size_t k;
      for (k = 0; k < run * (size_t)w; ++k) {
        l = (int)(k % (size_t)w);
        q[g * (size_t)w + k] = rice_value(in, bit + l, p1 + l, p2 + l);
      }
    }
    g += run;
    zero = !zero;
  }
}


#if defined(LIBXS_INTRINSICS_AVX2)
/* Decodes a group of four lanes. */
LIBXS_INLINE_ALWAYS LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void rice_step_avx2(rice_avx2_t* st, int64_t* q, const unsigned char* in)
{
  const __m256i lut = _mm256_setr_epi8(0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
    0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4);
  const __m256i one = _mm256_set1_epi64x(1), three = _mm256_set1_epi64x(3), four = _mm256_set1_epi64x(4);
  const __m256i nibble = _mm256_set1_epi64x(15), zeros = _mm256_setzero_si256();
  const __m256i v = _mm256_srlv_epi64(_mm256_i64gather_epi64((const long long*)in, _mm256_srli_epi64(st->bit, 3), 1),
    _mm256_and_si256(st->bit, _mm256_set1_epi64x(7)));
  const __m256i lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, nibble));
  const __m256i hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi64(v, 4), nibble));
  const __m256i m = _mm256_add_epi64(lo, _mm256_and_si256(_mm256_cmpeq_epi64(lo, four), hi));
  const __m256i adv = _mm256_add_epi64(m, three);
  const __m256i s = _mm256_or_si256(_mm256_slli_epi64(m, RICE_K),
    _mm256_and_si256(_mm256_srlv_epi64(v, _mm256_add_epi64(m, one)), three));
  const __m256i zero = _mm256_cmpeq_epi64(s, zeros);
  const __m256i u = _mm256_sub_epi64(s, one);
  const __m256i d = _mm256_xor_si256(_mm256_srli_epi64(u, 1), _mm256_sub_epi64(zeros, _mm256_and_si256(u, one)));
  const __m256i b = _mm256_andnot_si256(zero, _mm256_add_epi64(d,
    _mm256_srli_epi64(_mm256_add_epi64(_mm256_add_epi64(st->p1, st->p2), one), 1)));
  const __m256i nb = _mm256_andnot_si256(zero, _mm256_sub_epi64(b, one));
  const __m256i used = _mm256_add_epi64(adv, nb);
  if (0 == _mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi64(m, _mm256_set1_epi64x(8)),
    _mm256_cmpgt_epi64(used, _mm256_set1_epi64x(57)))))
  {
    const __m256i top = _mm256_sllv_epi64(one, nb);
    const __m256i x = _mm256_andnot_si256(zero, _mm256_or_si256(top,
      _mm256_and_si256(_mm256_srlv_epi64(v, adv), _mm256_sub_epi64(top, one))));
    _mm256_storeu_si256((__m256i*)q, _mm256_xor_si256(_mm256_srli_epi64(x, 1),
      _mm256_sub_epi64(zeros, _mm256_and_si256(x, one))));
    st->bit = _mm256_add_epi64(st->bit, used);
    st->p2 = _mm256_blendv_epi8(st->p1, st->p2, zero);
    st->p1 = _mm256_blendv_epi8(b, st->p1, zero);
  }
  else {
    uint64_t bits[4], p1[4], p2[4];
    int l;
    _mm256_storeu_si256((__m256i*)bits, st->bit);
    _mm256_storeu_si256((__m256i*)p1, st->p1);
    _mm256_storeu_si256((__m256i*)p2, st->p2);
    for (l = 0; l < 4; ++l) {
      size_t bit = (size_t)bits[l];
      q[l] = rice_value(in, &bit, p1 + l, p2 + l);
      bits[l] = bit;
    }
    st->bit = _mm256_loadu_si256((const __m256i*)bits);
    st->p1 = _mm256_loadu_si256((const __m256i*)p1);
    st->p2 = _mm256_loadu_si256((const __m256i*)p2);
  }
}


/* As rice_decode for 4 or 8 lanes. */
static LIBXS_INLINE_NEVER LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void rice_decode_avx2(int64_t* q, size_t n, const unsigned char* in, int w, const size_t* offs)
{
  const size_t ng = (n + (size_t)w - 1) / (size_t)w;
  size_t bits[RICE_MAXLANES] = { 0 }, cbit = 8 * offs[w], g = 0;
  rice_avx2_t a, b;
  int l, zero = 1;
  for (l = 0; l < w; ++l) bits[l] = 8 * offs[l];
  a.bit = _mm256_loadu_si256((const __m256i*)bits);
  b.bit = (8 == w ? _mm256_loadu_si256((const __m256i*)(bits + 4)) : a.bit);
  a.p1 = a.p2 = b.p1 = b.p2 = _mm256_setzero_si256();
  while (g < ng) {
    const size_t run = rice_run(in, &cbit);
    if (0 != zero) memset(q + g * (size_t)w, 0, run * (size_t)w * sizeof(*q));
    else {
      int64_t* p = q + g * (size_t)w;
      size_t k;
      if (8 == w) {
        for (k = 0; k < run; ++k, p += 8) {
          rice_step_avx2(&a, p, in);
          rice_step_avx2(&b, p + 4, in);
        }
      }
      else {
        for (k = 0; k < run; ++k, p += 4) rice_step_avx2(&a, p, in);
      }
    }
    g += run;
    zero = !zero;
  }
}
#endif


#if defined(LIBXS_INTRINSICS_AVX512)
/* As rice_decode for 8 lanes. */
static LIBXS_INLINE_NEVER LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void rice_decode_avx512(int64_t* q, size_t n, const unsigned char* in, const size_t* offs)
{
  const size_t ng = (n + 7) / 8;
  const __m512i one = _mm512_set1_epi64(1), three = _mm512_set1_epi64(3), seven = _mm512_set1_epi64(7);
  const __m512i high = _mm512_set1_epi64(63), esc = _mm512_set1_epi64(RICE_ESC), reach = _mm512_set1_epi64(57);
  const __m512i zeros = _mm512_setzero_si512();
  __m512i bit = _mm512_set_epi64((long long)(8 * offs[7]), (long long)(8 * offs[6]), (long long)(8 * offs[5]),
    (long long)(8 * offs[4]), (long long)(8 * offs[3]), (long long)(8 * offs[2]), (long long)(8 * offs[1]),
    (long long)(8 * offs[0]));
  __m512i p1 = zeros, p2 = zeros;
  size_t cbit = 8 * offs[8], g = 0;
  int zero = 1;
  while (g < ng) {
    const size_t run = rice_run(in, &cbit);
    if (0 != zero) memset(q + g * 8, 0, run * 8 * sizeof(*q));
    else {
      int64_t* p = q + g * 8;
      size_t k;
      for (k = 0; k < run; ++k, p += 8) {
        const __m512i v = _mm512_srlv_epi64(_mm512_i64gather_epi64(_mm512_srli_epi64(bit, 3), in, 1),
          _mm512_and_si512(bit, seven));
        const __m512i ones = _mm512_andnot_si512(v, _mm512_add_epi64(v, one));
        const __m512i m = _mm512_sub_epi64(high, _mm512_lzcnt_epi64(ones));
        const __m512i s = _mm512_or_si512(_mm512_slli_epi64(m, RICE_K),
          _mm512_and_si512(_mm512_srlv_epi64(v, _mm512_add_epi64(m, one)), three));
        const __mmask8 nz = _mm512_test_epi64_mask(s, s);
        const __m512i u = _mm512_sub_epi64(s, one);
        const __m512i d = _mm512_xor_si512(_mm512_srli_epi64(u, 1), _mm512_sub_epi64(zeros, _mm512_and_si512(u, one)));
        const __m512i b = _mm512_maskz_add_epi64(nz, d,
          _mm512_srli_epi64(_mm512_add_epi64(_mm512_add_epi64(p1, p2), one), 1));
        const __m512i nb = _mm512_maskz_sub_epi64(nz, b, one);
        const __m512i adv = _mm512_add_epi64(m, three), used = _mm512_add_epi64(adv, nb);
        if (0 == (_mm512_cmpge_epu64_mask(m, esc) | _mm512_cmpgt_epi64_mask(used, reach))) {
          const __m512i top = _mm512_sllv_epi64(one, nb);
          const __m512i x = _mm512_maskz_or_epi64(nz, top, _mm512_and_si512(_mm512_srlv_epi64(v, adv), _mm512_sub_epi64(top, one)));
          _mm512_storeu_si512(p, _mm512_xor_si512(_mm512_srli_epi64(x, 1),
            _mm512_sub_epi64(zeros, _mm512_and_si512(x, one))));
          bit = _mm512_add_epi64(bit, used);
          p2 = _mm512_mask_mov_epi64(p2, nz, p1);
          p1 = _mm512_mask_mov_epi64(p1, nz, b);
        }
        else {
          uint64_t bits[8], a1[8], a2[8];
          int l;
          _mm512_storeu_si512(bits, bit);
          _mm512_storeu_si512(a1, p1);
          _mm512_storeu_si512(a2, p2);
          for (l = 0; l < 8; ++l) {
            size_t b1 = (size_t)bits[l];
            p[l] = rice_value(in, &b1, a1 + l, a2 + l);
            bits[l] = b1;
          }
          bit = _mm512_loadu_si512(bits);
          p1 = _mm512_loadu_si512(a1);
          p2 = _mm512_loadu_si512(a2);
        }
      }
    }
    g += run;
    zero = !zero;
  }
}
#endif


/* Encodes, decodes, and checks the prototype, per record if given, in a reused buffer as vpack_bits. */
static double rice_bits(const int64_t* q, size_t n, int w, double* seconds, double* useconds, int* ok)
{
  const size_t r = (0 != record ? LIBXS_MIN(record, n) : n), nrec = (n + r - 1) / LIBXS_MAX(r, 1);
  const size_t nbytes = 8 * (2 * n + 8 * nrec * ((size_t)w + 1) + 8);
  const size_t cap = 2 * (r / (size_t)w + 2) + 8, ntmp = 8 * cap * ((size_t)w + 1);
  unsigned char *const packed = (unsigned char*)malloc(nbytes);
  uint64_t *const tmp = (uint64_t*)malloc(ntmp);
  size_t *const offs = (size_t*)malloc((nrec + 1) * ((size_t)w + 1) * sizeof(size_t));
  int64_t *const back = (int64_t*)malloc((n + (size_t)w) * sizeof(int64_t));
  int64_t *const buf = (int64_t*)malloc((r + (size_t)w) * sizeof(int64_t));
  double bits = 0;
  *seconds = *useconds = 0;
  if (NULL != packed && NULL != tmp && NULL != offs && NULL != back && NULL != buf && 0 != n) {
#if defined(LIBXS_INTRINSICS_AVX512)
    const int avx512 = (LIBXS_X86_AVX512 <= libxs_cpuid(NULL) ? 1 : 0);
#endif
#if defined(LIBXS_INTRINSICS_AVX2)
    const int avx2 = (LIBXS_X86_AVX2 <= libxs_cpuid(NULL) ? 1 : 0);
#endif
    double best = DBL_MAX, ubest = DBL_MAX;
    size_t irec, size = 0;
    int rep, l;
    first_touch(packed, nbytes);
    first_touch(tmp, ntmp);
    first_touch(back, (n + (size_t)w) * sizeof(int64_t));
    first_touch(buf, (r + (size_t)w) * sizeof(int64_t));
    for (rep = 0; rep < nrepeat; ++rep) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      double d;
      for (irec = 0, size = 0; irec < nrec; ++irec) {
        const size_t off = irec * r;
        size_t *const of = offs + irec * ((size_t)w + 1);
        const size_t len = rice_encode(packed + size, q + off, LIBXS_MIN(r, n - off), w, of, tmp, cap);
        for (l = 0; l <= w; ++l) of[l] += size;
        size += len;
      }
      d = libxs_timer_duration(t0, libxs_timer_tick());
      best = LIBXS_MIN(best, d);
    }
    *seconds = best;
    for (rep = 0; rep < nrepeat; ++rep) {
      double d = 0;
      for (irec = 0; irec < nrec; ++irec) {
        const size_t off = irec * r, m = LIBXS_MIN(r, n - off);
        const size_t *const of = offs + irec * ((size_t)w + 1);
        int64_t *const dst = (0 != record ? buf : back);
        const libxs_timer_tick_t t1 = libxs_timer_tick();
#if defined(LIBXS_INTRINSICS_AVX512)
        if (8 == w && 0 != avx512) rice_decode_avx512(dst, m, packed, of);
        else
#endif
#if defined(LIBXS_INTRINSICS_AVX2)
        if (0 != avx2) rice_decode_avx2(dst, m, packed, w, of);
        else
#endif
        rice_decode(dst, m, packed, w, of);
        d += libxs_timer_duration(t1, libxs_timer_tick());
        if (0 != memcmp(dst, q + off, m * sizeof(int64_t))) *ok = 0;
      }
      ubest = LIBXS_MIN(ubest, d);
    }
    *useconds = ubest;
    bits = 8.0 * (double)size + (0 != record ? (64.0 + 16.0 * (w + 1)) * (double)nrec : 0.0);
  }
  else if (0 != n) *ok = 0;
  free(packed);
  free(tmp);
  free(offs);
  free(back);
  free(buf);
  return bits;
}


/* As rice_bits over the streams per width of vpack_bits. */
static double rice_bits_streams(const stream_t* s, int w, double* seconds, double* useconds, int* ok)
{
  double bits = 0;
  int nbits;
  *seconds = *useconds = 0;
  for (nbits = 0; nbits <= MAXBITS; ++nbits) {
    if (0 != s[nbits].n) {
      double ds, du;
      bits += rice_bits(s[nbits].q, s[nbits].n, w, &ds, &du, ok);
      *seconds += ds;
      *useconds += du;
    }
  }
  return bits;
}


static int run_files(int nfiles, char* files[], double eps)
{
  stats_t st;
  int result = EXIT_SUCCESS, i;
  const char *const env_record = getenv("RECORD"), *const env_kdleaf = getenv("KDLEAF");
  const char *const env_kdexc = getenv("KDEXC");
  const char *const env_repeat = getenv("REPEAT");
  memset(&st, 0, sizeof(st));
  record = (NULL != env_record ? (size_t)LIBXS_MAX(atoi(env_record), 0) : 0);
  nrepeat = (NULL != env_repeat ? LIBXS_MAX(atoi(env_repeat), 1) : 1);
  kd_minleaf = (NULL != env_kdleaf ? (size_t)LIBXS_MAX(atoi(env_kdleaf), 0) : 0);
  kd_exc = (NULL != env_kdexc ? atoi(env_kdexc) : 1);
  for (i = 0; i < nfiles && EXIT_SUCCESS == result; ++i) result = add_file(&st, files[i], eps);
  if (EXIT_SUCCESS == result && 0 != st.nvalues) {
    const double hdr = (double)st.nblocks * WIDTH_HEADER, n = (double)st.nvalues;
    header_t hd[NGROUPS];
    double seconds, useconds, bits;
    int ok = 1, ig;
    printf("files %i  blocks %lu  values %lu  zeros %.1f%%  eps %g  skipped %lu  record %lu\n", nfiles,
      (unsigned long)st.nblocks, (unsigned long)st.nvalues, 100.0 * st.nzeros / n, st.eps,
      (unsigned long)st.nskipped, (unsigned long)record);
    printf("%-26s %10s %8s %14s %14s\n", "variant", "bits/value", "ratio", "pack ns/value", "unpack ns/value");
    fixed_seconds(st.plain, &seconds, &useconds, &ok);
    printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", "fixed", st.bits_fixed / n, 1.0,
      1E9 * seconds / n, 1E9 * useconds / n);
    for (ig = 0; ig < NGROUPS; ++ig) {
      char label[32];
      memset(&hd[ig], 0, sizeof(hd[ig]));
      bits = vpack_bits(st.plain, groups[ig], libxs_vpack, libxs_vunpack, &seconds, &useconds, &ok,
        &hd[ig]) + hdr;
      LIBXS_SNPRINTF(label, sizeof(label), "vpack group %i", groups[ig]);
      printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", label, bits / n, st.bits_fixed / bits,
        1E9 * seconds / n, 1E9 * useconds / n);
    }
#if defined(PACK_FORTRAN)
    bits = vpack_bits(st.plain, 8, vpack_fortran, vunpack_fortran, &seconds, &useconds, &ok, NULL) + hdr;
    printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", "fortran group 8", bits / n, st.bits_fixed / bits,
      1E9 * seconds / n, 1E9 * useconds / n);
# if defined(LIBXS_PACK_SW) /* same format: each decodes what the other packed */
    vpack_bits(st.plain, 8, vpack_fortran, libxs_vunpack, &seconds, &useconds, &ok, NULL);
    vpack_bits(st.plain, 8, libxs_vpack, vunpack_fortran, &seconds, &useconds, &ok, NULL);
# endif
#endif
    if (0 != st.shapes.n) { /* values of blocks without a shape are not covered */
      const double nb = (double)st.blocks.n;
      bits = kd_rows_bits(&st, &seconds, &useconds, &ok) + (double)(st.shapes.n / 3) * WIDTH_HEADER;
      printf("%-26s %10.2f %8.2f %14.2f %14.2f   (%.1f values per leaf, %.1f per run)\n",
        "kd-tree rows, exceptions", bits / nb, st.bits_fixed * (nb / n) / bits, 1E9 * seconds / nb,
        1E9 * useconds / nb, nb / LIBXS_MAX(1, kd_leaves), nb / LIBXS_MAX(1, kd_runs));
      for (i = 4; i <= RICE_MAXLANES; i *= 2) {
        char label[48];
        bits = rice_bits(st.blocks.q, st.blocks.n, i, &seconds, &useconds, &ok);
        LIBXS_SNPRINTF(label, sizeof(label), "rice lengths, %i lanes", i);
        printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", label, bits / nb,
          st.bits_fixed * (nb / n) / bits, 1E9 * seconds / nb, 1E9 * useconds / nb);
      }
      for (i = 4; i <= RICE_MAXLANES; i *= 2) {
        char label[48];
        bits = rice_bits_streams(st.plain, i, &seconds, &useconds, &ok) + hdr;
        LIBXS_SNPRINTF(label, sizeof(label), "rice per width, %i lanes", i);
        printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", label, bits / n, st.bits_fixed / bits,
          1E9 * seconds / n, 1E9 * useconds / n);
      }
    }
    bits = vpack_bits(st.delta, LIBXS_VPACK_GROUP, libxs_vpack, libxs_vunpack, &seconds, &useconds,
      &ok, NULL) + hdr;
    printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", "vpack deltas", bits / n, st.bits_fixed / bits,
      1E9 * seconds / n, 1E9 * useconds / n);
    if (0 != st.has_twice) {
      printf("%-26s %10.2f %8.2f\n", "fixed, step 2*eps", st.bits_twice / n, st.bits_fixed / st.bits_twice);
      bits = vpack_bits(st.twice, LIBXS_VPACK_GROUP, libxs_vpack, libxs_vunpack, &seconds,
        &useconds, &ok, NULL) + hdr;
      printf("%-26s %10.2f %8.2f %14.2f %14.2f\n", "vpack, step 2*eps", bits / n, st.bits_fixed / bits,
        1E9 * seconds / n, 1E9 * useconds / n);
    }
    printf("%-26s %10.2f %8.2f\n", "needed per value (bound)", (st.bits_needed + hdr) / n,
      st.bits_fixed / (st.bits_needed + hdr));
    printf("\n%-8s %9s %26s\n", "group", "payload", "+ header: bytes 7-bit base+3 base+2 entropy");
    for (ig = 0; ig < NGROUPS; ++ig) {
      const double nw = LIBXS_MAX(1.0, hd[ig].nwidths), pay = hd[ig].payload + hdr;
      double entropy = 0;
      int w;
      for (w = 0; w <= MAXBITS; ++w) {
        if (0 < hd[ig].hist[w]) entropy -= hd[ig].hist[w] * log2(hd[ig].hist[w] / nw);
      }
      printf("%-8i %9.2f %10.2f %6.2f %6.2f %6.2f %7.2f   bits/value\n", groups[ig], pay / n,
        (pay + hd[ig].byte) / n, (pay + hd[ig].seven) / n, (pay + hd[ig].base3) / n,
        (pay + hd[ig].base2) / n, (pay + entropy) / n);
      printf("%-8s %9s %10.2f %6.2f %6.2f %6.2f %7.2f   ratio\n", "", "",
        st.bits_fixed / (pay + hd[ig].byte), st.bits_fixed / (pay + hd[ig].seven),
        st.bits_fixed / (pay + hd[ig].base3), st.bits_fixed / (pay + hd[ig].base2),
        st.bits_fixed / (pay + entropy));
    }
    { /* cost models of a group of 8 (estimates, no format) */
      static const char* labels[NMODELS] = { "group 8", "zero mask", "exceptions", "mask+exceptions" };
      double model[NMODELS], exc[4];
      int k;
      stream_costs(st.plain, model, exc);
      printf("\n%-26s %10s %8s\n", "model (group of 8)", "bits/value", "ratio");
      for (k = 0; k < NMODELS; ++k) {
        printf("%-26s %10.2f %8.2f\n", labels[k], (model[k] + hdr) / n, st.bits_fixed / (model[k] + hdr));
      }
      stream_costs(st.delta, model, exc);
      for (k = 0; k < NMODELS; ++k) {
        char label[32];
        LIBXS_SNPRINTF(label, sizeof(label), "deltas, %s", labels[k]);
        printf("%-26s %10.2f %8.2f\n", label, (model[k] + hdr) / n, st.bits_fixed / (model[k] + hdr));
      }
      stream_costs(st.plain, model, exc);
      printf("exceptions: %.1f%% of groups, %.2f%% of values (%.2f per such group), high parts %.1f bits\n",
        100.0 * exc[1] / LIBXS_MAX(1, exc[0]), 100.0 * exc[2] / n, exc[2] / LIBXS_MAX(1, exc[1]),
        exc[3] / LIBXS_MAX(1, exc[1]));
      printf("%-26s %10.2f %8.2f\n", "per block, as is", (st.bits_axis[0] + hdr) / n,
        st.bits_fixed / (st.bits_axis[0] + hdr));
      for (k = 1; k <= NAXES; ++k) {
        char label[32];
        LIBXS_SNPRINTF(label, sizeof(label), "per block, delta axis %i", k);
        printf("%-26s %10.2f %8.2f\n", label, (st.bits_axis[k] + hdr) / n,
          st.bits_fixed / (st.bits_axis[k] + hdr));
      }
      printf("%-26s %10.2f %8.2f\n", "per block, best of these", (st.bits_axis_best + hdr) / n,
        st.bits_fixed / (st.bits_axis_best + hdr));
      { static const char* kdlabels[NKD] = { "kd-tree, all axes", "kd-tree, all axes, exc.",
          "kd-tree, rows, exc.", "kd-tree, ranges, exc." };
        for (k = 0; k < NKD; ++k) {
          printf("%-26s %10.2f %8.2f   (%.1f values per box)\n", kdlabels[k], (st.bits_kd[k] + hdr) / n,
            st.bits_fixed / (st.bits_kd[k] + hdr), n / LIBXS_MAX(1, st.nleaves_kd[k]));
        }
      }
      { /* traversal orders of the axes (innermost first) */
        static const char* olabels[NORDERS] = { "012", "021", "102", "120", "201", "210" };
        for (k = 0; k < NORDERS; ++k) {
          char label[32];
          LIBXS_SNPRINTF(label, sizeof(label), "order %s, exceptions", olabels[k]);
          printf("%-26s %10.2f %8.2f   (without: %.2f)\n", label, (st.bits_order[k][1] + hdr) / n,
            st.bits_fixed / (st.bits_order[k][1] + hdr), st.bits_fixed / (st.bits_order[k][0] + hdr));
        }
        printf("%-26s %10.2f %8.2f   (without: %.2f)\n", "order per block, exc.", (st.bits_order_best[1] + hdr) / n,
          st.bits_fixed / (st.bits_order_best[1] + hdr), st.bits_fixed / (st.bits_order_best[0] + hdr));
      }
      { /* dyadic trees in stream order */
        double tree[NTREES];
        tree_costs(st.plain, tree);
        printf("%-26s %10.2f %8.2f\n", "tree 4..64", (tree[0] + hdr) / n, st.bits_fixed / (tree[0] + hdr));
        printf("%-26s %10.2f %8.2f\n", "tree 4..64, exceptions", (tree[1] + hdr) / n, st.bits_fixed / (tree[1] + hdr));
      }
    }
    if (0 == ok) {
      fprintf(stderr, "ERROR: a round trip failed\n");
      result = EXIT_FAILURE;
    }
  }
  free(st.blocks.q);
  free(st.shapes.q);
  for (i = 0; i <= MAXBITS; ++i) {
    free(st.plain[i].q);
    free(st.delta[i].q);
    free(st.twice[i].q);
  }
  return result;
}


/* Signed values of up to nbits bits: per block, magnitudes log-uniform over range bits. */
static void synthesize(int64_t* q, size_t n, int nbits, int range, int zeros)
{
  size_t i;
  for (i = 0; i < n; ++i) {
    const int top = nbits - 1 - (int)((i / BLOCK) % (uint64_t)LIBXS_MAX(1, nbits / 4));
    const int drop = (int)(rng_next() % (uint64_t)LIBXS_MAX(1, range)); /* LIBXS_MAX evaluates twice */
    const int e = LIBXS_MAX(0, top - drop);
    const int64_t v = (int64_t)(rng_next() & ((((uint64_t)1) << e) - 1));
    q[i] = ((int)(rng_next() % 100) < zeros) ? 0 : (0 != (1 & rng_next()) ? -v : v);
  }
}


static int run_synthetic(size_t n, int nbits, int range, int zeros)
{
  const size_t nfixed = (n * (size_t)nbits + 63) / 64;
  const size_t bound = libxs_vpack_bound(n, nbits, 1);
  int64_t *const q = (int64_t*)malloc(n * sizeof(int64_t));
  int64_t *const back = (int64_t*)malloc(n * sizeof(int64_t));
  uint64_t *const packed = (uint64_t*)malloc(LIBXS_MAX(bound, nfixed + 1) * sizeof(uint64_t));
  int result = EXIT_FAILURE;
  if (NULL != q && NULL != back && NULL != packed) {
    const uint64_t offset = ((uint64_t)1) << (nbits - 1);
    double dpack = DBL_MAX, dunpack = DBL_MAX;
    size_t i, nwords = 0;
    int r, ig;
    result = EXIT_SUCCESS;
    synthesize(q, n, nbits, range, zeros);
    printf("values %lu  nbits %i  range %i  zeros %i%%\n", (unsigned long)n, nbits, range, zeros);
    printf("%-14s %10s %8s %14s %14s\n", "variant", "bits/value", "ratio", "pack ns/value", "unpack ns/value");
    for (i = 0; i < n; ++i) back[i] = (int64_t)((uint64_t)q[i] + offset); /* CP2K: offset binary */
    for (r = 0; r < nrepeat; ++r) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      double d;
      libxs_pack(packed, back, LIBXS_DATATYPE_U64, n, nbits);
      d = libxs_timer_duration(t0, libxs_timer_tick());
      dpack = LIBXS_MIN(dpack, d);
    }
    for (r = 0; r < nrepeat; ++r) {
      const libxs_timer_tick_t t0 = libxs_timer_tick();
      double d;
      libxs_unpack(back, packed, LIBXS_DATATYPE_U64, n, nbits);
      d = libxs_timer_duration(t0, libxs_timer_tick());
      dunpack = LIBXS_MIN(dunpack, d);
    }
    for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
      if ((uint64_t)back[i] != (uint64_t)q[i] + offset) result = EXIT_FAILURE;
    }
    printf("%-14s %10.2f %8.2f %14.2f %14.2f\n", "fixed", (double)nbits, 1.0,
      1E9 * dpack / n, 1E9 * dunpack / n);
    for (ig = 0; ig < NGROUPS && EXIT_SUCCESS == result; ++ig) {
      char label[32];
      dpack = dunpack = DBL_MAX;
      for (r = 0; r < nrepeat; ++r) {
        const libxs_timer_tick_t t0 = libxs_timer_tick();
        double d;
        nwords = libxs_vpack(packed, q, LIBXS_DATATYPE_I64, n, nbits, groups[ig]);
        d = libxs_timer_duration(t0, libxs_timer_tick());
        dpack = LIBXS_MIN(dpack, d);
      }
      for (r = 0; r < nrepeat; ++r) {
        const libxs_timer_tick_t t0 = libxs_timer_tick();
        double d;
        if (nwords != libxs_vunpack(back, packed, nwords, LIBXS_DATATYPE_I64, n)) result = EXIT_FAILURE;
        d = libxs_timer_duration(t0, libxs_timer_tick());
        dunpack = LIBXS_MIN(dunpack, d);
      }
      if (0 != memcmp(back, q, n * sizeof(int64_t))) result = EXIT_FAILURE;
      LIBXS_SNPRINTF(label, sizeof(label), "vpack %i", groups[ig]);
      printf("%-14s %10.2f %8.2f %14.2f %14.2f\n", label, 64.0 * nwords / n, (double)nfixed / nwords,
        1E9 * dpack / n, 1E9 * dunpack / n);
    }
    if (EXIT_SUCCESS == result) {
      int ok = 1, w;
      for (w = 4; w <= RICE_MAXLANES; w *= 2) {
        char label[48];
        const double bits = rice_bits(q, n, w, &dpack, &dunpack, &ok);
        LIBXS_SNPRINTF(label, sizeof(label), "rice %i lanes", w);
        printf("%-14s %10.2f %8.2f %14.2f %14.2f\n", label, bits / n, (double)nbits * n / bits,
          1E9 * dpack / n, 1E9 * dunpack / n);
      }
      if (0 == ok) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result) fprintf(stderr, "ERROR: a round trip failed\n");
  }
  free(q);
  free(back);
  free(packed);
  return result;
}


int main(int argc, char* argv[])
{
  FILE *const file = (1 < argc ? fopen(argv[1], "rb") : NULL);
  int result;
  libxs_init();
  if (NULL != file) { /* MHD files, optionally followed by eps */
    double eps = 0;
    int nfiles = argc - 1;
    fclose(file);
    if (2 < argc && 0 < atof(argv[argc - 1])) {
      FILE *const last = fopen(argv[argc - 1], "rb");
      if (NULL == last) {
        eps = atof(argv[argc - 1]);
        --nfiles;
      }
      else fclose(last);
    }
    result = run_files(nfiles, argv + 1, eps);
  }
  else {
    const size_t n = (size_t)LIBXS_MAX(1, 1 < argc ? atoi(argv[1]) : (1 << 20));
    const int nbits = LIBXS_CLMP(2 < argc ? atoi(argv[2]) : 24, 2, 63);
    const int range = LIBXS_CLMP(3 < argc ? atoi(argv[3]) : 12, 1, nbits);
    const int zeros = LIBXS_CLMP(4 < argc ? atoi(argv[4]) : 10, 0, 100);
    nrepeat = LIBXS_MAX(1, 5 < argc ? atoi(argv[5]) : 10);
    result = run_synthetic(n, nbits, range, zeros);
  }
  libxs_finalize();
  return result;
}
