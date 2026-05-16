/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_perm.h>
#include <libxs_mem.h>
#include <libxs_math.h>
#include <libxs_malloc.h>
#include "libxs_main.h"

#define LIBXS_MEM_SHUFFLE_COPRIME(N) libxs_coprime2(N)
#define LIBXS_MEM_SHUFFLE_MALLOC(SIZE, POOL) \
  internal_libxs_perm_shuffle_malloc(SIZE, &(POOL))
#define LIBXS_MEM_SHUFFLE_FREE(PTR, POOL) \
  internal_libxs_perm_shuffle_free(PTR, POOL)
#define LIBXS_MEM_SHUFFLE(INOUT, ELEMSIZE, COUNT, SHUFFLE, NREPEAT) do { \
  unsigned char *const LIBXS_RESTRICT shfl_data = (unsigned char*)(INOUT); \
  const size_t shfl_count = (COUNT), shfl_stride = (SHUFFLE); \
  const size_t shfl_last = shfl_count - 1, shfl_nrep = (NREPEAT); \
  const size_t shfl_nbitmask = (shfl_count + 7) / 8; \
  int shfl_pool_v = 0, shfl_pool_t = 0; \
  unsigned char *const shfl_visited = (unsigned char*) \
    LIBXS_MEM_SHUFFLE_MALLOC(shfl_nbitmask, shfl_pool_v); \
  if (NULL != shfl_visited && 0 != shfl_nrep) { \
    void *const shfl_tmp = LIBXS_MEM_SHUFFLE_MALLOC(ELEMSIZE, shfl_pool_t); \
    memset(shfl_visited, 0, shfl_nbitmask); \
    if (NULL != shfl_tmp) { \
      size_t shfl_i; \
      for (shfl_i = 0; shfl_i < shfl_count; ++shfl_i) { \
        size_t shfl_src, shfl_dst, shfl_k; \
        if (0 != (shfl_visited[shfl_i / 8] & (1u << (shfl_i % 8)))) continue; \
        shfl_src = shfl_i; \
        for (shfl_k = 0; shfl_k < shfl_nrep; ++shfl_k) { \
          shfl_src = shfl_last - ((shfl_stride * shfl_src) % shfl_count); \
        } \
        if (shfl_src == shfl_i) { \
          shfl_visited[shfl_i / 8] |= (unsigned char)(1u << (shfl_i % 8)); \
          continue; \
        } \
        LIBXS_MEMCPY(shfl_tmp, shfl_data + (ELEMSIZE) * shfl_i, ELEMSIZE); \
        shfl_dst = shfl_i; \
        do { \
          shfl_src = shfl_dst; \
          for (shfl_k = 0; shfl_k < shfl_nrep; ++shfl_k) { \
            shfl_src = shfl_last - ((shfl_stride * shfl_src) % shfl_count); \
          } \
          shfl_visited[shfl_dst / 8] |= (unsigned char)(1u << (shfl_dst % 8)); \
          if (shfl_src != shfl_i) { \
            LIBXS_MEMCPY(shfl_data + (ELEMSIZE) * shfl_dst, \
              shfl_data + (ELEMSIZE) * shfl_src, ELEMSIZE); \
          } \
          else { \
            LIBXS_MEMCPY(shfl_data + (ELEMSIZE) * shfl_dst, shfl_tmp, ELEMSIZE); \
          } \
          shfl_dst = shfl_src; \
        } while (shfl_src != shfl_i); \
      } \
      LIBXS_MEM_SHUFFLE_FREE(shfl_tmp, shfl_pool_t); \
    } \
    LIBXS_MEM_SHUFFLE_FREE(shfl_visited, shfl_pool_v); \
  } \
} while(0)


LIBXS_API_INLINE void* internal_libxs_perm_shuffle_malloc(size_t size, int* pool) {
  void* p = libxs_malloc(internal_libxs_default_pool, size, LIBXS_MALLOC_AUTO);
  if (NULL != p) { *pool = 1; return p; }
  *pool = 0; return malloc(size);
}

LIBXS_API_INLINE void internal_libxs_perm_shuffle_free(void* ptr, int pool) {
  if (0 != pool) libxs_free(ptr); else free(ptr);
}


LIBXS_API_INLINE void* internal_libxs_sort_malloc(size_t size, int* pool) {
  void* p = libxs_malloc(internal_libxs_default_pool, size, LIBXS_MALLOC_AUTO);
  if (NULL != p) { *pool = 1; return p; }
  *pool = 0; return malloc(size);
}


LIBXS_API_INLINE void internal_libxs_sort_free(void* ptr, int pool) {
  if (0 != pool) libxs_free(ptr); else free(ptr);
}


LIBXS_API_INLINE void internal_libxs_sort_swap(
  unsigned char* LIBXS_RESTRICT a, unsigned char* LIBXS_RESTRICT b, size_t size)
{
  size_t i;
  for (i = 0; i < size; ++i) {
    const unsigned char t = a[i];
    a[i] = b[i]; b[i] = t;
  }
}


LIBXS_API_INLINE void internal_libxs_sort_heap(
  void* base, int n, size_t size, libxs_sort_cmp_t cmp, void* ctx)
{
  unsigned char* const data = (unsigned char*)base;
  int i, end;
  for (i = n / 2 - 1; 0 <= i; --i) {
    int parent = i, child;
    while ((child = 2 * parent + 1) < n) {
      if (child + 1 < n && cmp(data + (size_t)child * size,
        data + (size_t)(child + 1) * size, ctx) < 0)
      {
        ++child;
      }
      if (cmp(data + (size_t)parent * size,
        data + (size_t)child * size, ctx) < 0)
      {
        internal_libxs_sort_swap(
          data + (size_t)parent * size,
          data + (size_t)child * size, size);
        parent = child;
      }
      else break;
    }
  }
  for (end = n - 1; 0 < end; --end) {
    int parent = 0, child;
    internal_libxs_sort_swap(data, data + (size_t)end * size, size);
    while ((child = 2 * parent + 1) < end) {
      if (child + 1 < end && cmp(data + (size_t)child * size,
        data + (size_t)(child + 1) * size, ctx) < 0)
      {
        ++child;
      }
      if (cmp(data + (size_t)parent * size,
        data + (size_t)child * size, ctx) < 0)
      {
        internal_libxs_sort_swap(
          data + (size_t)parent * size,
          data + (size_t)child * size, size);
        parent = child;
      }
      else break;
    }
  }
}


LIBXS_API int libxs_cmp_f64(const void* a, const void* b, void* ctx) {
  const double va = *(const double*)a, vb = *(const double*)b;
  (void)ctx;
  return (va > vb) - (va < vb);
}


LIBXS_API int libxs_cmp_f32(const void* a, const void* b, void* ctx) {
  const float va = *(const float*)a, vb = *(const float*)b;
  (void)ctx;
  return (va > vb) - (va < vb);
}


LIBXS_API int libxs_cmp_i32(const void* a, const void* b, void* ctx) {
  const int va = *(const int*)a, vb = *(const int*)b;
  (void)ctx;
  return (va > vb) - (va < vb);
}


LIBXS_API int libxs_cmp_u32(const void* a, const void* b, void* ctx) {
  const unsigned int va = *(const unsigned int*)a;
  const unsigned int vb = *(const unsigned int*)b;
  (void)ctx;
  return (va > vb) - (va < vb);
}


LIBXS_API_INLINE void internal_libxs_radix_f64(
  unsigned long long* LIBXS_RESTRICT dst,
  unsigned long long* LIBXS_RESTRICT src, int n)
{
  unsigned long long* LIBXS_RESTRICT a = src;
  unsigned long long* LIBXS_RESTRICT b = dst;
  int pass;
  for (pass = 0; pass < 8; ++pass) {
    const int shift = pass * 8;
    int count[256], i;
    memset(count, 0, sizeof(count));
    for (i = 0; i < n; ++i) {
      ++count[(a[i] >> shift) & 0xFF];
    }
    { int sum = 0, j;
      for (j = 0; j < 256; ++j) {
        const int c = count[j];
        count[j] = sum; sum += c;
      }
    }
    for (i = 0; i < n; ++i) {
      b[count[(a[i] >> shift) & 0xFF]++] = a[i];
    }
    { unsigned long long* t = a; a = b; b = t; }
  }
}


LIBXS_API_INLINE void internal_libxs_radix_u32(
  unsigned int* LIBXS_RESTRICT dst,
  unsigned int* LIBXS_RESTRICT src, int n)
{
  unsigned int* LIBXS_RESTRICT a = src;
  unsigned int* LIBXS_RESTRICT b = dst;
  int pass;
  for (pass = 0; pass < 4; ++pass) {
    const int shift = pass * 8;
    int count[256], i;
    memset(count, 0, sizeof(count));
    for (i = 0; i < n; ++i) {
      ++count[(a[i] >> shift) & 0xFF];
    }
    { int sum = 0, j;
      for (j = 0; j < 256; ++j) {
        const int c = count[j];
        count[j] = sum; sum += c;
      }
    }
    for (i = 0; i < n; ++i) {
      b[count[(a[i] >> shift) & 0xFF]++] = a[i];
    }
    { unsigned int* t = a; a = b; b = t; }
  }
}


LIBXS_API_INLINE void internal_libxs_sort_radix_f64(
  double* LIBXS_RESTRICT dst, const double* src, int n, void* scratch)
{
  unsigned long long* const keys = (unsigned long long*)dst;
  unsigned long long* const aux = (unsigned long long*)scratch;
  int i;
  for (i = 0; i < n; ++i) {
    unsigned long long bits;
    memcpy(&bits, src + i, 8);
    keys[i] = (bits >> 63) ? ~bits : (bits | 0x8000000000000000ULL);
  }
  internal_libxs_radix_f64(aux, keys, n);
  for (i = 0; i < n; ++i) {
    unsigned long long bits = keys[i];
    bits = (bits >> 63) ? (bits ^ 0x8000000000000000ULL) : ~bits;
    memcpy(dst + i, &bits, 8);
  }
}


LIBXS_API_INLINE void internal_libxs_sort_radix_f32(
  float* LIBXS_RESTRICT dst, const float* src, int n, void* scratch)
{
  unsigned int* const keys = (unsigned int*)dst;
  unsigned int* const aux = (unsigned int*)scratch;
  int i;
  for (i = 0; i < n; ++i) {
    unsigned int bits;
    memcpy(&bits, src + i, 4);
    keys[i] = (bits >> 31) ? ~bits : (bits | 0x80000000U);
  }
  internal_libxs_radix_u32(aux, keys, n);
  for (i = 0; i < n; ++i) {
    unsigned int bits = keys[i];
    bits = (bits >> 31) ? (bits ^ 0x80000000U) : ~bits;
    memcpy(dst + i, &bits, 4);
  }
}


LIBXS_API_INLINE void internal_libxs_sort_radix_i32(
  int* LIBXS_RESTRICT dst, const int* src, int n, void* scratch)
{
  unsigned int* const keys = (unsigned int*)dst;
  unsigned int* const aux = (unsigned int*)scratch;
  int i;
  for (i = 0; i < n; ++i) {
    keys[i] = (unsigned int)src[i] ^ 0x80000000U;
  }
  internal_libxs_radix_u32(aux, keys, n);
  for (i = 0; i < n; ++i) {
    dst[i] = (int)(keys[i] ^ 0x80000000U);
  }
}


LIBXS_API_INLINE void internal_libxs_sort_radix_u32(
  unsigned int* LIBXS_RESTRICT dst, const unsigned int* src, int n,
  void* scratch)
{
  unsigned int* const aux = (unsigned int*)scratch;
  memcpy(dst, src, (size_t)n * 4);
  internal_libxs_radix_u32(aux, dst, n);
}


LIBXS_API void libxs_sort(void* base, int n, size_t size,
  libxs_sort_cmp_t cmp, void* ctx)
{
  if (NULL == base || n < 2 || 0 == size || NULL == cmp) return;
  if (cmp == libxs_cmp_f64 || cmp == libxs_cmp_f32
    || cmp == libxs_cmp_i32 || cmp == libxs_cmp_u32)
  {
    const void* src = (NULL != ctx) ? ctx : base;
    int pool = 0;
    void* scratch = internal_libxs_sort_malloc((size_t)n * size, &pool);
    if (NULL != scratch) {
      if (cmp == libxs_cmp_f64) {
        internal_libxs_sort_radix_f64(
          (double*)base, (const double*)src, n, scratch);
      }
      else if (cmp == libxs_cmp_f32) {
        internal_libxs_sort_radix_f32(
          (float*)base, (const float*)src, n, scratch);
      }
      else if (cmp == libxs_cmp_i32) {
        internal_libxs_sort_radix_i32(
          (int*)base, (const int*)src, n, scratch);
      }
      else {
        internal_libxs_sort_radix_u32(
          (unsigned int*)base, (const unsigned int*)src, n, scratch);
      }
      internal_libxs_sort_free(scratch, pool);
    }
    else {
      if (NULL != ctx) memcpy(base, ctx, (size_t)n * size);
      internal_libxs_sort_heap(base, n, size, cmp, NULL);
    }
  }
  else {
    internal_libxs_sort_heap(base, n, size, cmp, ctx);
  }
}


LIBXS_API uint64_t libxs_hilbert(const unsigned int coords[], int ndims)
{
  const int bpd = 64 / ndims;
  unsigned int x[64];
  uint64_t code = 0;
  int i, level;
  for (i = 0; i < ndims; ++i) x[i] = coords[i];
  /* Transpose: apply inverse Gray code and rotations (Skilling method) */
  { const unsigned int m = 1u << (bpd - 1);
    unsigned int q, p, t;
    for (q = m; 0 < q; q >>= 1) {
      p = q - 1;
      for (i = 0; i < ndims; ++i) {
        if (0 != (x[i] & q)) {
          x[0] ^= p;
        }
        else {
          t = (x[0] ^ x[i]) & p;
          x[0] ^= t; x[i] ^= t;
        }
      }
    }
    for (i = 1; i < ndims; ++i) x[i] ^= x[i - 1];
    t = 0;
    for (q = m; 0 < q; q >>= 1) {
      if (0 != (x[ndims - 1] & q)) t ^= q - 1;
    }
    for (i = 0; i < ndims; ++i) x[i] ^= t;
  }
  /* Interleave transposed bits into the code */
  for (level = bpd - 1; 0 <= level; --level) {
    for (i = ndims - 1; 0 <= i; --i) {
      code = (code << 1) | ((x[i] >> level) & 1u);
    }
  }
  return code;
}


LIBXS_API uint64_t libxs_morton(const unsigned int coords[], int ndims)
{
  const int bpd = 64 / ndims;
  uint64_t code = 0;
  int bit, d;
  for (bit = bpd - 1; 0 <= bit; --bit) {
    for (d = ndims - 1; 0 <= d; --d) {
      code = (code << 1) | ((coords[d] >> bit) & 1u);
    }
  }
  return code;
}


LIBXS_API_INLINE void internal_libxs_kdtree2d_build(
  const double* pts, int* idx, int lo, int hi, int depth)
{
  if (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    const int k = depth & 1;
    int l = lo, r = hi - 1;
    while (l < r) {
      const double pivot = pts[2 * idx[mid] + k];
      int ll = l, rr = r;
      while (ll <= rr) {
        while (pts[2 * idx[ll] + k] < pivot) ++ll;
        while (pts[2 * idx[rr] + k] > pivot) --rr;
        if (ll <= rr) {
          { const int t = idx[ll]; idx[ll] = idx[rr]; idx[rr] = t; }
          ++ll; --rr;
        }
      }
      if (mid <= rr) r = rr;
      else if (mid >= ll) l = ll;
      else break;
    }
    internal_libxs_kdtree2d_build(pts, idx, lo, mid, depth + 1);
    internal_libxs_kdtree2d_build(pts, idx, mid + 1, hi, depth + 1);
  }
}


LIBXS_API_INTERN void internal_libxs_kdtree2d_build_entry(const double* pts, int* idx, int n);
LIBXS_API_INTERN void internal_libxs_kdtree2d_build_entry(const double* pts, int* idx, int n)
{
  if (NULL != pts && NULL != idx && n > 1) {
    internal_libxs_kdtree2d_build(pts, idx, 0, n, 0);
  }
}


LIBXS_EXTERN_C typedef struct internal_libxs_kdtree2d_ctx_t {
  const double* pts;
  const int* idx;
  const unsigned char* used;
  double qx, qy;
} internal_libxs_kdtree2d_ctx_t;


LIBXS_API_INLINE int internal_libxs_kdtree2d_find(
  const internal_libxs_kdtree2d_ctx_t* ctx,
  int lo, int hi, int depth, double best_d2, int best_idx)
{
  if (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const int k = depth & 1;
    const int pi = ctx->idx[mid];
    double split, dist;
    int near_lo, near_hi, far_lo, far_hi;
    if (NULL == ctx->used || 0 == ctx->used[pi]) {
      const double dx = ctx->pts[2*pi] - ctx->qx;
      const double dy = ctx->pts[2*pi+1] - ctx->qy;
      const double d2 = dx * dx + dy * dy;
      if (d2 <= best_d2) { best_d2 = d2; best_idx = pi; }
    }
    split = (0 == k) ? ctx->pts[2*pi] : ctx->pts[2*pi+1];
    dist = (0 == k) ? (ctx->qx - split) : (ctx->qy - split);
    if (dist <= 0) {
      near_lo = lo; near_hi = mid; far_lo = mid + 1; far_hi = hi;
    }
    else {
      near_lo = mid + 1; near_hi = hi; far_lo = lo; far_hi = mid;
    }
    best_idx = internal_libxs_kdtree2d_find(
      ctx, near_lo, near_hi, depth + 1, best_d2, best_idx);
    if (best_idx >= 0) {
      const double dx = ctx->pts[2*best_idx] - ctx->qx;
      const double dy = ctx->pts[2*best_idx+1] - ctx->qy;
      best_d2 = dx * dx + dy * dy;
    }
    if (dist * dist < best_d2) {
      best_idx = internal_libxs_kdtree2d_find(
        ctx, far_lo, far_hi, depth + 1, best_d2, best_idx);
    }
  }
  return best_idx;
}


LIBXS_API_INTERN int internal_libxs_kdtree2d_nearest_entry(
  const double* pts, const int* idx, const unsigned char* used,
  int n, double x, double y, double max_dist2);
LIBXS_API_INTERN int internal_libxs_kdtree2d_nearest_entry(
  const double* pts, const int* idx, const unsigned char* used,
  int n, double x, double y, double max_dist2)
{
  int result = -1;
  if (NULL != pts && NULL != idx && 0 < n) {
    internal_libxs_kdtree2d_ctx_t ctx;
    ctx.pts = pts; ctx.idx = idx; ctx.used = used;
    ctx.qx = x; ctx.qy = y;
    result = internal_libxs_kdtree2d_find(&ctx, 0, n, 0, max_dist2, -1);
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_kdtree_build(
  const double* pts, int* idx, int lo, int hi, int ndims, int stride, int depth)
{
  if (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    const int k = depth % ndims;
    int l = lo, r = hi - 1;
    while (l < r) {
      const double pivot = pts[(size_t)idx[mid] * stride + k];
      int ll = l, rr = r;
      while (ll <= rr) {
        while (pts[(size_t)idx[ll] * stride + k] < pivot) ++ll;
        while (pts[(size_t)idx[rr] * stride + k] > pivot) --rr;
        if (ll <= rr) {
          { const int t = idx[ll]; idx[ll] = idx[rr]; idx[rr] = t; }
          ++ll; --rr;
        }
      }
      if (mid <= rr) r = rr;
      else if (mid >= ll) l = ll;
      else break;
    }
    internal_libxs_kdtree_build(pts, idx, lo, mid, ndims, stride, depth + 1);
    internal_libxs_kdtree_build(pts, idx, mid + 1, hi, ndims, stride, depth + 1);
  }
}


LIBXS_API void libxs_kdtree_build(
  const double* pts, int* idx, int n, int ndims, int stride)
{
  if (NULL != pts && NULL != idx && n > 1 && ndims > 0) {
    if (2 == ndims && 2 == stride) {
      internal_libxs_kdtree2d_build_entry(pts, idx, n);
    }
    else {
      internal_libxs_kdtree_build(pts, idx, 0, n, ndims, stride, 0);
    }
  }
}


LIBXS_EXTERN_C typedef struct internal_libxs_kdtree_ctx_t {
  const double* pts;
  const int* idx;
  const unsigned char* used;
  const double* query;
  int ndims, stride;
} internal_libxs_kdtree_ctx_t;


LIBXS_API_INLINE int internal_libxs_kdtree_find(
  const internal_libxs_kdtree_ctx_t* ctx,
  int lo, int hi, int depth, double best_d2, int best_idx)
{
  int result = best_idx;
  if (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const int k = depth % ctx->ndims;
    const int pi = ctx->idx[mid];
    if (NULL == ctx->used || 0 == ctx->used[pi]) {
      double d2 = 0;
      int dd;
      for (dd = 0; dd < ctx->ndims; ++dd) {
        const double d = ctx->pts[(size_t)pi * ctx->stride + dd] - ctx->query[dd];
        d2 += d * d;
      }
      if (d2 <= best_d2) { best_d2 = d2; result = pi; }
    }
    {
      const double split = ctx->pts[(size_t)pi * ctx->stride + k];
      const double dist = ctx->query[k] - split;
      int near_lo, near_hi, far_lo, far_hi, cand;
      if (dist <= 0) {
        near_lo = lo; near_hi = mid; far_lo = mid + 1; far_hi = hi;
      }
      else {
        near_lo = mid + 1; near_hi = hi; far_lo = lo; far_hi = mid;
      }
      cand = internal_libxs_kdtree_find(
        ctx, near_lo, near_hi, depth + 1, best_d2, result);
      if (cand >= 0) {
        double cd2 = 0;
        int dd;
        for (dd = 0; dd < ctx->ndims; ++dd) {
          const double d = ctx->pts[(size_t)cand * ctx->stride + dd] - ctx->query[dd];
          cd2 += d * d;
        }
        if (cd2 <= best_d2) { best_d2 = cd2; result = cand; }
      }
      if (dist * dist < best_d2) {
        cand = internal_libxs_kdtree_find(
          ctx, far_lo, far_hi, depth + 1, best_d2, result);
        if (cand >= 0) result = cand;
      }
    }
  }
  return result;
}


LIBXS_API int libxs_kdtree_nearest(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2)
{
  int result = -1;
  if (NULL != pts && NULL != idx && NULL != query && 0 < n && 0 < ndims) {
    if (2 == ndims && 2 == stride) {
      result = internal_libxs_kdtree2d_nearest_entry(
        pts, idx, used, n, query[0], query[1], max_dist2);
    }
    else {
      internal_libxs_kdtree_ctx_t ctx;
      ctx.pts = pts; ctx.idx = idx; ctx.used = used;
      ctx.query = query; ctx.ndims = ndims; ctx.stride = stride;
      result = internal_libxs_kdtree_find(&ctx, 0, n, 0, max_dist2, -1);
    }
  }
  return result;
}


#if defined(LIBXS_INTRINSICS_AVX2)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_shuffle2_u32_avx2(
  unsigned int* LIBXS_RESTRICT dst, const unsigned int* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count8 = count & ~(size_t)7;
  size_t i = 0, j = 1;
  for (; i < count8; i += 8) {
    LIBXS_ALIGNED(int idx[8], LIBXS_ALIGNMENT);
    __m256i vidx, vdata;
    int lane;
    for (lane = 0; lane < 8; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (int)(count - j);
      j += stride;
    }
    vidx = _mm256_load_si256((__m256i*)idx);
    vdata = _mm256_i32gather_epi32((const int*)src, vidx, 4);
    _mm256_storeu_si256((__m256i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX2)
void internal_libxs_shuffle2_u64_avx2(
  unsigned long long* LIBXS_RESTRICT dst, const unsigned long long* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count4 = count & ~(size_t)3;
  size_t i = 0, j = 1;
  for (; i < count4; i += 4) {
    LIBXS_ALIGNED(long long idx[4], LIBXS_ALIGNMENT);
    __m256i vidx, vdata;
    int lane;
    for (lane = 0; lane < 4; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (long long)(count - j);
      j += stride;
    }
    vidx = _mm256_load_si256((__m256i*)idx);
    vdata = _mm256_i64gather_epi64((const long long*)src, vidx, 8);
    _mm256_storeu_si256((__m256i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}
#endif /* LIBXS_INTRINSICS_AVX2 */


#if defined(LIBXS_INTRINSICS_AVX512)
LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_shuffle2_u32_avx512(
  unsigned int* LIBXS_RESTRICT dst, const unsigned int* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count16 = count & ~(size_t)15;
  size_t i = 0, j = 1;
  for (; i < count16; i += 16) {
    LIBXS_ALIGNED(int idx[16], LIBXS_ALIGNMENT);
    __m512i vidx, vdata;
    int lane;
    for (lane = 0; lane < 16; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (int)(count - j);
      j += stride;
    }
    vidx = _mm512_load_si512((__m512i*)idx);
    vdata = _mm512_i32gather_epi32(vidx, src, 4);
    _mm512_storeu_si512((__m512i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
void internal_libxs_shuffle2_u64_avx512(
  unsigned long long* LIBXS_RESTRICT dst, const unsigned long long* LIBXS_RESTRICT src,
  size_t count, size_t stride)
{
  const size_t count8 = count & ~(size_t)7;
  size_t i = 0, j = 1;
  for (; i < count8; i += 8) {
    LIBXS_ALIGNED(long long idx[8], LIBXS_ALIGNMENT);
    __m512i vidx, vdata;
    int lane;
    for (lane = 0; lane < 8; ++lane) {
      if (count < j) j -= count;
      idx[lane] = (long long)(count - j);
      j += stride;
    }
    vidx = _mm512_load_si512((__m512i*)idx);
    vdata = _mm512_i64gather_epi64(vidx, src, 8);
    _mm512_storeu_si512((__m512i*)(dst + i), vdata);
  }
  for (; i < count; ++i) {
    if (count < j) j -= count;
    dst[i] = src[count - j];
    j += stride;
  }
}
#endif /* LIBXS_INTRINSICS_AVX512 */


LIBXS_API_INLINE int internal_libxs_sort_smooth_cmp(
  const void* a, const void* b, void* ctx)
{
  const double* scores = (const double*)ctx;
  const double va = scores[*(const int*)a];
  const double vb = scores[*(const int*)b];
  return (va > vb) - (va < vb);
}


LIBXS_API int libxs_sort_smooth(libxs_sort_t method, int m, int n,
  const void* mat, int ld, libxs_data_t datatype, int* perm)
{
  int result = EXIT_SUCCESS;
  if (NULL == perm || NULL == mat || m < 0 || n < 0 || ld < m) {
    return EXIT_FAILURE;
  }
  if (0 == m || LIBXS_SORT_NONE == method) {
    return EXIT_SUCCESS;
  }
  {
    int pool = 0;
    const size_t scores_size = (size_t)m * sizeof(double);
    const size_t visited_size = (LIBXS_SORT_GREEDY == method) ? (size_t)m : 0;
    double* scores = NULL;
    char* visited = NULL;

    if (LIBXS_SORT_IDENTITY != method) {
      scores = (double*)internal_libxs_sort_malloc(
        scores_size + visited_size, &pool);
      if (NULL == scores) return EXIT_FAILURE;
      if (0 != visited_size) visited = (char*)(scores + m);
    }

    switch ((int)datatype) {
      case LIBXS_DATATYPE_F64: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE double
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_F32: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE float
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I32: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE int
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U32: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE unsigned int
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I16: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE short
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U16: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE unsigned short
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I8: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE signed char
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U8: {
#       define LIBXS_SORT_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_SORT_TEMPLATE_ELEM_TYPE unsigned char
#       include "libxs_sort.h"
#       undef LIBXS_SORT_TEMPLATE_ELEM_TYPE
#       undef LIBXS_SORT_TEMPLATE_TYPE2FP64
      } break;
      default: {
        static int error_once = 0;
        if (0 != libxs_verbosity
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1,
                    LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr,
            "LIBXS ERROR: unsupported data-type for sort!\n");
        }
        result = EXIT_FAILURE;
      }
    }

    internal_libxs_sort_free(scores, pool);
  }
  return result;
}


LIBXS_API int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  int result;
  if (0 == count || 0 == elemsize) {
    result = EXIT_SUCCESS;
  }
  else if (NULL != inout) {
    const size_t s = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    const size_t n = (NULL == nrepeat ? 1 : *nrepeat);
    switch (elemsize) {
      case 8:   LIBXS_MEM_SHUFFLE(inout, 8, count, s, n); break;
      case 4:   LIBXS_MEM_SHUFFLE(inout, 4, count, s, n); break;
      case 2:   LIBXS_MEM_SHUFFLE(inout, 2, count, s, n); break;
      case 1:   LIBXS_MEM_SHUFFLE(inout, 1, count, s, n); break;
      default:  LIBXS_MEM_SHUFFLE(inout, elemsize, count, s, n);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API int libxs_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  const unsigned char *const LIBXS_RESTRICT inp = (const unsigned char*)src;
  unsigned char *const LIBXS_RESTRICT out = (unsigned char*)dst;
  const size_t size = elemsize * count;
  int result;
  if ((NULL != inp && NULL != out && ((out + size) <= inp || (inp + size) <= out)) || 0 == size) {
    const size_t s = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    size_t i = 0, j = 1;
    if (NULL == nrepeat || 1 == *nrepeat) {
#if defined(LIBXS_INTRINSICS_AVX512) || defined(LIBXS_INTRINSICS_AVX2)
      if (4 == elemsize && count <= 0x7FFFFFFFU) {
# if defined(LIBXS_INTRINSICS_AVX512)
#   if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
        internal_libxs_shuffle2_u32_avx512((unsigned int*)out, (const unsigned int*)inp, count, s);
        i = size;
#   elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
        if (LIBXS_X86_AVX512 <= libxs_cpuid(NULL)) {
          internal_libxs_shuffle2_u32_avx512((unsigned int*)out, (const unsigned int*)inp, count, s);
          i = size;
        }
#   endif
# endif
# if defined(LIBXS_INTRINSICS_AVX2)
        if (i < size) {
#   if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
          internal_libxs_shuffle2_u32_avx2((unsigned int*)out, (const unsigned int*)inp, count, s);
          i = size;
#   elif (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
          if (LIBXS_X86_AVX2 <= libxs_cpuid(NULL)) {
            internal_libxs_shuffle2_u32_avx2((unsigned int*)out, (const unsigned int*)inp, count, s);
            i = size;
          }
#   endif
        }
# endif
      }
      else if (8 == elemsize) {
# if defined(LIBXS_INTRINSICS_AVX512)
#   if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
        internal_libxs_shuffle2_u64_avx512((unsigned long long*)out, (const unsigned long long*)inp, count, s);
        i = size;
#   elif (LIBXS_X86_AVX512 <= LIBXS_MAX_STATIC_TARGET_ARCH)
        if (LIBXS_X86_AVX512 <= libxs_cpuid(NULL)) {
          internal_libxs_shuffle2_u64_avx512((unsigned long long*)out, (const unsigned long long*)inp, count, s);
          i = size;
        }
#   endif
# endif
# if defined(LIBXS_INTRINSICS_AVX2)
        if (i < size) {
#   if (LIBXS_X86_AVX2 <= LIBXS_STATIC_TARGET_ARCH)
          internal_libxs_shuffle2_u64_avx2((unsigned long long*)out, (const unsigned long long*)inp, count, s);
          i = size;
#   elif (LIBXS_X86_AVX2 <= LIBXS_MAX_STATIC_TARGET_ARCH)
          if (LIBXS_X86_AVX2 <= libxs_cpuid(NULL)) {
            internal_libxs_shuffle2_u64_avx2((unsigned long long*)out, (const unsigned long long*)inp, count, s);
            i = size;
          }
#   endif
        }
# endif
      }
#endif
      if (i < size && elemsize < 128) {
        switch (elemsize) {
          case 8: for (; i < size; i += 8, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - 8 * j, 8);
          } break;
          case 4: for (; i < size; i += 4, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - 4 * j, 4);
          } break;
          case 2: for (; i < size; i += 2, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - 2 * j, 2);
          } break;
          case 1: for (; i < size; ++i, j += s) {
            if (count < j) j -= count;
            out[i] = inp[size-j];
          } break;
          default: for (; i < size; i += elemsize, j += s) {
            if (count < j) j -= count;
            LIBXS_MEMCPY(out + i, inp + size - elemsize * j, elemsize);
          }
        }
      }
      if (i < size) { /* generic path */
        for (; i < size; i += elemsize, j += s) {
          if (count < j) j -= count;
          memcpy(out + i, inp + size - elemsize * j, elemsize);
        }
      }
    }
    else if (0 != *nrepeat) { /* generic path */
      const size_t c = count - 1;
      for (; i < count; ++i) {
        size_t k = 0;
        LIBXS_ASSERT(NULL != inp && NULL != out);
        for (j = i; k < *nrepeat; ++k) j = c - ((s * j) % count);
        memcpy(out + elemsize * i, inp + elemsize * j, elemsize);
      }
    }
    else { /* ordinary copy */
      memcpy(out, inp, size);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXS_API size_t libxs_unshuffle(size_t count, const size_t* shuffle)
{
  size_t result = 0;
  if (0 < count) {
    const size_t n = (NULL == shuffle ? LIBXS_MEM_SHUFFLE_COPRIME(count) : *shuffle);
    size_t c = count - 1, j = c, d;
    do {
      d = (j * n) % count;
      ++result;
      j = c - d;
    } while (0 != d && result < count);
  }
  LIBXS_ASSERT(result <= count);
  return result;
}
