/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

LIBXS_EXTERN_C typedef struct internal_libxs_kdtree2d_ctx_t {
  const double* pts;
  const int* idx;
  const unsigned char* used;
  double qx, qy;
} internal_libxs_kdtree2d_ctx_t;

LIBXS_EXTERN_C typedef struct internal_libxs_kdtree_ctx_t {
  const double* pts;
  const int* idx;
  const unsigned char* used;
  const double* query;
  int ndims, stride;
} internal_libxs_kdtree_ctx_t;

LIBXS_EXTERN_C typedef struct internal_libxs_kdtree_knn_ctx_t {
  const double* pts;
  const int* idx;
  const unsigned char* used;
  const double* query;
  int* out_idx;
  double* out_dist2;
  double max_dist2;
  int ndims, stride, k, count;
} internal_libxs_kdtree_knn_ctx_t;

/* A subtree waiting in best-bin-first order: range and depth, and a lower bound of its squared distance. */
LIBXS_EXTERN_C typedef struct internal_libxs_kdtree_bbf_t {
  double bound;
  int lo, hi, depth;
} internal_libxs_kdtree_bbf_t;


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


/**
 * libxs_kdtree_split_middle: the axis of largest extent at the middle of its range (threshold), the
 * points below first (idx reordered); returns zero for a split, nonzero for a leaf (no extent).
 */
LIBXS_API_INLINE int internal_libxs_kdtree_middle(int* dim, int* pos, double* threshold,
  const double* pts, int* idx, int count, int ndims, int stride)
{
  double ext = 0, lo = 0, hi = 0;
  int result = 1, best = -1, d, i;
  for (d = 0; d < ndims; ++d) {
    double a = pts[(size_t)idx[0] * stride + d], b = a;
    for (i = 1; i < count; ++i) {
      const double x = pts[(size_t)idx[i] * stride + d];
      if (x < a) a = x;
      else if (b < x) b = x;
    }
    if (ext < b - a) {
      ext = b - a;
      lo = a;
      hi = b;
      best = d;
    }
  }
  if (0 <= best) {
    const double mid = lo + 0.5 * (hi - lo);
    int l = 0;
    for (i = 0; i < count; ++i) {
      if (pts[(size_t)idx[i] * stride + best] < mid) {
        const int t = idx[l];
        idx[l] = idx[i];
        idx[i] = t;
        ++l;
      }
    }
    *dim = best;
    *pos = l;
    *threshold = mid;
    result = ((0 < l && l < count) ? 0 : 1);
  }
  return result;
}


/* Midway between the largest coordinate along dim of the left entries (pos) and the smallest of the right. */
LIBXS_API_INLINE double internal_libxs_kdtree_midway(const double* pts, const int* idx,
  int pos, int count, int dim, int stride)
{
  double a = pts[(size_t)idx[0] * stride + dim], b = pts[(size_t)idx[pos] * stride + dim];
  int i;
  for (i = 1; i < pos; ++i) a = LIBXS_MAX(a, pts[(size_t)idx[i] * stride + dim]);
  for (i = pos + 1; i < count; ++i) b = LIBXS_MIN(b, pts[(size_t)idx[i] * stride + dim]);
  return a + 0.5 * (b - a);
}


/* nleaves: the leaves (nonempty nodes without children) so far, in preorder as partition counts */
LIBXS_API_INLINE void internal_libxs_kdtree_build(
  const double* pts, int* idx, int lo, int hi,
  int ndims, int stride, int min_leaf,
  libxs_kdtree_split_t split_fn, void* split_ctx, int depth, int* nleaves)
{
  const int count = hi - lo;
  if (count > 1 && count > min_leaf) {
    double threshold = HUGE_VAL; /* not kept: the tree is implicit in idx */
    int k = 0, pos = 0, do_split = 1;
    if (libxs_kdtree_split_middle == split_fn) {
      do_split = (0 == internal_libxs_kdtree_middle(&k, &pos, &threshold, pts, idx + lo, count, ndims, stride));
    }
    else if (NULL != split_fn) {
      do_split = (0 == split_fn(
        &k, &pos, &threshold, pts, idx + lo, count, depth, *nleaves, split_ctx));
    }
    else {
      k = depth % ndims;
      pos = count / 2;
    }
    if (0 != do_split && pos > 0 && pos < count) {
      const int mid = lo + pos;
      if (NULL == split_fn) {
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
      }
      /**
       * idx[mid] is the PIVOT and belongs to neither subtree, exactly as the
       * two-dimensional specialization above splits and as the query traversals
       * assume when they read the split value from idx[mid] and descend
       * [lo,mid) and [mid+1,hi).
       *
       * Recursing into [mid,hi) instead put the pivot in the right subtree, so
       * every deeper node of that subtree sat at a different position than a
       * query computes for [mid+1,hi). The searches then pruned against split
       * values belonging to other nodes and discarded regions that held closer
       * points: libxs_kdtree_nearest returned a non-nearest point on 6 of 20
       * random three-dimensional queries. The 2-D path was unaffected, which is
       * why the callers in libxs_math (all two-dimensional) never saw it.
       */
      internal_libxs_kdtree_build(pts, idx, lo, mid,
        ndims, stride, min_leaf, split_fn, split_ctx, depth + 1, nleaves);
      internal_libxs_kdtree_build(pts, idx, mid + 1, hi,
        ndims, stride, min_leaf, split_fn, split_ctx, depth + 1, nleaves);
    }
    else ++*nleaves;
  }
  else if (0 < count) ++*nleaves;
}


LIBXS_API int libxs_kdtree_split_median(int* dim, int* pos, double* threshold,
  const double* pts, int* idx, int count, int depth, int nleaves, void* ctx)
{
  LIBXS_UNUSED(dim);
  LIBXS_UNUSED(pos);
  LIBXS_UNUSED(threshold);
  LIBXS_UNUSED(pts);
  LIBXS_UNUSED(idx);
  LIBXS_UNUSED(count);
  LIBXS_UNUSED(depth);
  LIBXS_UNUSED(nleaves);
  LIBXS_UNUSED(ctx);
  return 1; /* a marker: the tree splits itself */
}


LIBXS_API int libxs_kdtree_split_middle(int* dim, int* pos, double* threshold,
  const double* pts, int* idx, int count, int depth, int nleaves, void* ctx)
{
  LIBXS_UNUSED(dim);
  LIBXS_UNUSED(pos);
  LIBXS_UNUSED(threshold);
  LIBXS_UNUSED(pts);
  LIBXS_UNUSED(idx);
  LIBXS_UNUSED(count);
  LIBXS_UNUSED(depth);
  LIBXS_UNUSED(nleaves);
  LIBXS_UNUSED(ctx);
  return 1; /* a marker: the tree splits itself */
}


LIBXS_API void libxs_kdtree_build(
  const double* pts, int* idx, int n, int ndims, int stride,
  const libxs_kdtree_config_t* config)
{
  if (NULL != pts && NULL != idx && n > 1 && ndims > 0) {
    const int min_leaf = (NULL != config) ? config->min_leaf : 0;
    /* libxs_kdtree_split_median is the default (NULL) */
    libxs_kdtree_split_t split_fn = (NULL != config && libxs_kdtree_split_median != config->split)
      ? config->split : NULL;
    void* split_ctx = (NULL != config) ? config->ctx : NULL;
    if (2 == ndims && 2 == stride && NULL == split_fn && 0 >= min_leaf) {
      internal_libxs_kdtree2d_build_entry(pts, idx, n);
    }
    else {
      int nleaves = 0;
      internal_libxs_kdtree_build(pts, idx, 0, n, ndims, stride,
        min_leaf, split_fn, split_ctx, 0, &nleaves);
    }
  }
}


LIBXS_API int libxs_kdtree_partition(
  const double* pts, int* idx, int n, int ndims, int stride,
  int* assignments, const libxs_kdtree_config_t* config)
{
  int result = 0;
  if (NULL != pts && NULL != idx && n > 0 && ndims > 0
    && NULL != assignments)
  {
    const int min_leaf = (NULL != config) ? config->min_leaf : 0;
    /* libxs_kdtree_split_median is the default (NULL) */
    libxs_kdtree_split_t split_fn = (NULL != config && libxs_kdtree_split_median != config->split)
      ? config->split : NULL;
    void* split_ctx = (NULL != config) ? config->ctx : NULL;
    /**
     * Pending nodes, at most one per leaf still to come. A fixed depth dropped
     * the children of an unbalanced split chain silently, leaving their points
     * in leaf zero: a split that peels min_leaf points at a time is n/min_leaf
     * deep.
     */
    libxs_kdtree_node_t *const nodes = (NULL != config) ? config->nodes : NULL;
    int stack_pool = 0, sp = 0, nnodes = 0;
    int* const stack_lo = (int*)internal_libxs_scratch_malloc(
      (size_t)4 * (n + 1) * sizeof(int), &stack_pool);
    int* const stack_hi = (NULL != stack_lo) ? (stack_lo + n + 1) : NULL;
    int* const stack_depth = (NULL != stack_lo) ? (stack_hi + n + 1) : NULL;
    int* const stack_parent = (NULL != stack_lo) ? (stack_depth + n + 1) : NULL; /* next to patch */
    if (NULL != stack_lo) {
      stack_lo[0] = 0; stack_hi[0] = n; stack_depth[0] = 0; stack_parent[0] = -1; sp = 1;
    }
    while (sp > 0) { /* preorder: a node, then its left part, then its right part */
      const int lo = stack_lo[--sp];
      const int hi = stack_hi[sp];
      const int depth = stack_depth[sp];
      const int parent = stack_parent[sp];
      const int count = hi - lo, node = nnodes;
      double threshold = HUGE_VAL;
      int k = 0, pos = 0, is_leaf = 0;
      if (NULL != nodes) {
        if (0 <= parent) nodes[parent].next = node;
        ++nnodes;
      }
      if (count <= 1 || count <= min_leaf) {
        is_leaf = 1;
      }
      else if (libxs_kdtree_split_middle == split_fn) {
        is_leaf = internal_libxs_kdtree_middle(&k, &pos, &threshold, pts, idx + lo, count, ndims, stride);
      }
      else if (NULL != split_fn) {
        is_leaf = (0 != split_fn(&k, &pos, &threshold, pts, idx + lo,
          count, depth, result, split_ctx));
      }
      else {
        k = depth % ndims;
        pos = count / 2;
      }
      if (0 != is_leaf || pos < 1 || pos >= count) {
        int i;
        for (i = lo; i < hi; ++i) assignments[idx[i]] = result;
        if (NULL != nodes) {
          nodes[node].threshold = 0;
          nodes[node].dim = -1;
          nodes[node].next = result;
        }
        ++result;
      }
      else {
        const int mid = lo + pos;
        if (NULL == split_fn) {
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
        }
        if (NULL != nodes) {
          if (HUGE_VAL == threshold) {
            threshold = internal_libxs_kdtree_midway(pts, idx + lo, pos, count, k, stride);
          }
          nodes[node].threshold = threshold;
          nodes[node].dim = k;
          nodes[node].next = -1; /* the right part's node, once it is taken */
        }
        stack_lo[sp] = mid; stack_hi[sp] = hi;
        stack_depth[sp] = depth + 1; stack_parent[sp] = node; ++sp;
        stack_lo[sp] = lo; stack_hi[sp] = mid;
        stack_depth[sp] = depth + 1; stack_parent[sp] = -1; ++sp;
      }
    }
    internal_libxs_scratch_free(stack_lo, stack_pool);
  }
  return result;
}


LIBXS_API int libxs_kdtree_descend(const libxs_kdtree_node_t nodes[], const double point[])
{
  int result = -1;
  if (NULL != nodes && NULL != point) {
    int i = 0;
    while (0 <= i && 0 <= nodes[i].dim) { /* preorder: every step leads forward */
      const int j = (point[nodes[i].dim] < nodes[i].threshold ? (i + 1) : nodes[i].next);
      i = (i < j ? j : -1);
    }
    if (0 <= i) result = nodes[i].next;
  }
  return result;
}


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


/**
 * Offer one point to the running k-best set, which is kept sorted ascending so
 * the worst member is always last and the caller receives an ordered result
 * without a second pass.  k is small in practice, so an insertion costs less
 * than maintaining a heap.
 */
LIBXS_API_INLINE void internal_libxs_kdtree_knn_add(
  internal_libxs_kdtree_knn_ctx_t* ctx, int pi)
{
  double d2 = 0;
  int dd;
  for (dd = 0; dd < ctx->ndims; ++dd) {
    const double d = ctx->pts[(size_t)pi * ctx->stride + dd] - ctx->query[dd];
    d2 += d * d;
  }
  if (d2 <= ctx->max_dist2
    && (ctx->count < ctx->k || d2 < ctx->out_dist2[ctx->count - 1]))
  {
    int at = (ctx->count < ctx->k) ? ctx->count++ : (ctx->k - 1);
    while (0 < at && ctx->out_dist2[at - 1] > d2) {
      ctx->out_dist2[at] = ctx->out_dist2[at - 1];
      ctx->out_idx[at] = ctx->out_idx[at - 1];
      --at;
    }
    ctx->out_dist2[at] = d2;
    ctx->out_idx[at] = pi;
  }
}


/**
 * Same traversal as internal_libxs_kdtree_find: idx[mid] is the pivot for this
 * node and the subtrees are [lo,mid) and [mid+1,hi).  The far side is entered
 * only while the k-best set is short or the split plane is nearer than its worst
 * member, which is what makes the search exact rather than approximate.
 */
LIBXS_API_INLINE void internal_libxs_kdtree_knn_visit(
  internal_libxs_kdtree_knn_ctx_t* ctx, int lo, int hi, int depth)
{
  if (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const int pi = ctx->idx[mid];
    const int kd = depth % ctx->ndims;
    double dist;
    int near_lo, near_hi, far_lo, far_hi;
    if (NULL == ctx->used || 0 == ctx->used[pi]) {
      internal_libxs_kdtree_knn_add(ctx, pi);
    }
    dist = ctx->query[kd] - ctx->pts[(size_t)pi * ctx->stride + kd];
    if (dist <= 0) {
      near_lo = lo; near_hi = mid; far_lo = mid + 1; far_hi = hi;
    }
    else {
      near_lo = mid + 1; near_hi = hi; far_lo = lo; far_hi = mid;
    }
    internal_libxs_kdtree_knn_visit(ctx, near_lo, near_hi, depth + 1);
    if (ctx->count < ctx->k || dist * dist < ctx->out_dist2[ctx->count - 1]) {
      internal_libxs_kdtree_knn_visit(ctx, far_lo, far_hi, depth + 1);
    }
  }
}


LIBXS_API int libxs_kdtree_knearest(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2,
  int out_idx[], double out_dist2[], int k)
{
  int result = 0;
  if (NULL != pts && NULL != idx && NULL != query && NULL != out_idx
    && NULL != out_dist2 && 0 < n && 0 < ndims && 0 < k)
  {
    internal_libxs_kdtree_knn_ctx_t ctx;
    ctx.pts = pts; ctx.idx = idx; ctx.used = used; ctx.query = query;
    ctx.out_idx = out_idx; ctx.out_dist2 = out_dist2;
    ctx.max_dist2 = max_dist2;
    ctx.ndims = ndims; ctx.stride = stride; ctx.k = k; ctx.count = 0;
    internal_libxs_kdtree_knn_visit(&ctx, 0, n, 0);
    result = ctx.count;
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_kdtree_bbf_push(internal_libxs_kdtree_bbf_t* heap, int* size,
  double bound, int lo, int hi, int depth)
{
  int i = (*size)++;
  while (0 < i && bound < heap[(i - 1) / 2].bound) {
    heap[i] = heap[(i - 1) / 2];
    i = (i - 1) / 2;
  }
  heap[i].bound = bound;
  heap[i].lo = lo;
  heap[i].hi = hi;
  heap[i].depth = depth;
}


LIBXS_API_INLINE internal_libxs_kdtree_bbf_t internal_libxs_kdtree_bbf_pop(
  internal_libxs_kdtree_bbf_t* heap, int* size)
{
  const internal_libxs_kdtree_bbf_t result = heap[0], last = heap[--(*size)];
  int i = 0, c;
  for (c = 1; c < *size; c = 2 * i + 1) {
    if (c + 1 < *size && heap[c + 1].bound < heap[c].bound) ++c;
    if (last.bound <= heap[c].bound) break;
    heap[i] = heap[c];
    i = c;
  }
  if (0 < *size) heap[i] = last;
  return result;
}


LIBXS_API int libxs_kdtree_knearest_budget(
  const double* pts, const int* idx, const unsigned char* used,
  int n, int ndims, int stride, const double* query, double max_dist2,
  int out_idx[], double out_dist2[], int k, int budget)
{
  int result = 0;
  if (NULL != pts && NULL != idx && NULL != query && NULL != out_idx
    && NULL != out_dist2 && 0 < n && 0 < ndims && 0 < k)
  {
    /* every point examined adds at most one subtree: the bound or n (no budget) plus the root */
    const int cap = ((0 < budget && budget < n) ? budget : n) + 1;
    int pool = 0, size = 0, examined = 0;
    internal_libxs_kdtree_bbf_t *const heap = (internal_libxs_kdtree_bbf_t*)internal_libxs_scratch_malloc(
      (size_t)cap * sizeof(internal_libxs_kdtree_bbf_t), &pool);
    internal_libxs_kdtree_knn_ctx_t ctx;
    ctx.pts = pts; ctx.idx = idx; ctx.used = used; ctx.query = query;
    ctx.out_idx = out_idx; ctx.out_dist2 = out_dist2;
    ctx.max_dist2 = max_dist2;
    ctx.ndims = ndims; ctx.stride = stride; ctx.k = k; ctx.count = 0;
    if (NULL != heap) internal_libxs_kdtree_bbf_push(heap, &size, 0, 0, n, 0);
    while (0 < size && (0 >= budget || examined < budget)) {
      const internal_libxs_kdtree_bbf_t node = internal_libxs_kdtree_bbf_pop(heap, &size);
      int lo = node.lo, hi = node.hi, depth = node.depth;
      /* the nearest subtree waiting cannot hold a nearer point: exact from here on */
      if (node.bound > max_dist2 || (ctx.count == k && node.bound >= out_dist2[k - 1])) break;
      while (lo < hi && (0 >= budget || examined < budget)) { /* down the near side */
        const int mid = lo + (hi - lo) / 2, pi = idx[mid], kd = depth % ndims;
        const double dist = query[kd] - pts[(size_t)pi * stride + kd];
        const double far = LIBXS_MAX(node.bound, dist * dist);
        if (NULL == used || 0 == used[pi]) internal_libxs_kdtree_knn_add(&ctx, pi);
        ++examined;
        if (dist <= 0) { /* far: [mid+1,hi) */
          if (mid + 1 < hi && far <= max_dist2 && (ctx.count < k || far < out_dist2[k - 1])) {
            internal_libxs_kdtree_bbf_push(heap, &size, far, mid + 1, hi, depth + 1);
          }
          hi = mid;
        }
        else { /* far: [lo,mid) */
          if (lo < mid && far <= max_dist2 && (ctx.count < k || far < out_dist2[k - 1])) {
            internal_libxs_kdtree_bbf_push(heap, &size, far, lo, mid, depth + 1);
          }
          lo = mid + 1;
        }
        ++depth;
      }
    }
    internal_libxs_scratch_free(heap, pool);
    result = ctx.count;
  }
  return result;
}
