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


LIBXS_API_INLINE void internal_libxs_kdtree_build(
  const double* pts, int* idx, int lo, int hi,
  int ndims, int stride, int min_leaf,
  libxs_kdtree_split_t split_fn, void* split_ctx, int depth)
{
  const int count = hi - lo;
  if (count > 1 && count > min_leaf) {
    int k = 0, pos = 0, do_split = 1;
    if (NULL != split_fn) {
      do_split = (0 == split_fn(
        &k, &pos, pts, idx + lo, count, depth, 0, split_ctx));
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
      internal_libxs_kdtree_build(pts, idx, lo, mid,
        ndims, stride, min_leaf, split_fn, split_ctx, depth + 1);
      internal_libxs_kdtree_build(pts, idx, mid, hi,
        ndims, stride, min_leaf, split_fn, split_ctx, depth + 1);
    }
  }
}


LIBXS_API void libxs_kdtree_build(
  const double* pts, int* idx, int n, int ndims, int stride,
  const libxs_kdtree_config_t* config)
{
  if (NULL != pts && NULL != idx && n > 1 && ndims > 0) {
    const int min_leaf = (NULL != config) ? config->min_leaf : 0;
    libxs_kdtree_split_t split_fn = (NULL != config) ? config->split : NULL;
    void* split_ctx = (NULL != config) ? config->ctx : NULL;
    if (2 == ndims && 2 == stride && NULL == split_fn && 0 >= min_leaf) {
      internal_libxs_kdtree2d_build_entry(pts, idx, n);
    }
    else {
      internal_libxs_kdtree_build(pts, idx, 0, n, ndims, stride,
        min_leaf, split_fn, split_ctx, 0);
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
    libxs_kdtree_split_t split_fn = (NULL != config) ? config->split : NULL;
    void* split_ctx = (NULL != config) ? config->ctx : NULL;
    int stack_lo[64], stack_hi[64], stack_depth[64], sp = 0;
    stack_lo[0] = 0; stack_hi[0] = n; stack_depth[0] = 0; sp = 1;
    while (sp > 0) {
      const int lo = stack_lo[--sp];
      const int hi = stack_hi[sp];
      const int depth = stack_depth[sp];
      const int count = hi - lo;
      int k = 0, pos = 0, is_leaf = 0;
      if (count <= 1 || count <= min_leaf) {
        is_leaf = 1;
      }
      else if (NULL != split_fn) {
        is_leaf = (0 != split_fn(&k, &pos, pts, idx + lo,
          count, depth, result, split_ctx));
      }
      else {
        k = depth % ndims;
        pos = count / 2;
      }
      if (0 != is_leaf || pos < 1 || pos >= count) {
        int i;
        for (i = lo; i < hi; ++i) assignments[idx[i]] = result;
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
        if (sp < 64) {
          stack_lo[sp] = mid; stack_hi[sp] = hi;
          stack_depth[sp] = depth + 1; ++sp;
        }
        if (sp < 64) {
          stack_lo[sp] = lo; stack_hi[sp] = mid;
          stack_depth[sp] = depth + 1; ++sp;
        }
      }
    }
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


