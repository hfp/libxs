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


