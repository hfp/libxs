/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/* Template body for libxs_sort_smooth.
 * Included once per element type from libxs_perm.c.
 *
 * Required macros:
 *   LIBXS_SORT_TEMPLATE_ELEM_TYPE   - C type for one element
 *   LIBXS_SORT_TEMPLATE_TYPE2FP64(V) - convert element to double
 *
 * Expected variables in enclosing scope:
 *   method (libxs_sort_t), m, n, ld (int), mat (const void*),
 *   perm (int*), scores (double*), visited (char*)
 */

{
  const LIBXS_SORT_TEMPLATE_ELEM_TYPE *const real_mat =
    (const LIBXS_SORT_TEMPLATE_ELEM_TYPE*)mat;
  int ii, jj;

  for (ii = 0; ii < m; ++ii) perm[ii] = ii;

  if (LIBXS_SORT_IDENTITY == method) {
    /* identity permutation -- done */
  }
  else if (LIBXS_SORT_NORM == method) {
    for (ii = 0; ii < m; ++ii) {
      double acc = 0.0, comp = 0.0;
      for (jj = 0; jj < n; ++jj) {
        const double v = LIBXS_SORT_TEMPLATE_TYPE2FP64(
          real_mat[(size_t)jj * ld + ii]);
        libxs_kahan_sum(v < 0 ? -v : v, &acc, &comp);
      }
      scores[ii] = acc;
    }
    libxs_sort(perm, m, sizeof(int),
      internal_libxs_sort_smooth_cmp, (void*)scores);
  }
  else if (LIBXS_SORT_MEAN == method) {
    for (ii = 0; ii < m; ++ii) {
      double acc = 0.0, comp = 0.0;
      for (jj = 0; jj < n; ++jj) {
        libxs_kahan_sum(LIBXS_SORT_TEMPLATE_TYPE2FP64(
          real_mat[(size_t)jj * ld + ii]), &acc, &comp);
      }
      scores[ii] = (0 < n) ? (acc / n) : 0.0;
    }
    libxs_sort(perm, m, sizeof(int),
      internal_libxs_sort_smooth_cmp, (void*)scores);
  }
  else if (LIBXS_SORT_GREEDY == method) {
    int use_kd = 0;
    if (2 == n) {
      int pool_kd = 0;
      const size_t kdsz = (size_t)m * 2 * sizeof(double)
        + (size_t)m * sizeof(int) + (size_t)m;
      double* pts = (double*)internal_libxs_sort_malloc(kdsz, &pool_kd);
      if (NULL != pts) {
        int* idx = (int*)(pts + 2 * m);
        unsigned char* used = (unsigned char*)(idx + m);
        for (ii = 0; ii < m; ++ii) {
          pts[2*ii] = LIBXS_SORT_TEMPLATE_TYPE2FP64(real_mat[ii]);
          pts[2*ii+1] = LIBXS_SORT_TEMPLATE_TYPE2FP64(
            real_mat[(size_t)ld + ii]);
          idx[ii] = ii;
        }
        libxs_kdtree2d_build(pts, idx, m);
        memset(used, 0, (size_t)m);
        perm[0] = 0; used[0] = 1;
        for (ii = 1; ii < m; ++ii) {
          const int prev = perm[ii - 1];
          const double qx = pts[2*prev], qy = pts[2*prev+1];
          const int hit = libxs_kdtree2d_nearest(
            pts, idx, used, m, qx, qy, DBL_MAX);
          perm[ii] = (0 <= hit) ? hit : 0;
          if (0 <= hit) used[hit] = 1;
        }
        internal_libxs_sort_free(pts, pool_kd);
        use_kd = 1;
      }
    }
    if (0 == use_kd) {
      for (ii = 0; ii < m; ++ii) visited[ii] = 0;
      perm[0] = 0;
      visited[0] = 1;
      for (ii = 1; ii < m; ++ii) {
        const int prev = perm[ii - 1];
        double best_dist = DBL_MAX;
        int best_row = 0;
        for (jj = 0; jj < m; ++jj) {
          double dist = 0.0;
          int kk;
          if (0 != visited[jj]) continue;
          for (kk = 0; kk < n; ++kk) {
            const double d =
              LIBXS_SORT_TEMPLATE_TYPE2FP64(
                real_mat[(size_t)kk * ld + prev]) -
              LIBXS_SORT_TEMPLATE_TYPE2FP64(
                real_mat[(size_t)kk * ld + jj]);
            dist += d * d;
            if (dist >= best_dist) break;
          }
          if (dist < best_dist) {
            best_dist = dist;
            best_row = jj;
          }
        }
        perm[ii] = best_row;
        visited[best_row] = 1;
      }
    }
  }
  else if (LIBXS_SORT_MORTON == method) {
    const int bpd = LIBXS_MIN(53 / LIBXS_MAX(n, 1), 21);
    const unsigned int levels = (1u << bpd) - 1;
    double col_min[64], col_range[64];
    for (jj = 0; jj < n && jj < 64; ++jj) {
      double lo, hi;
      lo = hi = LIBXS_SORT_TEMPLATE_TYPE2FP64(real_mat[(size_t)jj * ld]);
      for (ii = 1; ii < m; ++ii) {
        const double v = LIBXS_SORT_TEMPLATE_TYPE2FP64(
          real_mat[(size_t)jj * ld + ii]);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      col_min[jj] = lo;
      col_range[jj] = (hi > lo) ? (hi - lo) : 1.0;
    }
    for (ii = 0; ii < m; ++ii) {
      unsigned int coords[64];
      for (jj = 0; jj < n && jj < 64; ++jj) {
        const double v = LIBXS_SORT_TEMPLATE_TYPE2FP64(
          real_mat[(size_t)jj * ld + ii]);
        unsigned int q = (unsigned int)((v - col_min[jj]) * levels / col_range[jj]);
        if (q > levels) q = levels;
        coords[jj] = q;
      }
      scores[ii] = (double)libxs_morton(coords, n);
    }
    libxs_sort(perm, m, sizeof(int),
      internal_libxs_sort_smooth_cmp, (void*)scores);
  }
  else if (LIBXS_SORT_HILBERT == method) {
    const int bpd = LIBXS_MIN(53 / LIBXS_MAX(n, 1), 21);
    const unsigned int levels = (1u << bpd) - 1;
    double col_min[64], col_range[64];
    for (jj = 0; jj < n && jj < 64; ++jj) {
      double lo, hi;
      lo = hi = LIBXS_SORT_TEMPLATE_TYPE2FP64(real_mat[(size_t)jj * ld]);
      for (ii = 1; ii < m; ++ii) {
        const double v = LIBXS_SORT_TEMPLATE_TYPE2FP64(
          real_mat[(size_t)jj * ld + ii]);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      col_min[jj] = lo;
      col_range[jj] = (hi > lo) ? (hi - lo) : 1.0;
    }
    for (ii = 0; ii < m; ++ii) {
      unsigned int coords[64];
      for (jj = 0; jj < n && jj < 64; ++jj) {
        const double v = LIBXS_SORT_TEMPLATE_TYPE2FP64(
          real_mat[(size_t)jj * ld + ii]);
        unsigned int q = (unsigned int)((v - col_min[jj]) * levels / col_range[jj]);
        if (q > levels) q = levels;
        coords[jj] = q;
      }
      scores[ii] = (double)libxs_hilbert(coords, n);
    }
    libxs_sort(perm, m, sizeof(int),
      internal_libxs_sort_smooth_cmp, (void*)scores);
  }
}
