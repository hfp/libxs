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
    for (ii = 1; ii < m; ++ii) {
      const int key = perm[ii];
      const double keyval = scores[key];
      jj = ii - 1;
      while (jj >= 0 && scores[perm[jj]] > keyval) {
        perm[jj + 1] = perm[jj];
        --jj;
      }
      perm[jj + 1] = key;
    }
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
    for (ii = 1; ii < m; ++ii) {
      const int key = perm[ii];
      const double keyval = scores[key];
      jj = ii - 1;
      while (jj >= 0 && scores[perm[jj]] > keyval) {
        perm[jj + 1] = perm[jj];
        --jj;
      }
      perm[jj + 1] = key;
    }
  }
  else if (LIBXS_SORT_GREEDY == method) {
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
