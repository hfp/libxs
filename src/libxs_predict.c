/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_predict.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_str.h>
#include <libxs/libxs_malloc.h>
#include <libxs/libxs_gemm.h>
#include "libxs_main.h"

#if !defined(LIBXS_PREDICT_MAXITER)
#  define LIBXS_PREDICT_MAXITER 100
#endif
#if !defined(LIBXS_PREDICT_MAGIC)
#  define LIBXS_PREDICT_MAGIC 0x58535052U /* "XSPR" */
#endif
#if !defined(LIBXS_PREDICT_MAGIC_HKNN)
#  define LIBXS_PREDICT_MAGIC_HKNN 0x58534B4EU /* "XSKN" */
#endif
#if !defined(LIBXS_PREDICT_VERSION)
#  define LIBXS_PREDICT_VERSION 1
#endif
#if !defined(LIBXS_PREDICT_KNN)
#  define LIBXS_PREDICT_KNN 32
#endif

#define LIBXS_PREDICT_MALLOC(SIZE, POOL) internal_libxs_scratch_malloc(SIZE, &(POOL))
#define LIBXS_PREDICT_FREE(PTR, POOL) internal_libxs_scratch_free(PTR, POOL)


typedef struct internal_libxs_predict_entry_t {
  double* inputs;
  double* outputs;
} internal_libxs_predict_entry_t;

typedef struct internal_libxs_predict_cluster_t {
  double* centroid;
  double* coeffs;
  double* errors;
  double* kd_pts;
  double* raw_outputs;
  int* order;
  int* interpolated;
  int* mode;
  int* ndistinct;
  int* sorted_idx;
  double* sorted_dist;
  double dmax;
  double fprint_sig;
  int nentries;
  int maxorder;
  int k_eff;
} internal_libxs_predict_cluster_t;

typedef struct internal_libxs_predict_order_ctx_t {
  libxs_predict_t* model;
  int nclusters;
} internal_libxs_predict_order_ctx_t;

typedef struct internal_libxs_predict_rf_node_t {
  int feature;
  double threshold;
  int left, right;
  int label;
} internal_libxs_predict_rf_node_t;

typedef struct internal_libxs_predict_rf_tree_t {
  internal_libxs_predict_rf_node_t* nodes;
  int nnodes;
} internal_libxs_predict_rf_tree_t;

typedef struct internal_libxs_predict_rf_t {
  internal_libxs_predict_rf_tree_t* trees;
  int* label_offset;
  int ntrees;
  int noutputs;
} internal_libxs_predict_rf_t;

typedef struct {
  double val;
  int idx;
} internal_libxs_predict_rf_pair_t;

LIBXS_EXTERN_C struct libxs_predict_t {
  internal_libxs_predict_entry_t* entries;
  internal_libxs_predict_cluster_t* clusters;
  int* assignments;
  int* hknn_assignments;
  int** hknn_po_assignments;
  int* hknn_po_nclusters;
  internal_libxs_predict_cluster_t** hknn_po_clusters;
  double* eval_buf;
  double* input_min;
  double* input_rng;
  double* weights;
  int* transforms;
  double* ts_buf;
  double* decompose_mat;
  internal_libxs_predict_rf_t* rf;
  libxs_lock_t lock;
  int order;
  int ninputs, noutputs;
  int nentries, capacity;
  int nclusters;
  int hknn_nclusters;
  int built;
  int eval_mode;
  int iterations;
  int nseries, window, target, decompose;
  int nts, ts_capacity;
  int diff_mode, diff_order;
  int refine;
  double smooth;
  double quality;
  double consistency;
  volatile int phase;
};


LIBXS_API_INLINE void internal_libxs_predict_free_clusters(libxs_predict_t* model)
{
  if (NULL != model->clusters) {
    int c;
    for (c = 0; c < model->nclusters; ++c) {
      internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      free(cl->centroid);
      free(cl->coeffs);
      free(cl->errors);
      free(cl->kd_pts);
      free(cl->raw_outputs);
      free(cl->order);
      free(cl->interpolated);
      free(cl->mode);
      free(cl->ndistinct);
      free(cl->sorted_idx);
      free(cl->sorted_dist);
    }
    free(model->clusters);
    model->clusters = NULL;
  }
  free(model->assignments);
  model->assignments = NULL;
  free(model->eval_buf);
  model->eval_buf = NULL;
  model->nclusters = 0;
  model->built = 0;
}


LIBXS_API_INLINE void internal_libxs_predict_normalize(
  const libxs_predict_t* model, const double* inputs, double* norm)
{
  const int m = model->ninputs;
  int i;
  for (i = 0; i < m; ++i) {
    norm[i] = (NULL != model->input_rng && model->input_rng[i] > 0)
      ? (inputs[i] - model->input_min[i]) / model->input_rng[i] : inputs[i];
    if (NULL != model->weights) norm[i] *= model->weights[i];
  }
}


LIBXS_API_INLINE void internal_libxs_predict_kmeans(libxs_predict_t* model, int nclusters)
{
  const int m = model->ninputs;
  const int p = model->nentries;
  int pool_pts = 0, pool_cen = 0, pool_comp = 0, pool_cnt = 0, pool_dist = 0;
  double* pts = (double*)LIBXS_PREDICT_MALLOC((size_t)p * (size_t)m * sizeof(double), pool_pts);
  double* centroids = (double*)LIBXS_PREDICT_MALLOC((size_t)nclusters * (size_t)m * sizeof(double), pool_cen);
  double* comp = (double*)LIBXS_PREDICT_MALLOC((size_t)nclusters * (size_t)m * sizeof(double), pool_comp);
  int* counts = (int*)LIBXS_PREDICT_MALLOC((size_t)nclusters * sizeof(int), pool_cnt);
  double* dists = (double*)LIBXS_PREDICT_MALLOC((size_t)p * sizeof(double), pool_dist);
  if (NULL != pts && NULL != centroids && NULL != counts && NULL != comp && NULL != dists) {
    int c, i, j, iter;
    for (i = 0; i < p; ++i) {
      internal_libxs_predict_normalize(model,
        model->entries[i].inputs, pts + (size_t)i * m);
    }
    { const size_t seed = (0 == (model->eval_mode & LIBXS_PREDICT_TEMPORAL))
        ? LIBXS_SHUFFLE_INDEX(0, (size_t)p, libxs_coprime2((size_t)p), 0)
        : 0;
      memcpy(centroids, pts + seed * m, (size_t)m * sizeof(double));
    }
    for (i = 0; i < p; ++i) dists[i] = DBL_MAX;
    for (c = 1; c < nclusters; ++c) {
      int farthest = 0;
      double maxd = 0;
      for (i = 0; i < p; ++i) {
        const double d = libxs_dist2(
          pts + (size_t)i * m, centroids + (size_t)(c - 1) * m, m);
        if (d < dists[i]) dists[i] = d;
        if (dists[i] > maxd) { maxd = dists[i]; farthest = i; }
      }
      memcpy(centroids + (size_t)c * m, pts + (size_t)farthest * m,
        (size_t)m * sizeof(double));
    }
    /* Lloyd iterations with Kahan-compensated centroid accumulation */
    for (iter = 0; iter < LIBXS_PREDICT_MAXITER; ++iter) {
      int changed = 0;
      for (i = 0; i < p; ++i) {
        double best = libxs_dist2(pts + (size_t)i * m, centroids, m);
        int bestc = 0;
        for (c = 1; c < nclusters; ++c) {
          const double d = libxs_dist2(
            pts + (size_t)i * m, centroids + (size_t)c * m, m);
          if (d < best) { best = d; bestc = c; }
        }
        if (model->assignments[i] != bestc) {
          model->assignments[i] = bestc;
          changed = 1;
        }
      }
      if (0 == changed) iter = LIBXS_PREDICT_MAXITER;
      else {
        memset(centroids, 0, (size_t)nclusters * (size_t)m * sizeof(double));
        memset(comp, 0, (size_t)nclusters * (size_t)m * sizeof(double));
        memset(counts, 0, (size_t)nclusters * sizeof(int));
        for (i = 0; i < p; ++i) {
          const int ci = model->assignments[i];
          double* cen = centroids + (size_t)ci * m;
          double* cmp = comp + (size_t)ci * m;
          for (j = 0; j < m; ++j) {
            libxs_kahan_sum(pts[(size_t)i * m + j], &cen[j], &cmp[j]);
          }
          ++counts[ci];
        }
        for (c = 0; c < nclusters; ++c) {
          if (0 < counts[c]) {
            double* cen = centroids + (size_t)c * m;
            for (j = 0; j < m; ++j) cen[j] /= counts[c];
          }
        }
      }
    }
    for (c = 0; c < nclusters; ++c) {
      memcpy(model->clusters[c].centroid, centroids + (size_t)c * m, (size_t)m * sizeof(double));
    }
  }
  LIBXS_PREDICT_FREE(dists, pool_dist);
  LIBXS_PREDICT_FREE(comp, pool_comp);
  LIBXS_PREDICT_FREE(centroids, pool_cen);
  LIBXS_PREDICT_FREE(counts, pool_cnt);
  LIBXS_PREDICT_FREE(pts, pool_pts);
}


LIBXS_API_INLINE double internal_libxs_predict_local_error(
  const libxs_predict_t* model, const internal_libxs_predict_cluster_t* cl,
  int pos, int output_j)
{
  double result = cl->errors[output_j];
  const int nc = cl->nentries;
  const int radius = LIBXS_MIN(4, nc / 2);
  if (radius >= 2 && NULL != model->entries && NULL != cl->sorted_idx) {
    const int lo = LIBXS_MAX(pos - radius, 0);
    const int hi = LIBXS_MIN(pos + radius, nc - 1);
    const int len = hi - lo + 1;
    if (len >= 3) {
      double local[9];
      libxs_fprint_t fp;
      const size_t shape = (size_t)len;
      int k;
      for (k = 0; k < len; ++k) {
        local[k] = model->entries[cl->sorted_idx[lo + k]].outputs[output_j];
      }
      if (EXIT_SUCCESS == libxs_fprint(&fp, LIBXS_DATATYPE_F64, local,
        1, &shape, NULL, LIBXS_MIN(2, len - 1), 0, 0, 0))
      {
        const double raw1 = libxs_fprint_raw(&fp, 1, fp.linf[1]);
        result = LIBXS_MIN(result, raw1);
      }
    }
  }
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_position(
  const libxs_predict_t* model, const internal_libxs_predict_cluster_t* cl,
  const double* inputs)
{
  const int nc = cl->nentries;
  const int m = model->ninputs;
  /* linear scan: clusters are small (sqrt(P) entries typically) */
  double best = DBL_MAX;
  int best_k = 0, k;
  for (k = 0; k < nc; ++k) {
    const double d = libxs_dist2(inputs, cl->kd_pts + (size_t)k * m, m);
    if (d < best) { best = d; best_k = k; }
  }
  return (double)best_k;
}


LIBXS_API_INLINE void internal_libxs_predict_cluster_refit(
  internal_libxs_predict_cluster_t* cl, int n, int use_fprint)
{
  const int nc = cl->nentries;
  const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
  int j, k;
  int buf_pool = 0;
  double* buf = (double*)LIBXS_PREDICT_MALLOC(
    (size_t)nc * sizeof(double), buf_pool);
  if (NULL != buf) {
    for (j = 0; j < n; ++j) {
      int ndistinct = 0, d;
      double prev;
      for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)k * n + j];
      libxs_sort(buf, nc, sizeof(double), libxs_cmp_f64, NULL);
      prev = buf[0]; ndistinct = 1;
      for (k = 1; k < nc; ++k) {
        if (buf[k] != prev) { ++ndistinct; prev = buf[k]; }
      }
      cl->ndistinct[j] = ndistinct;
      for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)k * n + j];
      if (0 != use_fprint) {
        const size_t shape = (size_t)nc;
        const size_t stride = (size_t)n;
        libxs_fprint_t fp;
        int decay_order = 0;
        libxs_fprint(&fp, LIBXS_DATATYPE_F64, cl->raw_outputs + j,
          1, &shape, &stride, LIBXS_FPRINT_MAXORDER, 0, 0, 0);
        cl->order[j] = cl->maxorder;
        cl->interpolated[j] = 0;
        if (0 < fp.l2[0]) {
          for (d = 1; d <= fp.order; ++d) {
            if (fp.l2[d] < fp.l2[d - 1]) ++decay_order;
            else d = fp.order + 1;
          }
        }
        if (ndistinct <= ndistinct_thresh || decay_order < 2) {
          cl->mode[j] = 1;
        }
        else {
          cl->mode[j] = 0;
          cl->interpolated[j] = 1;
          cl->order[j] = LIBXS_MIN(decay_order, cl->maxorder);
        }
      }
      else {
        if (ndistinct <= ndistinct_thresh) {
          cl->mode[j] = 1;
          cl->interpolated[j] = 0;
        }
      }
      { const int trunc_order = LIBXS_MIN(cl->order[j], cl->maxorder);
        cl->order[j] = LIBXS_MIN(trunc_order, nc - 1);
        cl->coeffs[(size_t)j * (cl->maxorder + 1)] = buf[0];
        for (d = 1; d <= cl->order[j] && d < nc; ++d) {
          for (k = 0; k < nc - d; ++k) buf[k] = buf[k + 1] - buf[k];
          cl->coeffs[(size_t)j * (cl->maxorder + 1) + d] = buf[0];
        }
        if (cl->order[j] < nc - 1) {
          double emax = 0;
          for (k = 0; k < nc - cl->order[j] - 1; ++k) {
            buf[k] = buf[k + 1] - buf[k];
          }
          for (k = 0; k < nc - cl->order[j] - 1; ++k) {
            const double a = buf[k] < 0 ? -buf[k] : buf[k];
            if (a > emax) emax = a;
          }
          cl->errors[j] = emax;
        }
        else {
          cl->errors[j] = 0;
        }
      }
    }
    LIBXS_PREDICT_FREE(buf, buf_pool);
  }
  { int nclassify = 0;
    double sig_sum = 0;
    for (j = 0; j < n; ++j) {
      nclassify += cl->mode[j];
      sig_sum += cl->errors[j];
    }
    cl->fprint_sig = (n > 0) ? sig_sum / n : 0.0;
    cl->k_eff = (nclassify > n / 2)
      ? LIBXS_MIN(LIBXS_MAX(5, nc / 3), LIBXS_PREDICT_KNN)
      : LIBXS_MIN(LIBXS_MAX(3, (int)(sqrt((double)nc) + 0.5)),
          LIBXS_PREDICT_KNN);
  }
}


LIBXS_API_INLINE double internal_libxs_predict_coverage(
  int nentries, int total_entries, int nclusters)
{
  const double expected = (double)total_entries / nclusters;
  const double ratio = (expected > 0) ? (double)nentries / expected : 1.0;
  return ratio < 1.0 ? ratio : 1.0;
}


LIBXS_API_INLINE double internal_libxs_predict_classify2(
  const internal_libxs_predict_cluster_t* cl, const double* kd_pts,
  int nc, int m, const double* inputs, int output_j, int nouts,
  int ndistinct, int extrapolate, int skip_local,
  const int* po_groups, int query_group,
  double* confidence, double* out_variance)
{
  const int k = cl->k_eff;
  const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
  double candidates[LIBXS_PREDICT_KNN];
  double dists[LIBXS_PREDICT_KNN];
  double best_val = 0.0;
  int nfound = 0, exact = 0, i, max_idx = 0;
  if (NULL != confidence) *confidence = 0.0;
  if (NULL != out_variance) *out_variance = 0.0;
  if (nc > 0 && NULL != cl->raw_outputs) {
    best_val = cl->raw_outputs[output_j];
    if (0 != extrapolate) {
      for (i = 0; i < nc; ++i) {
        if (cl->sorted_idx[i] > max_idx) max_idx = cl->sorted_idx[i];
      }
    }
    for (i = 0; i < nc; ++i) {
      double d2;
      if (i == skip_local) continue;
      d2 = libxs_dist2(inputs, kd_pts + (size_t)i * m, m);
      if (0 != extrapolate && max_idx > 0) {
        const double age = 1.0 - (double)cl->sorted_idx[i] / (double)max_idx;
        d2 *= 1.0 + 0.5 * age;
      }
      if (NULL != po_groups && query_group >= 0
        && po_groups[cl->sorted_idx[i]] != query_group)
      {
        continue;
      }
      if (nfound < k) {
        candidates[nfound] = cl->raw_outputs[(size_t)i * nouts + output_j];
        dists[nfound] = sqrt(d2);
        if (0 == nfound && 0.0 == d2) {
          best_val = candidates[0];
          exact = 1;
        }
        ++nfound;
      }
      else {
        int worst = 0, wi;
        for (wi = 1; wi < nfound; ++wi) {
          if (dists[wi] > dists[worst]) worst = wi;
        }
        if (sqrt(d2) < dists[worst]) {
          candidates[worst] = cl->raw_outputs[(size_t)i * nouts + output_j];
          dists[worst] = sqrt(d2);
        }
      }
    }
    if (NULL != out_variance) {
      if (0 != exact || nfound <= 1) {
        *out_variance = 0;
      }
      else {
        double mean = 0, v = 0;
        for (i = 0; i < nfound; ++i) mean += candidates[i];
        mean /= nfound;
        for (i = 0; i < nfound; ++i) {
          const double d = candidates[i] - mean;
          v += d * d;
        }
        *out_variance = v / nfound;
      }
    }
    if (0 == exact && nfound > 0) {
      if (ndistinct > ndistinct_thresh) {
        double wsum = 0, wavg = 0;
        for (i = 0; i < nfound; ++i) {
          const double wi = (dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30;
          wavg += wi * candidates[i];
          wsum += wi;
        }
        wavg = (wsum > 0.0) ? wavg / wsum : candidates[0];
        if (0 != extrapolate) {
          best_val = wavg;
        }
        else {
          double best_dist = DBL_MAX;
          for (i = 0; i < nc; ++i) {
            const double v = cl->raw_outputs[(size_t)i * nouts + output_j];
            const double d = (v > wavg) ? (v - wavg) : (wavg - v);
            if (d < best_dist) { best_dist = d; best_val = v; }
          }
        }
        if (NULL != confidence) {
          *confidence = 1.0;
        }
      }
      else {
        double best_weight = 0;
        for (i = 0; i < nfound; ++i) {
          double ws = 0;
          int ii;
          for (ii = 0; ii < nfound; ++ii) {
            if (candidates[ii] == candidates[i]) {
              ws += (dists[ii] > 0.0) ? (1.0 / dists[ii]) : 1e30;
            }
          }
          if (ws > best_weight) { best_weight = ws; best_val = candidates[i]; }
        }
        if (NULL != confidence) {
          double total_weight = 0;
          for (i = 0; i < nfound; ++i) {
            total_weight += (dists[i] > 0.0) ? (1.0 / dists[i]) : 1e30;
          }
          *confidence = (total_weight > 0.0) ? best_weight / total_weight : 1.0;
        }
      }
    }
    else if (NULL != confidence) {
      *confidence = 1.0;
    }
  }
  return best_val;
}


LIBXS_API_INLINE double internal_libxs_predict_classify(
  const internal_libxs_predict_cluster_t* cl, const double* kd_pts,
  int nc, int m, const double* inputs, int output_j, int nouts,
  int ndistinct, int extrapolate, int skip_local,
  double* confidence, double* out_variance)
{
  return internal_libxs_predict_classify2(cl, kd_pts, nc, m, inputs,
    output_j, nouts, ndistinct, extrapolate, skip_local,
    NULL, -1, confidence, out_variance);
}


LIBXS_API libxs_predict_t* libxs_predict_create(int ninputs, int noutputs)
{
  libxs_predict_t* model = NULL;
  if (0 < ninputs && 0 < noutputs) {
    model = (libxs_predict_t*)calloc(1, sizeof(libxs_predict_t));
    if (NULL != model) {
      model->ninputs = ninputs;
      model->noutputs = noutputs;
      model->eval_mode = LIBXS_PREDICT_AUTO;
      model->diff_mode = -1;
    }
  }
  return model;
}


LIBXS_API void libxs_predict_destroy(libxs_predict_t* model)
{
  if (NULL != model) {
    int i;
    internal_libxs_predict_free_clusters(model);
    if (NULL != model->entries) {
      for (i = 0; i < model->nentries; ++i) {
        free(model->entries[i].inputs);
        free(model->entries[i].outputs);
      }
      free(model->entries);
    }
    free(model->input_min);
    free(model->input_rng);
    free(model->weights);
    free(model->transforms);
    free(model->ts_buf);
    free(model->decompose_mat);
    free(model->hknn_assignments);
    if (NULL != model->hknn_po_assignments) {
      int oi;
      for (oi = 0; oi < model->noutputs; ++oi) {
        free(model->hknn_po_assignments[oi]);
      }
      free(model->hknn_po_assignments);
    }
    if (NULL != model->hknn_po_clusters) {
      int oi;
      for (oi = 0; oi < model->noutputs; ++oi) {
        if (NULL != model->hknn_po_clusters[oi]) {
          const int nc = (NULL != model->hknn_po_nclusters)
            ? model->hknn_po_nclusters[oi] : 0;
          int ci;
          for (ci = 0; ci < nc; ++ci) {
            internal_libxs_predict_cluster_t* cl =
              &model->hknn_po_clusters[oi][ci];
            free(cl->centroid);
            free(cl->kd_pts);
            free(cl->raw_outputs);
            free(cl->sorted_idx);
            free(cl->sorted_dist);
            free(cl->order);
            free(cl->mode);
            free(cl->ndistinct);
            free(cl->interpolated);
            free(cl->coeffs);
            free(cl->errors);
          }
          free(model->hknn_po_clusters[oi]);
        }
      }
      free(model->hknn_po_clusters);
    }
    free(model->hknn_po_nclusters);
    if (NULL != model->rf) {
      int ti;
      const int total_trees = model->rf->ntrees * model->rf->noutputs;
      for (ti = 0; ti < total_trees; ++ti) free(model->rf->trees[ti].nodes);
      free(model->rf->trees);
      free(model->rf->label_offset);
      free(model->rf);
    }
    free(model);
  }
}


LIBXS_API libxs_lock_t* libxs_predict_lock(libxs_predict_t* model)
{
  return (NULL != model) ? &model->lock : NULL;
}


LIBXS_API void libxs_predict_set_mode(libxs_predict_t* model, int mode)
{
  LIBXS_ASSERT(NULL != model);
  model->eval_mode = mode;
}


LIBXS_API void libxs_predict_set_refine(libxs_predict_t* model, int iterations)
{
  LIBXS_ASSERT(NULL != model);
  model->refine = iterations;
}


LIBXS_API void libxs_predict_set_smooth(libxs_predict_t* model, double amount)
{
  LIBXS_ASSERT(NULL != model);
  model->smooth = amount;
}


LIBXS_API void libxs_predict_set_consistency(
  libxs_predict_t* model, double amount)
{
  LIBXS_ASSERT(NULL != model);
  model->consistency = amount;
}


LIBXS_API void libxs_predict_set_weights(libxs_predict_t* model, const double weights[])
{
  LIBXS_ASSERT(NULL != model);
  if (NULL == weights) {
    free(model->weights);
    model->weights = NULL;
  }
  else {
    const int m = model->ninputs;
    if (NULL == model->weights) {
      model->weights = (double*)malloc((size_t)m * sizeof(double));
    }
    if (NULL != model->weights) {
      memcpy(model->weights, weights, (size_t)m * sizeof(double));
    }
  }
}


LIBXS_API void libxs_predict_set_transform(libxs_predict_t* model, int output, int transform)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL == model->transforms) {
    model->transforms = (int*)calloc((size_t)model->noutputs, sizeof(int));
  }
  if (NULL != model->transforms) {
    if (0 > output) {
      int j;
      for (j = 0; j < model->noutputs; ++j) model->transforms[j] = transform;
    }
    else if (output < model->noutputs) {
      model->transforms[output] = transform;
    }
  }
}


LIBXS_API_INLINE double internal_libxs_predict_fwd(int transform, double v)
{
  double result = v;
  switch (transform) {
    case LIBXS_PREDICT_LOG: result = log(v + 1.0); break;
    case LIBXS_PREDICT_SQRT: result = sqrt(v > 0 ? v : 0); break;
    default: break;
  }
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_inv(int transform, double v)
{
  double result = v;
  switch (transform) {
    case LIBXS_PREDICT_LOG: result = exp(v) - 1.0; break;
    case LIBXS_PREDICT_SQRT: result = v * v; break;
    default: break;
  }
  return result;
}


LIBXS_API void libxs_predict_set_series(libxs_predict_t* model, int nseries, int window)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 < nseries && 0 < window
    && model->ninputs == nseries * window)
  {
    model->nseries = nseries;
    model->window = window;
  }
}


LIBXS_API void libxs_predict_set_target(libxs_predict_t* model, int target)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model && 0 <= target && target < model->nseries) {
    model->target = target;
  }
}


LIBXS_API void libxs_predict_set_decompose(libxs_predict_t* model, int decompose)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->decompose = decompose;
  }
}


LIBXS_API void libxs_predict_set_diff(libxs_predict_t* model, int order)
{
  LIBXS_ASSERT(NULL != model);
  if (NULL != model) {
    model->diff_mode = order;
  }
}



LIBXS_API_INLINE void internal_libxs_predict_decompose_apply(
  const libxs_predict_t* model, const double* raw, double* out)
{
  const int m = model->ninputs;
  if (NULL != model->decompose_mat) {
    const double alpha = 1.0, beta = 0.0;
    const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
      LIBXS_DATATYPE_F64, 'N', 'N', m, 1, m, m, m, m,
      &alpha, &beta, NULL);
    libxs_gemm_call(gemm, model->decompose_mat, raw, out);
    libxs_gemm_release(gemm);
  }
  else {
    memcpy(out, raw, (size_t)m * sizeof(double));
  }
}


LIBXS_API_INLINE void internal_libxs_predict_decompose_inverse(
  const libxs_predict_t* model, const double* modes, double* raw)
{
  const int m = model->ninputs;
  if (NULL != model->decompose_mat) {
    const double alpha = 1.0, beta = 0.0;
    const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
      LIBXS_DATATYPE_F64, 'T', 'N', m, 1, m, m, m, m,
      &alpha, &beta, NULL);
    libxs_gemm_call(gemm, model->decompose_mat, modes, raw);
    libxs_gemm_release(gemm);
  }
  else {
    memcpy(raw, modes, (size_t)m * sizeof(double));
  }
}


LIBXS_API_INLINE void internal_libxs_predict_pca_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const size_t msz = (size_t)m * (size_t)m;
  int pool_mean = 0, pool_cov = 0, pool_evec = 0, pool_eval = 0;
  double* mean = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), pool_mean);
  double* cov = (double*)LIBXS_PREDICT_MALLOC(msz * sizeof(double), pool_cov);
  double* evec = (double*)LIBXS_PREDICT_MALLOC(msz * sizeof(double), pool_evec);
  double* eval = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), pool_eval);
  if (NULL != mean && NULL != cov && NULL != evec && NULL != eval) {
  memset(mean, 0, (size_t)m * sizeof(double));
  memset(cov, 0, msz * sizeof(double));
  { int i, j, k;
    for (i = 0; i < p; ++i) {
      const double* inp = model->entries[i].inputs;
      for (j = 0; j < m; ++j) mean[j] += inp[j];
    }
    for (j = 0; j < m; ++j) mean[j] /= p;
    for (i = 0; i < p; ++i) {
      const double* inp = model->entries[i].inputs;
      for (j = 0; j < m; ++j) {
        const double dj = inp[j] - mean[j];
        for (k = j; k < m; ++k) {
          cov[j * m + k] += dj * (inp[k] - mean[k]);
        }
      }
    }
    for (j = 0; j < m; ++j) {
      for (k = j; k < m; ++k) {
        cov[j * m + k] /= p;
        cov[k * m + j] = cov[j * m + k];
      }
    }
    for (j = 0; j < m; ++j) {
      for (k = 0; k < m; ++k) evec[j * m + k] = (j == k) ? 1.0 : 0.0;
    }
    { int iter;
      for (iter = 0; iter < 100 * m; ++iter) {
        int pi = 0, qi = 1;
        double maxoff = 0;
        for (j = 0; j < m; ++j) {
          for (k = j + 1; k < m; ++k) {
            const double a = cov[j * m + k] < 0 ? -cov[j * m + k] : cov[j * m + k];
            if (a > maxoff) { maxoff = a; pi = j; qi = k; }
          }
        }
        if (maxoff < 1e-12) break;
        { const double app = cov[pi * m + pi], aqq = cov[qi * m + qi];
          const double apq = cov[pi * m + qi];
          const double tau = (aqq - app) / (2.0 * apq);
          const double t = (tau >= 0 ? 1.0 : -1.0)
            / (LIBXS_FABS(tau) + sqrt(1.0 + tau * tau));
          const double c = 1.0 / sqrt(1.0 + t * t);
          const double s = t * c;
          for (k = 0; k < m; ++k) {
            const double ik = cov[k * m + pi], jk = cov[k * m + qi];
            cov[k * m + pi] = c * ik - s * jk;
            cov[k * m + qi] = s * ik + c * jk;
            cov[pi * m + k] = cov[k * m + pi];
            cov[qi * m + k] = cov[k * m + qi];
          }
          cov[pi * m + pi] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
          cov[qi * m + qi] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
          cov[pi * m + qi] = 0;
          cov[qi * m + pi] = 0;
          for (k = 0; k < m; ++k) {
            const double ek = evec[k * m + pi], fk = evec[k * m + qi];
            evec[k * m + pi] = c * ek - s * fk;
            evec[k * m + qi] = s * ek + c * fk;
          }
        }
      }
    }
    for (j = 0; j < m; ++j) eval[j] = cov[j * m + j];
    for (j = 0; j < m - 1; ++j) {
      int best = j;
      for (k = j + 1; k < m; ++k) {
        if (eval[k] > eval[best]) best = k;
      }
      if (best != j) {
        { double tmp = eval[j]; eval[j] = eval[best]; eval[best] = tmp; }
        for (k = 0; k < m; ++k) {
          double tmp = evec[k * m + j];
          evec[k * m + j] = evec[k * m + best];
          evec[k * m + best] = tmp;
        }
      }
    }
    free(model->decompose_mat);
    model->decompose_mat = (double*)malloc(msz * sizeof(double));
    if (NULL != model->decompose_mat) {
      double total_var = 0, cum_var = 0;
      int npc = m;
      for (j = 0; j < m; ++j) total_var += (eval[j] > 0 ? eval[j] : 0);
      for (j = 0; j < m; ++j) {
        for (k = 0; k < m; ++k) {
          model->decompose_mat[j * m + k] = evec[k * m + j];
        }
      }
      for (j = 0; j < m; ++j) {
        cum_var += (eval[j] > 0 ? eval[j] : 0);
        if (cum_var >= 0.95 * total_var && npc == m) npc = j + 1;
      }
      if (npc < m && LIBXS_PREDICT_PCA == model->decompose) {
        if (NULL == model->weights) {
          model->weights = (double*)malloc((size_t)m * sizeof(double));
        }
        if (NULL != model->weights) {
          for (j = 0; j < m; ++j) model->weights[j] = (j < npc) ? 1.0 : 0.0;
        }
      }
      { int xmat_pool = 0, ymat_pool = 0;
        double* xmat = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)p * (size_t)m * sizeof(double), xmat_pool);
        double* ymat = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)p * (size_t)m * sizeof(double), ymat_pool);
        if (NULL != xmat && NULL != ymat) {
          { const double alpha = 1.0, beta = 0.0;
            const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
              LIBXS_DATATYPE_F64, 'N', 'N', m, p, m, m, m, m,
              &alpha, &beta, NULL);
            for (i = 0; i < p; ++i) {
              memcpy(xmat + (size_t)i * m, model->entries[i].inputs,
                (size_t)m * sizeof(double));
            }
            libxs_gemm_call(gemm, model->decompose_mat, xmat, ymat);
            libxs_gemm_release(gemm);
          }
          for (i = 0; i < p; ++i) {
            memcpy(model->entries[i].inputs, ymat + (size_t)i * m,
              (size_t)m * sizeof(double));
          }
        }
        LIBXS_PREDICT_FREE(ymat, ymat_pool);
        LIBXS_PREDICT_FREE(xmat, xmat_pool);
      }
    }
  }
  }
  LIBXS_PREDICT_FREE(mean, pool_mean);
  LIBXS_PREDICT_FREE(cov, pool_cov);
  LIBXS_PREDICT_FREE(evec, pool_evec);
  LIBXS_PREDICT_FREE(eval, pool_eval);
}


LIBXS_API_INLINE void internal_libxs_predict_fisher_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  int nclasses = 0, j, i, ci;
  int class_id[128], class_count[128];
  double class_mean[128][128], class_var[128][128];
  double scores[128], sorted_scores[128], thr;
  LIBXS_ASSERT(m <= 128);
  memset(class_count, 0, sizeof(class_count));
  memset(class_mean, 0, sizeof(class_mean));
  memset(class_var, 0, sizeof(class_var));
  for (i = 0; i < p; ++i) {
    const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
    int found = 0;
    for (ci = 0; ci < nclasses; ++ci) {
      if (class_id[ci] == label) { found = 1; break; }
    }
    if (0 == found && nclasses < 128) { class_id[nclasses] = label; ci = nclasses++; }
    if (ci < 128) {
      ++class_count[ci];
      for (j = 0; j < m; ++j) class_mean[ci][j] += model->entries[i].inputs[j];
    }
  }
  for (ci = 0; ci < nclasses; ++ci) {
    if (0 < class_count[ci]) {
      for (j = 0; j < m; ++j) class_mean[ci][j] /= class_count[ci];
    }
  }
  for (i = 0; i < p; ++i) {
    const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
    for (ci = 0; ci < nclasses; ++ci) {
      if (class_id[ci] == label) {
        for (j = 0; j < m; ++j) {
          double d = model->entries[i].inputs[j] - class_mean[ci][j];
          class_var[ci][j] += d * d;
        }
        break;
      }
    }
  }
  if (nclasses >= 2 && 1 == model->noutputs) {
    for (j = 0; j < m; ++j) {
      double between = 0, within = 0, grand_mean = 0;
      int total_n = 0;
      for (ci = 0; ci < nclasses; ++ci) {
        if (0 < class_count[ci]) {
          grand_mean += class_mean[ci][j] * class_count[ci];
          total_n += class_count[ci];
          within += class_var[ci][j];
        }
      }
      if (0 < total_n) grand_mean /= total_n;
      for (ci = 0; ci < nclasses; ++ci) {
        if (0 < class_count[ci]) {
          double d = class_mean[ci][j] - grand_mean;
          between += class_count[ci] * d * d;
        }
      }
      scores[j] = (within > 0) ? between / within : 0.0;
    }
    memcpy(sorted_scores, scores, (size_t)m * sizeof(double));
    libxs_sort(sorted_scores, m, sizeof(double), libxs_cmp_f64, NULL);
    thr = sorted_scores[m / 2];
    if (NULL == model->weights) {
      model->weights = (double*)malloc((size_t)m * sizeof(double));
    }
    if (NULL != model->weights) {
      for (j = 0; j < m; ++j) {
        model->weights[j] = (scores[j] >= thr) ? sqrt(scores[j]) : 0.0;
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_setdiff_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int m = model->ninputs;
  const int n = model->noutputs;
  int nclasses = 0, j, i, a, b;
  int class_id[128], class_count[128];
  double scores[128], sorted_scores[128], thr;
  LIBXS_ASSERT(m <= 128);
  memset(class_count, 0, sizeof(class_count));
  memset(scores, 0, sizeof(scores));
  for (i = 0; i < p; ++i) {
    const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
    int found = 0, ci;
    for (ci = 0; ci < nclasses; ++ci) {
      if (class_id[ci] == label) { ++class_count[ci]; found = 1; break; }
    }
    if (0 == found && nclasses < 128) {
      class_id[nclasses] = label;
      class_count[nclasses] = 1;
      ++nclasses;
    }
  }
  if (nclasses >= 2 && 1 == n) {
  for (j = 0; j < m; ++j) {
    double score = 0;
    int npairs = 0;
    { double fmin = model->entries[0].inputs[j];
      double fmax = fmin, frange;
      for (i = 1; i < p; ++i) {
        const double v = model->entries[i].inputs[j];
        if (v < fmin) fmin = v;
        if (v > fmax) fmax = v;
      }
      frange = fmax - fmin;
      if (frange <= 0) frange = 1.0;
      for (a = 0; a < nclasses; ++a) {
        for (b = a + 1; b < nclasses; ++b) {
          int ca_pool = 0, cb_pool = 0;
          double* va = (double*)LIBXS_PREDICT_MALLOC(
            (size_t)class_count[a] * sizeof(double), ca_pool);
          double* vb = (double*)LIBXS_PREDICT_MALLOC(
            (size_t)class_count[b] * sizeof(double), cb_pool);
          if (NULL != va && NULL != vb) {
            int na = 0, nb = 0;
            for (i = 0; i < p; ++i) {
              const int label = LIBXS_ROUNDX(int, model->entries[i].outputs[0]);
              if (label == class_id[a]) va[na++] = model->entries[i].inputs[j];
              else if (label == class_id[b]) vb[nb++] = model->entries[i].inputs[j];
            }
            { const int sd = libxs_setdiff(LIBXS_DATATYPE_F64,
                va, na, vb, nb, frange * 0.05);
              score += (double)sd / LIBXS_MAX(na, nb);
            }
            ++npairs;
          }
          LIBXS_PREDICT_FREE(vb, cb_pool);
          LIBXS_PREDICT_FREE(va, ca_pool);
        }
      }
    }
      scores[j] = (npairs > 0) ? score / npairs : 0.0;
    }
    memcpy(sorted_scores, scores, (size_t)m * sizeof(double));
    libxs_sort(sorted_scores, m, sizeof(double), libxs_cmp_f64, NULL);
    thr = sorted_scores[m / 2];
    if (NULL == model->weights) {
      model->weights = (double*)malloc((size_t)m * sizeof(double));
    }
    if (NULL != model->weights) {
      for (j = 0; j < m; ++j) {
        model->weights[j] = (scores[j] >= thr) ? scores[j] : 0.0;
      }
    }
  }
}


#include "libxs_predict_rf.h"
#include "libxs_predict_hknn.h"


LIBXS_API_INLINE int internal_libxs_predict_grow(libxs_predict_t* model)
{
  int result = EXIT_SUCCESS;
  if (model->nentries >= model->capacity) {
    const int newcap = (0 < model->capacity) ? (model->capacity * 2) : 64;
    internal_libxs_predict_entry_t* ne = (internal_libxs_predict_entry_t*)realloc(
      model->entries, (size_t)newcap * sizeof(internal_libxs_predict_entry_t));
    if (NULL != ne) {
      memset(ne + model->capacity, 0,
        (size_t)(newcap - model->capacity) * sizeof(internal_libxs_predict_entry_t));
      model->entries = ne;
      model->capacity = newcap;
    }
    else result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_ts_diff_order(
  const libxs_predict_t* model)
{
  const int s = model->nseries;
  const int n = model->nts;
  const int tgt = model->target;
  const size_t shape = (size_t)n;
  const size_t stride = (size_t)s;
  libxs_fprint_t fp;
  int result = 0;
  if (n >= 4 && EXIT_SUCCESS == libxs_fprint(&fp, LIBXS_DATATYPE_F64,
    model->ts_buf + tgt, 1, &shape, &stride,
    LIBXS_MIN(3, n - 1), 0, 1, 0))
  {
    const double decay = libxs_fprint_decay(&fp);
    if (decay == decay && decay < 1.0) {
      result = 1;
    }
  }
  return result;
}


LIBXS_API_INLINE void internal_libxs_predict_ts_diff_apply(
  double* buf, int n, int s, int d)
{
  int dd, i, si;
  for (dd = 0; dd < d; ++dd) {
    for (i = 0; i < n - 1 - dd; ++i) {
      for (si = 0; si < s; ++si) {
        buf[i * s + si] = buf[(i + 1) * s + si] - buf[i * s + si];
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_ts_expand(libxs_predict_t* model)
{
  const int s = model->nseries;
  const int h = model->noutputs;
  int nts = model->nts, diff_d, w, m, nwindows;
  int raw_pool = 0;
  double* raw;
  int t;
  if (model->diff_mode > 0) {
    model->diff_order = model->diff_mode;
  }
  else if (0 == model->diff_mode) {
    model->diff_order = internal_libxs_predict_ts_diff_order(model);
  }
  diff_d = model->diff_order;
  if (diff_d > 0) {
    internal_libxs_predict_ts_diff_apply(model->ts_buf, nts, s, diff_d);
    nts -= diff_d;
    model->nts = nts;
  }
  w = model->window - diff_d;
  m = s * w;
  if (diff_d > 0) {
    model->ninputs = m;
  }
  nwindows = nts - w - h + 1;
  raw = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), raw_pool);
  if (NULL != raw && nwindows > 0) {
  for (t = 0; t < nwindows; ++t) {
    double* inputs = (double*)malloc((size_t)m * sizeof(double));
    double* outputs = (double*)malloc((size_t)h * sizeof(double));
    if (NULL != inputs && NULL != outputs) {
      int si, i;
      for (si = 0; si < s; ++si) {
        for (i = 0; i < w; ++i) {
          raw[si * w + i] = model->ts_buf[(t + i) * s + si];
        }
      }
      if (LIBXS_PREDICT_RAW != model->decompose && s >= 2) {
        internal_libxs_predict_decompose_apply(model, raw, inputs);
      }
      else {
        memcpy(inputs, raw, (size_t)m * sizeof(double));
      }
      for (i = 0; i < h; ++i) {
        outputs[i] = model->ts_buf[(t + w + i) * s + model->target];
      }
      if (EXIT_SUCCESS == internal_libxs_predict_grow(model)) {
        internal_libxs_predict_entry_t* e = &model->entries[model->nentries];
        e->inputs = inputs;
        if (NULL != model->transforms) {
          int j;
          for (j = 0; j < h; ++j) {
            outputs[j] = internal_libxs_predict_fwd(model->transforms[j], outputs[j]);
          }
        }
        e->outputs = outputs;
        ++model->nentries;
      }
      else {
        free(inputs);
        free(outputs);
      }
    }
    else {
      free(inputs);
      free(outputs);
    }
  }
  }
  LIBXS_PREDICT_FREE(raw, raw_pool);
}


LIBXS_API int libxs_predict_push(
  libxs_lock_t* lock, libxs_predict_t* model, const double inputs[], const double outputs[])
{
  int result = EXIT_SUCCESS;
  if (NULL == model || NULL == inputs) {
    result = EXIT_FAILURE;
  }
  else if (NULL == outputs && 0 < model->nseries) {
    const int s = model->nseries;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (model->nts >= model->ts_capacity) {
      const int newcap = (0 < model->ts_capacity) ? (model->ts_capacity * 2) : 256;
      double* nb = (double*)realloc(model->ts_buf, (size_t)newcap * (size_t)s * sizeof(double));
      if (NULL != nb) {
        model->ts_buf = nb;
        model->ts_capacity = newcap;
      }
      else result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      memcpy(model->ts_buf + (size_t)model->nts * s, inputs, (size_t)s * sizeof(double));
      ++model->nts;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  else if (NULL == outputs) {
    result = EXIT_FAILURE;
  }
  else {
    const int m = model->ninputs, n = model->noutputs;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    result = internal_libxs_predict_grow(model);
    if (EXIT_SUCCESS == result) {
      internal_libxs_predict_entry_t* e = &model->entries[model->nentries];
      e->inputs = (double*)malloc((size_t)m * sizeof(double));
      e->outputs = (double*)malloc((size_t)n * sizeof(double));
      if (NULL != e->inputs && NULL != e->outputs) {
        memcpy(e->inputs, inputs, (size_t)m * sizeof(double));
        if (NULL != model->transforms) {
          int j;
          for (j = 0; j < n; ++j) {
            e->outputs[j] = internal_libxs_predict_fwd(model->transforms[j], outputs[j]);
          }
        }
        else {
          memcpy(e->outputs, outputs, (size_t)n * sizeof(double));
        }
        ++model->nentries;
      }
      else {
        free(e->inputs);
        free(e->outputs);
        e->inputs = NULL;
        e->outputs = NULL;
        result = EXIT_FAILURE;
      }
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  return result;
}


LIBXS_API_INLINE double internal_libxs_predict_order_fn(
  double x, const void* data)
{
  const internal_libxs_predict_order_ctx_t* ctx =
    (const internal_libxs_predict_order_ctx_t*)data;
  const int ord = LIBXS_MAX(LIBXS_ROUNDX(int, x), 1);
  double total_err = 1e30;
  if (EXIT_SUCCESS == libxs_predict_build(ctx->model, ctx->nclusters, ord, 0)) {
    const int p = ctx->model->nentries;
    const int n = ctx->model->noutputs;
    const int saved_decompose = ctx->model->decompose;
    int i, j;
    ctx->model->decompose = LIBXS_PREDICT_RAW;
    total_err = 0;
    for (i = 0; i < p; ++i) {
      double outputs[128];
      libxs_predict_eval(NULL, ctx->model,
        ctx->model->entries[i].inputs, outputs, NULL, 1);
      for (j = 0; j < n; ++j) {
        total_err += LIBXS_DELTA(outputs[j], ctx->model->entries[i].outputs[j]);
      }
    }
    ctx->model->decompose = saved_decompose;
  }
  return total_err;
}


#include "libxs_predict_compress.h"


LIBXS_API int libxs_predict_build(libxs_predict_t* model,
  int nclusters, int order, double quality)
{
  int result = EXIT_SUCCESS;
  if (NULL != model && 0 < model->nts && 0 == model->nentries) {
    internal_libxs_predict_ts_expand(model);
  }
  if (NULL != model && 0 < model->nentries && NULL == model->decompose_mat
    && (LIBXS_PREDICT_PCA == model->decompose
      || (LIBXS_PREDICT_SPREAD == model->decompose && model->nseries >= 2)))
  {
    internal_libxs_predict_pca_build(model);
  }
  if (NULL != model && 0 < model->nentries && NULL == model->weights) {
    if (LIBXS_PREDICT_SETDIFF == model->decompose) {
      internal_libxs_predict_setdiff_build(model);
    }
    else if (LIBXS_PREDICT_FISHER == model->decompose) {
      internal_libxs_predict_fisher_build(model);
    }
  }
  if (NULL != model && 0 < model->nentries
    && LIBXS_PREDICT_HKNN == model->decompose
    && NULL == model->hknn_assignments)
  {
    model->hknn_assignments = (int*)calloc((size_t)model->nentries, sizeof(int));
    if (NULL != model->hknn_assignments) {
      model->hknn_nclusters = 0;
      internal_libxs_predict_hknn_partition(model, &model->hknn_nclusters);
      if (model->hknn_nclusters < 1) model->hknn_nclusters = 1;
    }
  }
  if (NULL != model && 0 < model->nentries
    && LIBXS_PREDICT_RF == model->decompose && NULL == model->rf)
  {
    internal_libxs_predict_rf_build(model);
    if (model->phase <= 0) {
      internal_libxs_predict_rf_build_tasks(model, 0, 1);
    }
  }
  if (NULL == model || 0 >= model->nentries) {
    result = EXIT_FAILURE;
  }
  else if (order <= 0) {
    internal_libxs_predict_order_ctx_t ctx;
    const int max_ord = (order < 0) ? -order : LIBXS_FPRINT_MAXORDER;
    int best_ord = 1, ord;
    double best_err = 1e30;
    ctx.model = model;
    ctx.nclusters = nclusters;
    for (ord = 1; ord <= max_ord; ++ord) {
      const double err = internal_libxs_predict_order_fn((double)ord, &ctx);
      if (err < best_err) { best_err = err; best_ord = ord; }
    }
    model->iterations = max_ord;
    result = libxs_predict_build(model, nclusters, best_ord, quality);
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    int c, i;
    if (order > LIBXS_FPRINT_MAXORDER) order = LIBXS_FPRINT_MAXORDER;
    model->order = order;
    model->quality = quality;
    internal_libxs_predict_free_clusters(model);
    free(model->input_min); free(model->input_rng);
    model->input_min = (double*)malloc((size_t)m * sizeof(double));
    model->input_rng = (double*)malloc((size_t)m * sizeof(double));
    if (NULL != model->input_min && NULL != model->input_rng) {
      int j;
      for (j = 0; j < m; ++j) {
        model->input_min[j] = model->entries[0].inputs[j];
        model->input_rng[j] = model->entries[0].inputs[j];
      }
      for (i = 1; i < p; ++i) {
        for (j = 0; j < m; ++j) {
          if (model->entries[i].inputs[j] < model->input_min[j]) {
            model->input_min[j] = model->entries[i].inputs[j];
          }
          if (model->entries[i].inputs[j] > model->input_rng[j]) {
            model->input_rng[j] = model->entries[i].inputs[j];
          }
        }
      }
      for (j = 0; j < m; ++j) {
        model->input_rng[j] -= model->input_min[j];
      }
    }
    if (0 >= nclusters) {
      nclusters = (int)(sqrt((double)p) + 0.5);
      if (nclusters < 1) nclusters = 1;
    }
    if (nclusters > p) nclusters = p;
    model->assignments = (int*)calloc((size_t)p, sizeof(int));
    model->eval_buf = (double*)malloc((size_t)n * 4 * sizeof(double) + (size_t)n * sizeof(int));
    if (NULL == model->assignments || NULL == model->eval_buf) {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && LIBXS_PREDICT_HKNN == model->decompose
      && NULL != model->hknn_assignments && model->hknn_nclusters > 0)
    {
      memcpy(model->assignments, model->hknn_assignments,
        (size_t)p * sizeof(int));
      nclusters = model->hknn_nclusters;
    }
    model->clusters = (internal_libxs_predict_cluster_t*)calloc(
      (size_t)nclusters, sizeof(internal_libxs_predict_cluster_t));
    if (NULL == model->clusters) {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      model->nclusters = nclusters;
      for (c = 0; c < nclusters && EXIT_SUCCESS == result; ++c) {
        model->clusters[c].centroid = (double*)malloc((size_t)m * sizeof(double));
        if (NULL == model->clusters[c].centroid) result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      if (LIBXS_PREDICT_HKNN == model->decompose) {
        internal_libxs_predict_hknn_centroids(model, nclusters);
        internal_libxs_predict_hknn_refine(model, nclusters);
      }
      else {
        internal_libxs_predict_kmeans(model, nclusters);
      }
      for (i = 0; i < p; ++i) {
        ++model->clusters[model->assignments[i]].nentries;
      }
      { int dst = 0, has_empty = 0;
        for (c = 0; c < nclusters; ++c) {
          if (0 >= model->clusters[c].nentries) { has_empty = 1; break; }
        }
        if (0 != has_empty) {
          for (i = 0; i < p; ++i) {
            int gap = 0, a = model->assignments[i];
            for (c = 0; c < a; ++c) {
              if (0 >= model->clusters[c].nentries) ++gap;
            }
            model->assignments[i] = a - gap;
          }
          for (c = 0; c < nclusters; ++c) {
            if (model->clusters[c].nentries > 0) {
              if (dst != c) {
                model->clusters[dst] = model->clusters[c];
                memset(&model->clusters[c], 0,
                  sizeof(internal_libxs_predict_cluster_t));
              }
              ++dst;
            }
            else {
              free(model->clusters[c].centroid);
              memset(&model->clusters[c], 0,
                sizeof(internal_libxs_predict_cluster_t));
            }
          }
          nclusters = dst;
          model->nclusters = nclusters;
        }
      }
    }
    for (c = 0; c < nclusters && EXIT_SUCCESS == result; ++c) {
      internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      const int nc = cl->nentries;
      int j, k, maxorder;
      if (0 >= nc) continue;
      cl->sorted_idx = (int*)malloc((size_t)nc * sizeof(int));
      cl->sorted_dist = (double*)malloc((size_t)nc * sizeof(double));
      cl->order = (int*)malloc((size_t)n * sizeof(int));
      cl->interpolated = (int*)malloc((size_t)n * sizeof(int));
      cl->mode = (int*)malloc((size_t)n * sizeof(int));
      cl->ndistinct = (int*)malloc((size_t)n * sizeof(int));
      if (NULL == cl->sorted_idx || NULL == cl->sorted_dist
        || NULL == cl->order || NULL == cl->interpolated
        || NULL == cl->mode || NULL == cl->ndistinct)
      {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        int pool_inmat = 0, pool_perm = 0, pool_map = 0;
        double *const inmat = (double*)LIBXS_PREDICT_MALLOC((size_t)nc * (size_t)m * sizeof(double), pool_inmat);
        int *const sort_perm = (int*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(int), pool_perm);
        int *const entry_map = (int*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(int), pool_map);
        if (NULL == entry_map || NULL == inmat || NULL == sort_perm) {
          result = EXIT_FAILURE;
        }
        else {
          int ki = 0;
          for (i = 0; i < p; ++i) {
            if (model->assignments[i] == c) {
              entry_map[ki] = i;
              for (k = 0; k < m; ++k) {
                inmat[(size_t)k * nc + ki] = model->entries[i].inputs[k];
              }
              ++ki;
            }
          }
          libxs_sort_smooth(LIBXS_SORT_HILBERT, nc, m, inmat, nc,
            LIBXS_DATATYPE_F64, sort_perm);
          for (k = 0; k < nc; ++k) {
            cl->sorted_idx[k] = entry_map[sort_perm[k]];
            cl->sorted_dist[k] = sqrt(
              libxs_dist2(model->entries[cl->sorted_idx[k]].inputs,
                cl->centroid, m));
          }
          cl->dmax = 0;
          for (k = 0; k < nc; ++k) {
            if (cl->sorted_dist[k] > cl->dmax) cl->dmax = cl->sorted_dist[k];
          }
          if (cl->dmax <= 0.0) cl->dmax = 1.0;
          cl->kd_pts = (double*)malloc((size_t)nc * (size_t)m * sizeof(double));
          if (NULL != cl->kd_pts) {
            for (k = 0; k < nc; ++k) {
              internal_libxs_predict_normalize(model,
                model->entries[cl->sorted_idx[k]].inputs,
                cl->kd_pts + (size_t)k * m);
            }
          }
        }
        LIBXS_PREDICT_FREE(sort_perm, pool_perm);
        LIBXS_PREDICT_FREE(entry_map, pool_map);
        LIBXS_PREDICT_FREE(inmat, pool_inmat);
      }
      if (EXIT_SUCCESS == result) {
        maxorder = LIBXS_MIN(nc - 1, order);
        maxorder = LIBXS_MIN(maxorder, LIBXS_FPRINT_MAXORDER);
        if (maxorder < 1) maxorder = 1;
        cl->maxorder = maxorder;
        cl->coeffs = (double*)calloc((size_t)n * (size_t)(maxorder + 1), sizeof(double));
        cl->errors = (double*)calloc((size_t)n, sizeof(double));
        cl->raw_outputs = (double*)malloc((size_t)nc * (size_t)n * sizeof(double));
        if (NULL == cl->coeffs || NULL == cl->errors || NULL == cl->raw_outputs) {
          result = EXIT_FAILURE;
        }
        else {
          for (k = 0; k < nc; ++k) {
            for (j = 0; j < n; ++j) {
              cl->raw_outputs[(size_t)k * n + j] =
                model->entries[cl->sorted_idx[k]].outputs[j];
            }
          }
        }
      }
      if (EXIT_SUCCESS == result) {
        internal_libxs_predict_cluster_refit(cl, n, 1);
      }
    }
    if (EXIT_SUCCESS == result) {
      model->built = 1;
      if (model->smooth < 0) {
        int nsmooth = 0, ntotal_modes = 0, j;
        for (c = 0; c < nclusters; ++c) {
          const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
          if (NULL != cl->mode) {
            for (j = 0; j < n; ++j) {
              if (0 == cl->mode[j]) ++nsmooth;
              ++ntotal_modes;
            }
          }
        }
        model->smooth = (ntotal_modes > 0)
          ? 0.5 * (double)nsmooth / ntotal_modes : 0.0;
      }
      if (LIBXS_PREDICT_HKNN == model->decompose && n > 1
        && NULL == model->hknn_po_clusters)
      {
        internal_libxs_predict_hknn_build_po(model);
      }
      if (quality > 0 && NULL == model->rf
        && NULL != model->entries && NULL != model->assignments)
      {
        internal_libxs_predict_compress(model, nclusters, order, quality);
      }
    }
    else {
      internal_libxs_predict_free_clusters(model);
    }
  }
  return result;
}


LIBXS_API int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  double quality, int tid, int ntasks)
{
  int result;
  LIBXS_ASSERT(NULL != model);
  if (0 == tid) {
    model->phase = 1;
    result = libxs_predict_build(model, nclusters, order, quality);
    model->phase = -1;
  }
  else {
    LIBXS_SYNC_CYCLE_EQ(&model->phase, -1, LIBXS_SYNC_NPAUSE);
    result = (0 != model->built) ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && NULL != model->rf) {
    internal_libxs_predict_rf_build_tasks(model, tid, ntasks);
  }
  LIBXS_UNUSED(lock);
  return result;
}


LIBXS_API void libxs_predict_eval(libxs_lock_t* lock, const libxs_predict_t* model,
  const double inputs[], double outputs[], libxs_predict_info_t* info, int nblend)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != inputs);
  {
    const int m = model->ninputs, n = model->noutputs;
    const int mode = model->eval_mode;
    const int diff_d = (model->diff_mode >= 0) ? model->diff_order : 0;
    const int force_classify = (0 != (mode & LIBXS_PREDICT_CLASSIFY)) ? 1 : 0;
    const int force_interp = (0 != (mode & LIBXS_PREDICT_INTERPOLATE)) ? 1 : 0;
    const int extrapolate_mode = (0 != (mode & LIBXS_PREDICT_TEMPORAL)) ? 1 : 0;
    const double* raw_inputs = inputs;
    int extrapolate = 0;
    int norm_pool = 0, decomp_pool = 0, diff_pool = 0;
    double* decomp_inputs = NULL;
    double* diff_inputs = NULL;
    double* norm_inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), norm_pool);
    double local_buf[256];
    double *vals, *errs, *conf, *var, best_dist;
    int *rels, c, j, best_c = 0;
    if (diff_d > 0 && model->nseries > 0) {
      const int raw_w = model->window;
      const int s = model->nseries;
      const int raw_m = s * raw_w;
      const int dw = raw_w - diff_d;
      int i, si, dd;
      diff_inputs = (double*)LIBXS_PREDICT_MALLOC(
        (size_t)raw_m * sizeof(double), diff_pool);
      if (NULL != diff_inputs) {
        memcpy(diff_inputs, inputs, (size_t)raw_m * sizeof(double));
        for (dd = 0; dd < diff_d; ++dd) {
          const int len = raw_w - dd;
          for (si = 0; si < s; ++si) {
            for (i = 0; i < len - 1; ++i) {
              diff_inputs[si * raw_w + i] =
                diff_inputs[si * raw_w + i + 1] - diff_inputs[si * raw_w + i];
            }
          }
        }
        for (si = 1; si < s; ++si) {
          for (i = 0; i < dw; ++i) {
            diff_inputs[si * dw + i] = diff_inputs[si * raw_w + i];
          }
        }
        inputs = diff_inputs;
      }
    }
    if (LIBXS_PREDICT_RAW != model->decompose
      && (model->nseries >= 2 || NULL != model->decompose_mat)) {
      decomp_inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), decomp_pool);
      internal_libxs_predict_decompose_apply(model, inputs, decomp_inputs);
      inputs = decomp_inputs;
    }
    if (0 != extrapolate_mode && model->nseries > 0) {
      extrapolate = 1;
    }
    else if (model->nseries > 0
      && NULL != model->input_min && NULL != model->input_rng)
    {
      for (j = 0; j < m && 0 == extrapolate; ++j) {
        if (inputs[j] < model->input_min[j]
          || (model->input_rng[j] > 0
            && inputs[j] > model->input_min[j] + model->input_rng[j]))
        {
          extrapolate = 1;
        }
      }
    }
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    internal_libxs_predict_normalize(model, inputs, norm_inputs);
    if (NULL == lock && NULL == info && NULL != outputs && n * 4 + n <= 256) {
      vals = local_buf;
    }
    else {
      vals = model->eval_buf;
    }
    errs = vals + n;
    conf = errs + n;
    var = conf + n;
    rels = (int*)(var + n);
    if (nblend < 0) nblend = 0;
    if (nblend > model->nclusters) nblend = model->nclusters;
    best_dist = libxs_dist2(norm_inputs, model->clusters[0].centroid, m);
    for (c = 1; c < model->nclusters; ++c) {
      const double d = libxs_dist2(norm_inputs, model->clusters[c].centroid, m);
      if (d < best_dist && model->clusters[c].nentries > 0) {
        best_dist = d; best_c = c;
      }
    }
    if (model->clusters[best_c].nentries <= 0) {
      for (c = 0; c < model->nclusters; ++c) {
        if (model->clusters[c].nentries > 0) { best_c = c; break; }
      }
    }
    if (model->clusters[best_c].nentries <= 0) {
      nblend = 0;
    }
    else if (NULL != model->rf) {
      for (j = 0; j < n; ++j) {
        double rf_conf = 0;
        const int rf_label = internal_libxs_predict_rf_eval_output(
          model->rf, j, inputs, &rf_conf);
        vals[j] = (double)rf_label;
        conf[j] = rf_conf;
        var[j] = 0;
        errs[j] = 0;
        rels[j] = 0;
      }
      if (NULL != info) {
        info->cluster = -1;
        info->distance = 0;
      }
      nblend = 0;
    }
    else if (nblend <= 1) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[best_c];
      const int nearest = (int)internal_libxs_predict_position(model, cl, norm_inputs);
      for (j = 0; j < n; ++j) {
        { const int use_classify = (0 != force_classify)
            ? 1 : ((0 != force_interp) ? 0 : cl->mode[j]);
          if (0 != use_classify && NULL != model->hknn_po_clusters
            && NULL != model->hknn_po_clusters[j]
            && NULL != model->hknn_po_assignments
            && NULL != model->hknn_po_assignments[j])
          {
            const int nn_entry = cl->sorted_idx[
              (nearest < cl->nentries) ? nearest : 0];
            const int po_c = (nn_entry >= 0 && nn_entry < model->nentries)
              ? model->hknn_po_assignments[j][nn_entry] : 0;
            const internal_libxs_predict_cluster_t* pcl =
              &model->hknn_po_clusters[j][po_c];
            double po_conf = 0, po_var = 0;
            if (pcl->nentries > 0 && NULL != pcl->kd_pts) {
              vals[j] = internal_libxs_predict_classify(
                pcl, pcl->kd_pts, pcl->nentries, m, norm_inputs, 0, 1,
                pcl->ndistinct[0], extrapolate, -1, &po_conf, &po_var);
            }
            else {
              vals[j] = internal_libxs_predict_classify(
                cl, cl->kd_pts, cl->nentries, m, norm_inputs, j, n,
                cl->ndistinct[j], extrapolate, -1, &po_conf, &po_var);
            }
            internal_libxs_predict_classify(
              cl, cl->kd_pts, cl->nentries, m, norm_inputs, j, n,
              cl->ndistinct[j], extrapolate, -1, &conf[j], &var[j]);
            errs[j] = 0;
            rels[j] = 0;
          }
          else if (0 != use_classify) {
            vals[j] = internal_libxs_predict_classify(
              cl, cl->kd_pts, cl->nentries, m, norm_inputs, j, n,
              cl->ndistinct[j], extrapolate, -1, &conf[j], &var[j]);
            errs[j] = 0;
            rels[j] = 0;
          }
          else {
            const double t = (0 != extrapolate)
              ? (double)cl->nentries : (double)nearest;
            const int d = cl->order[j];
            const double* cj = cl->coeffs + (size_t)j * (cl->maxorder + 1);
            double val = 0;
            int k;
            for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
            vals[j] = val;
            errs[j] = (NULL != info)
              ? internal_libxs_predict_local_error(model, cl, nearest, j)
              : cl->errors[j];
            conf[j] = 1.0;
            var[j] = 0;
            rels[j] = 1;
          }
        }
      }
      if (model->quality > 0) {
        const double cov_inter = internal_libxs_predict_coverage(
          cl->nentries, model->nentries, model->nclusters);
        const double d_rel = sqrt(best_dist) / cl->dmax;
        const double cov_intra = 1.0 / (1.0 + d_rel);
        const double cov = cov_inter * cov_intra;
        for (j = 0; j < n; ++j) {
          conf[j] = model->quality + cov * (conf[j] - model->quality);
        }
      }
      if (model->nclusters > 1) {
        double avg_conf = 0;
        for (j = 0; j < n; ++j) avg_conf += conf[j];
        avg_conf /= n;
        if (avg_conf < 0.7) nblend = LIBXS_MIN(3, model->nclusters);
        else if (model->smooth > 0) {
          const double radius = sqrt(best_dist) * (1.0 + model->smooth);
          int nb = 1;
          for (c = 0; c < model->nclusters; ++c) {
            if (c != best_c) {
              const double d = sqrt(libxs_dist2(
                norm_inputs, model->clusters[c].centroid, m));
              if (d <= radius) ++nb;
            }
          }
          nblend = LIBXS_MIN(nb, model->nclusters);
        }
      }
      if (nblend <= 1 && NULL != info) {
        info->cluster = best_c;
        info->distance = (cl->dmax > 0.0)
          ? sqrt(best_dist) / cl->dmax : 0.0;
      }
    }
    if (nblend > 1) {
      typedef struct { double dist; int idx; } dc_t;
      const double conf_thr = 0.7;
      int dc_pool = 0;
      dc_t* dists = (dc_t*)LIBXS_PREDICT_MALLOC(
        (size_t)model->nclusters * sizeof(dc_t), dc_pool);
      int b;
      LIBXS_ASSERT(NULL != dists);
      for (c = 0; c < model->nclusters; ++c) {
        dists[c].dist = sqrt(libxs_dist2(norm_inputs, model->clusters[c].centroid, m));
        dists[c].idx = c;
      }
      for (b = 0; b < nblend; ++b) {
        int minj = b;
        for (c = b + 1; c < model->nclusters; ++c) {
          if (dists[c].dist < dists[minj].dist) minj = c;
        }
        if (minj != b) { dc_t tmp = dists[b]; dists[b] = dists[minj]; dists[minj] = tmp; }
      }
      for (j = 0; j < n; ++j) {
        { const internal_libxs_predict_cluster_t* cl_primary = &model->clusters[dists[0].idx];
          const int use_classify = (0 != force_classify)
            ? 1 : ((0 != force_interp) ? 0 : cl_primary->mode[j]);
          double blend_val = 0, blend_conf = 0, blend_var = 0, blend_err = 0;
          double wsum = 0;
          int blend_rel = 0;
          if (conf[j] >= conf_thr && (0.0 >= model->smooth
            || 0 != use_classify)) continue;
          for (b = 0; b < nblend; ++b) {
            const int ci = dists[b].idx;
            const internal_libxs_predict_cluster_t* cl2 = &model->clusters[ci];
            double w = (dists[b].dist > 0) ? (1.0 / dists[b].dist) : 1e30;
            if (0 != extrapolate && cl_primary->fprint_sig > 0) {
              const double sim = 1.0 / (1.0
                + LIBXS_FABS(cl2->fprint_sig - cl_primary->fprint_sig)
                / cl_primary->fprint_sig);
              w *= sim;
            }
            if (0 != use_classify) {
              double cj_conf = 1.0, cj_var = 0;
              const double v = internal_libxs_predict_classify(
                cl2, cl2->kd_pts, cl2->nentries, m, norm_inputs, j, n,
                cl2->ndistinct[j], extrapolate, -1, &cj_conf, &cj_var);
              blend_val += w * v;
              blend_conf += w * cj_conf;
              blend_var += w * cj_var;
            }
            else {
              const int nearest2 = (int)internal_libxs_predict_position(model, cl2, norm_inputs);
              const double t = (0 != extrapolate)
                ? (double)cl2->nentries : (double)nearest2;
              const int d = cl2->order[j];
              const double* cj = cl2->coeffs + (size_t)j * (cl2->maxorder + 1);
              double val = 0;
              int k;
              for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
              blend_val += w * val;
              blend_err += w * cl2->errors[j];
              blend_conf += w;
              blend_rel = 1;
            }
            wsum += w;
          }
          if (wsum > 0) {
            vals[j] = blend_val / wsum;
            conf[j] = blend_conf / wsum;
            var[j] = blend_var / wsum;
            errs[j] = blend_err / wsum;
            rels[j] = blend_rel;
          }
        }
      }
      if (NULL != info) {
        const internal_libxs_predict_cluster_t* cl0 = &model->clusters[dists[0].idx];
        info->cluster = -1;
        info->distance = (cl0->dmax > 0.0)
          ? dists[0].dist / cl0->dmax : 0.0;
      }
      LIBXS_PREDICT_FREE(dists, dc_pool);
    }
    if (0 != extrapolate && n > 2) {
      int k;
      for (k = 1; k < n - 1; ++k) {
        const double avg = 0.5 * (vals[k - 1] + vals[k + 1]);
        vals[k] = 0.75 * vals[k] + 0.25 * avg;
      }
    }
    if (0 == extrapolate) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[best_c];
      for (j = 0; j < n; ++j) {
        if (var[j] > 0 && 0 != cl->mode[j]) {
          double mean = 0, global_var = 0;
          int k;
          for (k = 0; k < cl->nentries; ++k) {
            mean += cl->raw_outputs[(size_t)k * n + j];
          }
          mean /= cl->nentries;
          for (k = 0; k < cl->nentries; ++k) {
            const double d = cl->raw_outputs[(size_t)k * n + j] - mean;
            global_var += d * d;
          }
          global_var /= cl->nentries;
          if (global_var > 0) {
            const double ratio = var[j] / global_var;
            if (ratio > 1.5) {
              const double alpha = LIBXS_MIN((ratio - 1.5) * 0.1, 0.3);
              vals[j] = (1.0 - alpha) * vals[j] + alpha * mean;
            }
          }
        }
      }
    }
    { double min_conf = 1.0;
      int iter_count = 0, max_iter = (model->refine > 0) ? model->refine : 1;
      for (j = 0; j < n; ++j) {
        if (conf[j] < min_conf) min_conf = conf[j];
      }
      if (0 >= model->refine && min_conf >= 0.9) max_iter = 0;
      for (; iter_count < max_iter && NULL != model->entries; ++iter_count) {
        double target[128];
        int canon_pool = 0;
        double* canon = (double*)LIBXS_PREDICT_MALLOC(
          (size_t)m * sizeof(double), canon_pool);
        if (NULL != canon) {
          for (j = 0; j < n; ++j) {
            target[j] = (NULL != model->transforms)
              ? internal_libxs_predict_inv(model->transforms[j], vals[j])
              : vals[j];
          }
          libxs_predict_inverse(NULL, model, target, canon, NULL);
          { double refined[128], rconf[128];
            int rpool = 0;
            double* rnorm = (double*)LIBXS_PREDICT_MALLOC(
              (size_t)m * sizeof(double), rpool);
            if (NULL != rnorm) {
              int decomp2_pool = 0;
              double* dcinp = NULL;
              const double* eval_inp = canon;
              if (LIBXS_PREDICT_RAW != model->decompose
                && (model->nseries >= 2 || NULL != model->decompose_mat))
              {
                dcinp = (double*)LIBXS_PREDICT_MALLOC(
                  (size_t)m * sizeof(double), decomp2_pool);
                if (NULL != dcinp) {
                  internal_libxs_predict_decompose_apply(model, canon, dcinp);
                  eval_inp = dcinp;
                }
              }
              internal_libxs_predict_normalize(model, eval_inp, rnorm);
              { const int rc = best_c;
                const internal_libxs_predict_cluster_t* rcl = &model->clusters[rc];
                const double rt_dist = sqrt(libxs_dist2(norm_inputs, rnorm, m));
                if (rt_dist <= rcl->dmax) {
                  for (j = 0; j < n; ++j) {
                    if (conf[j] >= 0.9) continue;
                    { const int use_classify = (0 != force_classify)
                        ? 1 : ((0 != force_interp) ? 0 : rcl->mode[j]);
                      if (0 != use_classify) {
                        double rc_conf = 0;
                        refined[j] = internal_libxs_predict_classify(
                          rcl, rcl->kd_pts, rcl->nentries, m, rnorm, j, n,
                          rcl->ndistinct[j], extrapolate, -1, &rc_conf, NULL);
                        rconf[j] = rc_conf;
                      }
                      else {
                        refined[j] = vals[j];
                        rconf[j] = conf[j];
                      }
                    }
                  }
                  for (j = 0; j < n; ++j) {
                    if (conf[j] >= 0.9) continue;
                    if (rconf[j] > conf[j]) {
                      vals[j] = refined[j];
                      conf[j] = rconf[j];
                    }
                  }
                }
                else if (model->consistency > 0) {
                  const double q = model->quality;
                  const double s = 1.0
                    / (1.0 + model->consistency * rt_dist / rcl->dmax);
                  for (j = 0; j < n; ++j) {
                    conf[j] = q + s * (conf[j] - q);
                  }
                }
              }
              if (NULL != dcinp) LIBXS_PREDICT_FREE(dcinp, decomp2_pool);
              LIBXS_PREDICT_FREE(rnorm, rpool);
            }
          }
          LIBXS_PREDICT_FREE(canon, canon_pool);
        }
      }
    }
    if (NULL != model->transforms) {
      for (j = 0; j < n; ++j) {
        vals[j] = internal_libxs_predict_inv(model->transforms[j], vals[j]);
      }
    }
    if (diff_d > 0 && model->nseries > 0) {
      const int tgt = model->target;
      const int raw_w = model->window;
      int dd;
      for (dd = diff_d - 1; dd >= 0; --dd) {
        double base = raw_inputs[tgt * raw_w + raw_w - 1];
        int k;
        for (k = 0; k < dd; ++k) {
          base = base - raw_inputs[tgt * raw_w + raw_w - 2 - k];
        }
        for (j = 0; j < n; ++j) {
          base += vals[j];
          vals[j] = base;
        }
      }
    }
    if (NULL != outputs) {
      memcpy(outputs, vals, (size_t)n * sizeof(double));
    }
    if (NULL != info) {
      info->values = vals;
      info->error = errs;
      info->confidence = conf;
      info->variance = var;
      info->interpolated = rels;
      info->noutputs = n;
    }
    LIBXS_PREDICT_FREE(norm_inputs, norm_pool);
    if (NULL != decomp_inputs) LIBXS_PREDICT_FREE(decomp_inputs, decomp_pool);
    if (NULL != diff_inputs) LIBXS_PREDICT_FREE(diff_inputs, diff_pool);
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_predict_eval_batch_task(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend, int tid, int ntasks)
{
  const int m = model->ninputs, n = model->noutputs;
  const int chunk = (int)LIBXS_UPDIV(count, ntasks);
  const int begin = tid * chunk;
  const int end = LIBXS_MIN(begin + chunk, count);
  int i;
  LIBXS_ASSERT(NULL != model && 0 != model->built);
  LIBXS_ASSERT(NULL != inputs_batch && NULL != outputs_batch);
  for (i = begin; i < end; ++i) {
    libxs_predict_eval(NULL, model,
      inputs_batch + (size_t)i * m,
      outputs_batch + (size_t)i * n,
      NULL, nblend);
  }
}


LIBXS_API void libxs_predict_eval_batch(
  const libxs_predict_t* model,
  const double inputs_batch[], double outputs_batch[],
  int count, int nblend)
{
  libxs_predict_eval_batch_task(model, inputs_batch, outputs_batch,
    count, nblend, 0, 1);
}


LIBXS_API void libxs_predict_inverse(libxs_lock_t* lock,
  const libxs_predict_t* model,
  const double target_outputs[], double inputs[],
  libxs_predict_info_t* info)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != target_outputs && NULL != inputs);
  if (NULL == model->entries) {
    memset(inputs, 0, (size_t)model->ninputs * sizeof(double));
    if (NULL != info) {
      info->noutputs = model->noutputs;
      info->cluster = -1;
      info->distance = DBL_MAX;
      info->values = NULL;
      info->error = NULL;
      info->confidence = NULL;
      info->interpolated = NULL;
    }
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    double best_score = DBL_MAX;
    int best_i = 0, i, j;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    for (i = 0; i < p; ++i) {
      const double* eout = model->entries[i].outputs;
      double score = 0;
      int disqualified = 0;
      for (j = 0; j < n && 0 == disqualified; ++j) {
        double target = target_outputs[j];
        double actual = eout[j];
        if (NULL != model->transforms) {
          target = internal_libxs_predict_fwd(model->transforms[j], target);
        }
        { const int c = model->assignments[i];
          const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
          if (0 != cl->mode[j]) {
            if (target != actual) disqualified = 1;
          }
          else {
            const double d = target - actual;
            score += d * d;
          }
        }
      }
      if (0 == disqualified && score < best_score) {
        best_score = score;
        best_i = i;
      }
    }
    if (best_score >= DBL_MAX) {
      for (i = 0; i < p; ++i) {
        const double* eout = model->entries[i].outputs;
        double score = 0;
        for (j = 0; j < n; ++j) {
          double target = target_outputs[j];
          double actual = eout[j];
          double d;
          if (NULL != model->transforms) {
            target = internal_libxs_predict_fwd(model->transforms[j], target);
          }
          d = target - actual;
          score += d * d;
        }
        if (score < best_score) { best_score = score; best_i = i; }
      }
    }
    if (LIBXS_PREDICT_RAW != model->decompose
      && (model->nseries >= 2 || NULL != model->decompose_mat)) {
      int inv_pool = 0;
      double* raw = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), inv_pool);
      internal_libxs_predict_decompose_inverse(model, model->entries[best_i].inputs, raw);
      memcpy(inputs, raw, (size_t)m * sizeof(double));
      LIBXS_PREDICT_FREE(raw, inv_pool);
    }
    else {
      memcpy(inputs, model->entries[best_i].inputs, (size_t)m * sizeof(double));
    }
    if (NULL != info) {
      info->noutputs = n;
      info->cluster = (NULL != model->assignments) ? model->assignments[best_i] : -1;
      info->distance = sqrt(best_score);
      info->values = NULL;
      info->error = NULL;
      info->confidence = NULL;
      info->interpolated = NULL;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_predict_query(
  const libxs_predict_t* model, libxs_predict_query_t* info)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != info);
  { const double raw = (double)model->nentries * (model->ninputs + model->noutputs);
    double compressed = 0;
    int c;
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      int j;
      compressed += model->ninputs;
      for (j = 0; j < model->noutputs; ++j) {
        compressed += cl->order[j] + 1;
      }
    }
    info->compression = (compressed > 0) ? (raw / compressed) : 0;
  }
  info->order = model->order;
  info->nclusters = model->nclusters;
  info->nentries = model->nentries;
  info->iterations = model->iterations;
  info->diff_order = model->diff_order;
}


LIBXS_API void libxs_predict_get(const libxs_predict_t* model, int index,
  double inputs[], double outputs[])
{
  LIBXS_ASSERT(NULL != model && 0 <= index && index < model->nentries);
  if (NULL != model->entries) {
    if (NULL != inputs) {
      memcpy(inputs, model->entries[index].inputs, (size_t)model->ninputs * sizeof(double));
    }
    if (NULL != outputs) {
      memcpy(outputs, model->entries[index].outputs, (size_t)model->noutputs * sizeof(double));
    }
  }
  else {
    int c, offset = 0;
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      if (index < offset + cl->nentries) {
        const int local = index - offset;
        if (NULL != inputs) {
          const double* pt = cl->kd_pts + (size_t)local * model->ninputs;
          int i;
          for (i = 0; i < model->ninputs; ++i) {
            double v = pt[i];
            if (NULL != model->weights && model->weights[i] != 0) {
              v /= model->weights[i];
            }
            if (NULL != model->input_rng && model->input_rng[i] > 0) {
              v = v * model->input_rng[i] + model->input_min[i];
            }
            inputs[i] = v;
          }
          if (NULL != model->decompose_mat) {
            double* tmp = (double*)malloc((size_t)model->ninputs * sizeof(double));
            if (NULL != tmp) {
              memcpy(tmp, inputs, (size_t)model->ninputs * sizeof(double));
              internal_libxs_predict_decompose_inverse(model, tmp, inputs);
              free(tmp);
            }
          }
        }
        if (NULL != outputs) {
          memcpy(outputs, cl->raw_outputs + (size_t)local * model->noutputs,
            (size_t)model->noutputs * sizeof(double));
        }
        break;
      }
      offset += cl->nentries;
    }
  }
}


#include "libxs_predict_serial.h"


#include "libxs_predict_csv.h"
