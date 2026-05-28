/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_predict.h>
#include <libxs_perm.h>
#include <libxs_str.h>
#include <libxs_malloc.h>
#include <libxs_gemm.h>
#include "libxs_main.h"

#if !defined(LIBXS_PREDICT_MAXITER)
#  define LIBXS_PREDICT_MAXITER 100
#endif
#if !defined(LIBXS_PREDICT_MAGIC)
#  define LIBXS_PREDICT_MAGIC 0x58535052U /* "XSPR" */
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
  int built;
  int eval_mode;
  int iterations;
  int nseries, window, target, decompose;
  int nts, ts_capacity;
  int refine;
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
    /* farthest-first initialization */
    memcpy(centroids, pts, (size_t)m * sizeof(double));
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
        1, &shape, NULL, LIBXS_MIN(2, len - 1), -1))
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


LIBXS_API_INLINE double internal_libxs_predict_classify(
  const internal_libxs_predict_cluster_t* cl, const double* kd_pts,
  int nc, int m, const double* inputs, int output_j, int nouts,
  int ndistinct, int extrapolate, double* confidence, double* out_variance)
{
  const int k = cl->k_eff;
  const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
  double candidates[LIBXS_PREDICT_KNN];
  double dists[LIBXS_PREDICT_KNN];
  double best_val = cl->raw_outputs[output_j];
  int nfound = 0, exact = 0, i, max_idx = 0;
  if (0 != extrapolate) {
    for (i = 0; i < nc; ++i) {
      if (cl->sorted_idx[i] > max_idx) max_idx = cl->sorted_idx[i];
    }
  }
  /* linear scan: find k nearest by distance within cluster */
  for (i = 0; i < nc; ++i) {
    double d2 = libxs_dist2(inputs, kd_pts + (size_t)i * m, m);
    if (0 != extrapolate && max_idx > 0) {
      const double age = 1.0 - (double)cl->sorted_idx[i] / (double)max_idx;
      d2 *= 1.0 + 0.5 * age;
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
  return best_val;
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
    }
  }
  return model;
}


LIBXS_API void libxs_predict_destroy(libxs_predict_t* model)
{
  if (NULL != model) {
    int i;
    internal_libxs_predict_free_clusters(model);
    for (i = 0; i < model->nentries; ++i) {
      free(model->entries[i].inputs);
      free(model->entries[i].outputs);
    }
    free(model->entries);
    free(model->input_min);
    free(model->input_rng);
    free(model->weights);
    free(model->transforms);
    free(model->ts_buf);
    free(model->decompose_mat);
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


LIBXS_API_INLINE void internal_libxs_predict_decompose_apply(
  const libxs_predict_t* model, const double* raw, double* out)
{
  const int m = model->ninputs;
  if (LIBXS_PREDICT_PCA == model->decompose && NULL != model->decompose_mat) {
    const double alpha = 1.0, beta = 0.0;
    const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
      LIBXS_DATATYPE_F64, 'N', 'N', m, 1, m, m, m, m,
      &alpha, &beta, NULL);
    libxs_gemm_call(gemm, model->decompose_mat, raw, out);
    libxs_gemm_release(gemm);
  }
  else if (LIBXS_PREDICT_SPREAD == model->decompose
    && model->nseries >= 2 && model->window > 0)
  {
    const int w = model->window;
    int i;
    for (i = 0; i < w; ++i) {
      out[i] = 0.5 * (raw[i] + raw[w + i]);
      out[w + i] = 0.5 * (raw[i] - raw[w + i]);
    }
  }
  else {
    memcpy(out, raw, (size_t)m * sizeof(double));
  }
}


LIBXS_API_INLINE void internal_libxs_predict_decompose_inverse(
  const libxs_predict_t* model, const double* modes, double* raw)
{
  const int m = model->ninputs;
  if (LIBXS_PREDICT_PCA == model->decompose && NULL != model->decompose_mat) {
    const double alpha = 1.0, beta = 0.0;
    const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
      LIBXS_DATATYPE_F64, 'T', 'N', m, 1, m, m, m, m,
      &alpha, &beta, NULL);
    libxs_gemm_call(gemm, model->decompose_mat, modes, raw);
    libxs_gemm_release(gemm);
  }
  else if (LIBXS_PREDICT_SPREAD == model->decompose
    && model->nseries >= 2 && model->window > 0)
  {
    const int w = model->window;
    int i;
    for (i = 0; i < w; ++i) {
      raw[i] = modes[i] + modes[w + i];
      raw[w + i] = modes[i] - modes[w + i];
    }
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
      if (npc < m) {
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
          const double alpha = 1.0, beta = 0.0;
          const libxs_gemm_config_t *const gemm = libxs_gemm_dispatch(
            LIBXS_DATATYPE_F64, 'N', 'N', m, p, m, m, m, m,
            &alpha, &beta, NULL);
          for (i = 0; i < p; ++i) {
            memcpy(xmat + (size_t)i * m, model->entries[i].inputs,
              (size_t)m * sizeof(double));
          }
          libxs_gemm_call(gemm, model->decompose_mat, xmat, ymat);
          libxs_gemm_release(gemm);
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


LIBXS_API_INLINE int internal_libxs_predict_rf_pair_cmp(
  const void* a, const void* b, void* ctx)
{
  const double va = ((const internal_libxs_predict_rf_pair_t*)a)->val;
  const double vb = ((const internal_libxs_predict_rf_pair_t*)b)->val;
  (void)ctx;
  return (va > vb) - (va < vb);
}


LIBXS_API_INLINE int internal_libxs_predict_rf_split(
  const internal_libxs_predict_entry_t* entries,
  const int* subset, int nsub, int nfeat, int nfeatsub,
  internal_libxs_predict_rf_node_t* node, size_t seed,
  int output_idx, int label_off)
{
  int result = 0;
  double best_gini = 2.0;
  int trial;
  int pairs_pool = 0;
  const size_t feat_coprime = libxs_coprime2((size_t)nfeat);
  internal_libxs_predict_rf_pair_t* pairs =
    (internal_libxs_predict_rf_pair_t*)LIBXS_PREDICT_MALLOC(
      (size_t)nsub * sizeof(internal_libxs_predict_rf_pair_t), pairs_pool);
  node->feature = -1;
  node->label = -1;
  if (NULL != pairs) {
  for (trial = 0; trial < nfeatsub; ++trial) {
    const int f = (int)(LIBXS_SHUFFLE_INDEX(
      (size_t)trial, (size_t)nfeat, feat_coprime, seed) % (size_t)nfeat);
    int left_counts[128], right_counts[128];
    int nleft, nright, i, k;
    for (i = 0; i < nsub; ++i) {
      pairs[i].val = entries[subset[i]].inputs[f];
      pairs[i].idx = subset[i];
    }
    libxs_sort(pairs, nsub, sizeof(pairs[0]),
      internal_libxs_predict_rf_pair_cmp, NULL);
    memset(right_counts, 0, sizeof(right_counts));
    nright = nsub; nleft = 0;
    for (i = 0; i < nsub; ++i) {
      ++right_counts[(LIBXS_ROUNDX(int, entries[pairs[i].idx].outputs[output_idx]) + label_off) & 127];
    }
    memset(left_counts, 0, sizeof(left_counts));
    for (i = 0; i < nsub - 1; ++i) {
      const int label = (LIBXS_ROUNDX(int, entries[pairs[i].idx].outputs[output_idx]) + label_off) & 127;
      ++left_counts[label]; ++nleft;
      --right_counts[label]; --nright;
      if (pairs[i].val == pairs[i + 1].val) continue;
      { double gini_l = 1.0, gini_r = 1.0, gini;
        for (k = 0; k < 128; ++k) {
          if (left_counts[k] > 0) {
            double p = (double)left_counts[k] / nleft;
            gini_l -= p * p;
          }
          if (right_counts[k] > 0) {
            double p = (double)right_counts[k] / nright;
            gini_r -= p * p;
          }
        }
        gini = ((double)nleft * gini_l + (double)nright * gini_r) / nsub;
        if (gini < best_gini) {
          best_gini = gini;
          node->feature = f;
          node->threshold = 0.5 * (pairs[i].val + pairs[i + 1].val);
        }
      }
    }
  }
  }
  LIBXS_PREDICT_FREE(pairs, pairs_pool);
  result = (node->feature >= 0) ? 1 : 0;
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_rf_build_tree(
  const internal_libxs_predict_entry_t* entries,
  int* subset, int nsub, int nfeat, int max_depth, int min_leaf,
  internal_libxs_predict_rf_node_t* nodes, int max_nodes,
  int output_idx, int label_off)
{
  int stack_subset[64], stack_count[64], stack_depth[64], stack_node[64];
  int sp = 0, nnodes = 0;
  int nfeatsub = (int)(sqrt((double)nfeat) + 0.5);
  if (nfeatsub < 1) nfeatsub = 1;
  stack_subset[0] = 0;
  stack_count[0] = nsub;
  stack_depth[0] = 0;
  stack_node[0] = nnodes++;
  nodes[0].feature = -1;
  nodes[0].left = -1;
  nodes[0].right = -1;
  sp = 1;
  while (sp > 0 && nnodes < max_nodes - 2) {
    const int si = stack_subset[--sp];
    const int nc = stack_count[sp];
    const int depth = stack_depth[sp];
    const int ni = stack_node[sp];
    int counts[128] = { 0 }, best_label = 0, best_count = 0, k;
    internal_libxs_predict_rf_node_t split;
    for (k = 0; k < nc; ++k) {
      ++counts[(LIBXS_ROUNDX(int, entries[subset[si + k]].outputs[output_idx]) + label_off) & 127];
    }
    for (k = 0; k < 128; ++k) {
      if (counts[k] > best_count) { best_count = counts[k]; best_label = k; }
    }
    nodes[ni].label = best_label;
    if (depth >= max_depth || nc <= min_leaf || best_count == nc
      || 0 == internal_libxs_predict_rf_split(entries, subset + si, nc,
        nfeat, nfeatsub, &split, (size_t)ni, output_idx, label_off))
    {
      nodes[ni].feature = -1;
      continue;
    }
    { int* sub = subset + si;
      int i, nleft = 0, nright = 0;
      nodes[ni].feature = split.feature;
      nodes[ni].threshold = split.threshold;
      for (i = 0; i < nc; ++i) {
        if (entries[sub[i]].inputs[split.feature] <= split.threshold) ++nleft;
      }
      nright = nc - nleft;
      if (0 == nleft || 0 == nright) { nodes[ni].feature = -1; continue; }
      { int part_pool = 0;
        int* part = (int*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(int), part_pool);
        if (NULL != part) {
          int li = 0, ri = 0;
          for (i = 0; i < nc; ++i) {
            if (entries[sub[i]].inputs[split.feature] <= split.threshold) {
              part[li++] = sub[i];
            }
            else {
              part[nleft + ri++] = sub[i];
            }
          }
          memcpy(sub, part, (size_t)nc * sizeof(int));
          LIBXS_PREDICT_FREE(part, part_pool);
        }
        else { nodes[ni].feature = -1; continue; }
      }
      nodes[ni].left = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      if (sp < 64) {
        stack_subset[sp] = si;
        stack_count[sp] = nleft;
        stack_depth[sp] = depth + 1;
        stack_node[sp] = nnodes;
        ++sp;
      }
      ++nnodes;
      nodes[ni].right = nnodes;
      nodes[nnodes].feature = -1;
      nodes[nnodes].left = -1;
      nodes[nnodes].right = -1;
      if (sp < 64) {
        stack_subset[sp] = si + nleft;
        stack_count[sp] = nright;
        stack_depth[sp] = depth + 1;
        stack_node[sp] = nnodes;
        ++sp;
      }
      ++nnodes;
    }
  }
  return nnodes;
}


#if !defined(LIBXS_PREDICT_RF_NTREES)
#  define LIBXS_PREDICT_RF_NTREES 100
#endif

LIBXS_API_INLINE void internal_libxs_predict_rf_build(libxs_predict_t* model)
{
  const int p = model->nentries;
  const int n = model->noutputs;
  const int ntrees = LIBXS_PREDICT_RF_NTREES;
  internal_libxs_predict_rf_t* rf =
    (internal_libxs_predict_rf_t*)calloc(1, sizeof(internal_libxs_predict_rf_t));
  if (NULL != rf) {
    rf->trees = (internal_libxs_predict_rf_tree_t*)calloc(
      (size_t)ntrees * (size_t)n, sizeof(internal_libxs_predict_rf_tree_t));
    rf->label_offset = (int*)malloc((size_t)n * sizeof(int));
    rf->ntrees = ntrees;
    rf->noutputs = n;
    if (NULL != rf->trees && NULL != rf->label_offset) {
      int oi, i;
      for (oi = 0; oi < n; ++oi) {
        int vmin = LIBXS_ROUNDX(int, model->entries[0].outputs[oi]);
        for (i = 1; i < p; ++i) {
          const int v = LIBXS_ROUNDX(int, model->entries[i].outputs[oi]);
          if (v < vmin) vmin = v;
        }
        rf->label_offset[oi] = -vmin;
      }
      model->rf = rf;
    }
    else {
      free(rf->trees);
      free(rf->label_offset);
      free(rf);
    }
  }
}


LIBXS_API_INLINE void internal_libxs_predict_rf_build_tasks(
  libxs_predict_t* model, int tid, int ntasks)
{
  const internal_libxs_predict_rf_t* rf = model->rf;
  if (NULL != rf) {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = rf->noutputs;
    const int ntrees = rf->ntrees;
    const int total_trees = ntrees * n;
    const int max_depth = (int)(2.0 * log((double)p) / log(2.0));
    const int min_leaf = 5;
    const int max_nodes = LIBXS_MIN(p / min_leaf * 2 + 1, 65536);
    const int chunk = (total_trees + ntasks - 1) / ntasks;
    const int begin = tid * chunk;
    const int end = LIBXS_MIN(begin + chunk, total_trees);
    int bootstrap_pool = 0;
    int* bootstrap = (int*)LIBXS_PREDICT_MALLOC(
      (size_t)p * sizeof(int), bootstrap_pool);
    if (NULL != bootstrap) {
      int ti;
      for (ti = begin; ti < end; ++ti) {
        const int oi = ti / ntrees;
        const int t = ti % ntrees;
        const size_t boot_n = (size_t)p * 2 + 1;
        const size_t boot_coprime = libxs_coprime2(boot_n);
        int nodes_pool = 0;
        internal_libxs_predict_rf_node_t* nodes;
        int i, nn;
        if (NULL != rf->trees[ti].nodes) continue;
        nodes = (internal_libxs_predict_rf_node_t*)LIBXS_PREDICT_MALLOC(
            (size_t)max_nodes * sizeof(internal_libxs_predict_rf_node_t),
            nodes_pool);
        for (i = 0; i < p; ++i) {
          bootstrap[i] = (int)(LIBXS_SHUFFLE_INDEX(
            i, boot_n, boot_coprime,
            (size_t)(oi * ntrees + t) * 7 + 13) % (size_t)p);
        }
        if (NULL != nodes) {
          nn = internal_libxs_predict_rf_build_tree(
            model->entries, bootstrap, p, m, max_depth, min_leaf,
            nodes, max_nodes, oi, rf->label_offset[oi]);
          rf->trees[ti].nodes = (internal_libxs_predict_rf_node_t*)malloc(
            (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
          if (NULL != rf->trees[ti].nodes) {
            memcpy(rf->trees[ti].nodes, nodes,
              (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
            rf->trees[ti].nnodes = nn;
          }
          LIBXS_PREDICT_FREE(nodes, nodes_pool);
        }
      }
      LIBXS_PREDICT_FREE(bootstrap, bootstrap_pool);
    }
  }
}


LIBXS_API_INLINE int internal_libxs_predict_rf_eval_output(
  const internal_libxs_predict_rf_t* rf, int output_idx,
  const double* inputs, double* confidence)
{
  int votes[128] = { 0 };
  int best_label = 0, best_count = 0, t, k;
  const int base = output_idx * rf->ntrees;
  for (t = 0; t < rf->ntrees; ++t) {
    const internal_libxs_predict_rf_tree_t* tree = &rf->trees[base + t];
    int ni = 0;
    if (NULL == tree->nodes || 0 == tree->nnodes) continue;
    while (ni >= 0 && ni < tree->nnodes && tree->nodes[ni].feature >= 0) {
      const internal_libxs_predict_rf_node_t* nd = &tree->nodes[ni];
      ni = (inputs[nd->feature] <= nd->threshold) ? nd->left : nd->right;
    }
    if (ni >= 0 && ni < tree->nnodes) {
      ++votes[tree->nodes[ni].label & 127];
    }
  }
  for (k = 0; k < 128; ++k) {
    if (votes[k] > best_count) { best_count = votes[k]; best_label = k; }
  }
  if (NULL != confidence) {
    *confidence = (rf->ntrees > 0) ? (double)best_count / rf->ntrees : 0.0;
  }
  return best_label - rf->label_offset[output_idx];
}


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


LIBXS_API_INLINE void internal_libxs_predict_ts_expand(libxs_predict_t* model)
{
  const int s = model->nseries;
  const int w = model->window;
  const int h = model->noutputs;
  const int m = model->ninputs;
  const int nwindows = model->nts - w - h + 1;
  int raw_pool = 0;
  double* raw = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), raw_pool);
  int t;
  if (NULL != raw) {
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
  if (EXIT_SUCCESS == libxs_predict_build(ctx->model, ctx->nclusters, ord)) {
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


LIBXS_API int libxs_predict_build(libxs_predict_t* model, int nclusters, int order)
{
  int result = EXIT_SUCCESS;
  if (NULL != model && 0 < model->nts && 0 == model->nentries) {
    internal_libxs_predict_ts_expand(model);
  }
  if (NULL != model && 0 < model->nentries
    && LIBXS_PREDICT_PCA == model->decompose && NULL == model->decompose_mat)
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
    result = libxs_predict_build(model, nclusters, best_ord);
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    int c, i;
    if (order > LIBXS_FPRINT_MAXORDER) order = LIBXS_FPRINT_MAXORDER;
    model->order = order;
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
    model->clusters = (internal_libxs_predict_cluster_t*)calloc(
      (size_t)nclusters, sizeof(internal_libxs_predict_cluster_t));
    model->eval_buf = (double*)malloc((size_t)n * 4 * sizeof(double) + (size_t)n * sizeof(int));
    if (NULL == model->assignments || NULL == model->clusters || NULL == model->eval_buf) {
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
      internal_libxs_predict_kmeans(model, nclusters);
      for (i = 0; i < p; ++i) {
        ++model->clusters[model->assignments[i]].nentries;
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
        const size_t shape = (size_t)nc;
        const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
        for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
          int ndistinct = 0, d;
          double prev;
          double* seq = cl->raw_outputs + j;
          int buf_pool = 0;
          double* buf = (double*)LIBXS_PREDICT_MALLOC((size_t)nc * sizeof(double), buf_pool);
          libxs_fprint_t fp;
          if (NULL == buf) { result = EXIT_FAILURE; continue; }
          for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)k * n + j];
          libxs_sort(buf, nc, sizeof(double), libxs_cmp_f64, NULL);
          prev = buf[0]; ndistinct = 1;
          for (k = 1; k < nc; ++k) {
            if (buf[k] != prev) { ++ndistinct; prev = buf[k]; }
          }
          cl->ndistinct[j] = ndistinct;
          for (k = 0; k < nc; ++k) {
            buf[k] = cl->raw_outputs[(size_t)k * n + j];
          }
          { const size_t stride = (size_t)n;
            libxs_fprint(&fp, LIBXS_DATATYPE_F64, seq, 1, &shape, &stride,
              LIBXS_FPRINT_MAXORDER, -1);
          }
          { int trunc_order = maxorder;
            int decay_order = 0;
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
              trunc_order = LIBXS_MIN(decay_order, maxorder);
            }
            cl->order[j] = trunc_order;
            cl->coeffs[(size_t)j * (maxorder + 1)] = buf[0];
            for (d = 1; d <= trunc_order && d < nc; ++d) {
              for (k = 0; k < nc - d; ++k) buf[k] = buf[k + 1] - buf[k];
              cl->coeffs[(size_t)j * (maxorder + 1) + d] = buf[0];
            }
            if (trunc_order < nc - 1) {
              double emax = 0;
              for (k = 0; k < nc - d; ++k) buf[k] = buf[k + 1] - buf[k];
              for (k = 0; k < nc - trunc_order - 1; ++k) {
                const double a = buf[k] < 0 ? -buf[k] : buf[k];
                if (a > emax) emax = a;
              }
              cl->errors[j] = emax;
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
            : LIBXS_MIN(LIBXS_MAX(3, (int)(sqrt((double)nc) + 0.5)), LIBXS_PREDICT_KNN);
        }
      }
    }
    if (EXIT_SUCCESS == result) {
      model->built = 1;
    }
    else {
      internal_libxs_predict_free_clusters(model);
    }
  }
  return result;
}


LIBXS_API int libxs_predict_build_task(libxs_lock_t* lock,
  libxs_predict_t* model, int nclusters, int order,
  int tid, int ntasks)
{
  int result;
  LIBXS_ASSERT(NULL != model);
  if (0 == tid) {
    model->phase = 1;
    result = libxs_predict_build(model, nclusters, order);
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
    const int force_classify = (0 != (mode & LIBXS_PREDICT_CLASSIFY)) ? 1 : 0;
    const int force_interp = (0 != (mode & LIBXS_PREDICT_INTERPOLATE)) ? 1 : 0;
    const int extrapolate_mode = (0 != (mode & LIBXS_PREDICT_TEMPORAL)) ? 1 : 0;
    int extrapolate = 0;
    int norm_pool = 0, decomp_pool = 0;
    double* decomp_inputs = NULL;
    double* norm_inputs = (double*)LIBXS_PREDICT_MALLOC((size_t)m * sizeof(double), norm_pool);
    double local_buf[256];
    double *vals, *errs, *conf, *var, best_dist;
    int *rels, c, j, best_c = 0;
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
    if (NULL == lock && NULL != outputs && n * 4 + n <= 256) {
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
      if (d < best_dist) { best_dist = d; best_c = c; }
    }
    if (NULL != model->rf) {
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
          if (0 != use_classify) {
            vals[j] = internal_libxs_predict_classify(
              cl, cl->kd_pts, cl->nentries, m, norm_inputs, j, n,
              cl->ndistinct[j], extrapolate, &conf[j], &var[j]);
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
      if (model->nclusters > 1) {
        double avg_conf = 0;
        for (j = 0; j < n; ++j) avg_conf += conf[j];
        avg_conf /= n;
        if (avg_conf < 0.7) nblend = LIBXS_MIN(3, model->nclusters);
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
        if (conf[j] >= conf_thr) continue;
        { const internal_libxs_predict_cluster_t* cl_primary = &model->clusters[dists[0].idx];
          const int use_classify = (0 != force_classify)
            ? 1 : ((0 != force_interp) ? 0 : cl_primary->mode[j]);
          double blend_val = 0, blend_conf = 0, blend_var = 0, blend_err = 0;
          double wsum = 0;
          int blend_rel = 0;
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
                cl2->ndistinct[j], extrapolate, &cj_conf, &cj_var);
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
      for (; iter_count < max_iter && model->nentries > 0; ++iter_count) {
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
                for (j = 0; j < n; ++j) {
                  if (conf[j] >= 0.9) continue;
                  { const int use_classify = (0 != force_classify)
                      ? 1 : ((0 != force_interp) ? 0 : rcl->mode[j]);
                    if (0 != use_classify) {
                      double rc_conf = 0;
                      refined[j] = internal_libxs_predict_classify(
                        rcl, rcl->kd_pts, rcl->nentries, m, rnorm, j, n,
                        rcl->ndistinct[j], extrapolate, &rc_conf, NULL);
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
  {
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
}


LIBXS_API void libxs_predict_get(const libxs_predict_t* model, int index,
  double inputs[], double outputs[])
{
  LIBXS_ASSERT(NULL != model && 0 <= index && index < model->nentries);
  if (NULL != inputs) {
    memcpy(inputs, model->entries[index].inputs, (size_t)model->ninputs * sizeof(double));
  }
  if (NULL != outputs) {
    memcpy(outputs, model->entries[index].outputs, (size_t)model->noutputs * sizeof(double));
  }
}


LIBXS_API int libxs_predict_save(const libxs_predict_t* model, void* buffer, size_t* size)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || 0 == model->built || NULL == size) {
    result = EXIT_FAILURE;
  }
  else {
    size_t required = 0;
    int c, j;
    required += sizeof(uint32_t) + 4 * sizeof(uint16_t) + 2 * sizeof(uint8_t);
    required += 4 * sizeof(uint16_t);
    required += (size_t)model->ninputs * 2 * sizeof(double);
    if (NULL != model->weights) required += (size_t)model->ninputs * sizeof(double);
    if (NULL != model->transforms) required += (size_t)model->noutputs * sizeof(uint8_t);
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      required += (size_t)model->ninputs * sizeof(double);
      required += sizeof(double);
      required += sizeof(uint16_t) + 2 * sizeof(uint8_t);
      required += (size_t)model->noutputs * 3;
      required += (size_t)model->noutputs * sizeof(uint16_t);
      required += (size_t)model->noutputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double);
      for (j = 0; j < model->noutputs; ++j) {
        required += (size_t)(cl->order[j] + 1) * sizeof(double);
      }
    }
    if (NULL != model->rf) {
      const int n = model->rf->noutputs;
      const int total_trees = model->rf->ntrees * n;
      required += sizeof(uint16_t) + sizeof(uint16_t);
      required += (size_t)n * sizeof(int16_t);
      for (c = 0; c < total_trees; ++c) {
        required += sizeof(uint16_t);
        required += (size_t)model->rf->trees[c].nnodes * (2 + 8 + 2 + 2 + 1);
      }
    }
    if (NULL == buffer) {
      *size = required;
    }
    else if (*size < required) {
      *size = required;
      result = EXIT_FAILURE;
    }
    else {
      unsigned char* dst = (unsigned char*)buffer;
#define WRITE_U32(V) do { const uint32_t v_=(uint32_t)(V); memcpy(dst,&v_,4); dst+=4; } while(0)
#define WRITE_U16(V) do { const uint16_t v_=(uint16_t)(V); memcpy(dst,&v_,2); dst+=2; } while(0)
#define WRITE_U8(V)  do { *dst++ = (unsigned char)(V); } while(0)
#define WRITE_F64(V) do { const double v_=(V); memcpy(dst,&v_,8); dst+=8; } while(0)
#define WRITE_BLK(PTR,SZ) do { memcpy(dst,(PTR),(SZ)); dst+=(SZ); } while(0)
      WRITE_U32(LIBXS_PREDICT_MAGIC);
      WRITE_U16(LIBXS_PREDICT_VERSION);
      WRITE_U16(model->ninputs);
      WRITE_U16(model->noutputs);
      WRITE_U16(model->nclusters);
      WRITE_U8(NULL != model->weights ? 1 : 0);
      WRITE_U8(NULL != model->transforms ? 1 : 0);
      WRITE_U16(model->nseries);
      WRITE_U16(model->window);
      WRITE_U16(model->target);
      WRITE_U16(model->decompose);
      WRITE_BLK(model->input_min, (size_t)model->ninputs * sizeof(double));
      WRITE_BLK(model->input_rng, (size_t)model->ninputs * sizeof(double));
      if (NULL != model->weights) {
        WRITE_BLK(model->weights, (size_t)model->ninputs * sizeof(double));
      }
      if (NULL != model->transforms) {
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(model->transforms[j]);
      }
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        WRITE_BLK(cl->centroid, (size_t)model->ninputs * sizeof(double));
        WRITE_F64(cl->dmax);
        WRITE_U16(cl->nentries);
        WRITE_U8(cl->maxorder);
        WRITE_U8(cl->k_eff);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->order[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->interpolated[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->mode[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U16(cl->ndistinct[j]);
        WRITE_BLK(cl->errors, (size_t)model->noutputs * sizeof(double));
        WRITE_BLK(cl->kd_pts, (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double));
        WRITE_BLK(cl->raw_outputs, (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double));
        for (j = 0; j < model->noutputs; ++j) {
          WRITE_BLK(cl->coeffs + (size_t)j * (cl->maxorder + 1),
            (size_t)(cl->order[j] + 1) * sizeof(double));
        }
      }
      if (NULL != model->rf) {
        const int total_trees = model->rf->ntrees * model->rf->noutputs;
        WRITE_U16(model->rf->ntrees);
        WRITE_U16(model->rf->noutputs);
        for (j = 0; j < model->rf->noutputs; ++j) {
          const int16_t off = (int16_t)model->rf->label_offset[j];
          memcpy(dst, &off, 2); dst += 2;
        }
        for (c = 0; c < total_trees; ++c) {
          const internal_libxs_predict_rf_tree_t* tree = &model->rf->trees[c];
          int k;
          WRITE_U16(tree->nnodes);
          for (k = 0; k < tree->nnodes; ++k) {
            const internal_libxs_predict_rf_node_t* nd = &tree->nodes[k];
            { const int16_t f = (int16_t)nd->feature;
              memcpy(dst, &f, 2); dst += 2;
            }
            WRITE_F64(nd->threshold);
            { const int16_t l = (int16_t)nd->left;
              const int16_t r = (int16_t)nd->right;
              memcpy(dst, &l, 2); dst += 2;
              memcpy(dst, &r, 2); dst += 2;
            }
            WRITE_U8(nd->label);
          }
        }
      }
#undef WRITE_U32
#undef WRITE_U16
#undef WRITE_U8
#undef WRITE_F64
#undef WRITE_BLK
      *size = (size_t)(dst - (unsigned char*)buffer);
    }
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_read(
  const unsigned char** src, const unsigned char* end, void* dst, size_t sz)
{
  int result = EXIT_SUCCESS;
  if (*src + sz > end) {
    result = EXIT_FAILURE;
  }
  else {
    memcpy(dst, *src, sz);
    *src += sz;
  }
  return result;
}


LIBXS_API libxs_predict_t* libxs_predict_load(const void* buffer, size_t size)
{
  libxs_predict_t* model = NULL;
  if (NULL != buffer && size >= sizeof(uint32_t) + 4 * sizeof(uint16_t)) {
    const unsigned char* src = (const unsigned char*)buffer;
    const unsigned char* end = src + size;
    uint32_t magic = 0;
    uint16_t version = 0, ninp = 0, nout = 0, nclust = 0;
    int ok = EXIT_SUCCESS;
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &magic, 4);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &version, 2);
    if (EXIT_SUCCESS == ok && (magic != LIBXS_PREDICT_MAGIC
      || version != LIBXS_PREDICT_VERSION)) ok = EXIT_FAILURE;
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ninp, 2);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nout, 2);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &nclust, 2);
    if (EXIT_SUCCESS == ok) {
      model = libxs_predict_create((int)ninp, (int)nout);
      if (NULL == model) ok = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == ok) {
      uint8_t has_weights = 0, has_transforms = 0;
      uint16_t ts_nseries = 0, ts_window = 0, ts_target = 0, ts_decompose = 0;
      ok = internal_libxs_predict_read(&src, end, &has_weights, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &has_transforms, 1);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_nseries, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_window, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_target, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &ts_decompose, 2);
      if (EXIT_SUCCESS == ok) {
        model->nseries = (int)ts_nseries;
        model->window = (int)ts_window;
        model->target = (int)ts_target;
        model->decompose = (int)ts_decompose;
      }
      model->input_min = (double*)malloc((size_t)ninp * sizeof(double));
      model->input_rng = (double*)malloc((size_t)ninp * sizeof(double));
      if (NULL == model->input_min || NULL == model->input_rng) ok = EXIT_FAILURE;
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end,
          model->input_min, (size_t)ninp * sizeof(double));
      }
      if (EXIT_SUCCESS == ok) {
        ok = internal_libxs_predict_read(&src, end,
          model->input_rng, (size_t)ninp * sizeof(double));
      }
      if (EXIT_SUCCESS == ok && 0 != has_weights) {
        model->weights = (double*)malloc((size_t)ninp * sizeof(double));
        if (NULL != model->weights) {
          ok = internal_libxs_predict_read(&src, end,
            model->weights, (size_t)ninp * sizeof(double));
        }
        else ok = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == ok && 0 != has_transforms) {
        int j;
        model->transforms = (int*)calloc((size_t)nout, sizeof(int));
        if (NULL != model->transforms) {
          for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
            uint8_t v = 0;
            ok = internal_libxs_predict_read(&src, end, &v, 1);
            model->transforms[j] = (int)v;
          }
        }
        else ok = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == ok) {
      model->nclusters = (int)nclust;
      model->clusters = (internal_libxs_predict_cluster_t*)calloc(
        (size_t)nclust, sizeof(internal_libxs_predict_cluster_t));
      model->eval_buf = (double*)malloc(
        (size_t)nout * 3 * sizeof(double) + (size_t)nout * sizeof(int));
      if (NULL == model->clusters || NULL == model->eval_buf) ok = EXIT_FAILURE;
    }
    { int c;
      for (c = 0; c < (int)nclust && EXIT_SUCCESS == ok; ++c) {
        internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        uint16_t ne = 0;
        uint8_t mo = 0;
        int j;
        cl->centroid = (double*)malloc((size_t)ninp * sizeof(double));
        cl->order = (int*)malloc((size_t)nout * sizeof(int));
        cl->interpolated = (int*)malloc((size_t)nout * sizeof(int));
        cl->mode = (int*)malloc((size_t)nout * sizeof(int));
        cl->ndistinct = (int*)malloc((size_t)nout * sizeof(int));
        cl->errors = (double*)malloc((size_t)nout * sizeof(double));
        if (NULL == cl->centroid || NULL == cl->order || NULL == cl->interpolated
          || NULL == cl->mode || NULL == cl->ndistinct || NULL == cl->errors) ok = EXIT_FAILURE;
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end,
            cl->centroid, (size_t)ninp * sizeof(double));
        }
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &cl->dmax, 8);
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end, &ne, 2);
          if (EXIT_SUCCESS == ok) cl->nentries = (int)ne;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end, &mo, 1);
          if (EXIT_SUCCESS == ok) cl->maxorder = (int)mo;
        }
        if (EXIT_SUCCESS == ok) {
          uint8_t ke = 0;
          ok = internal_libxs_predict_read(&src, end, &ke, 1);
          if (EXIT_SUCCESS == ok) cl->k_eff = (int)ke;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->order[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->interpolated[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->mode[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint16_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 2);
          cl->ndistinct[j] = (int)v;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end,
            cl->errors, (size_t)nout * sizeof(double));
        }
        if (EXIT_SUCCESS == ok) {
          cl->kd_pts = (double*)malloc(
            (size_t)cl->nentries * (size_t)ninp * sizeof(double));
          if (NULL == cl->kd_pts) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read(&src, end,
              cl->kd_pts, (size_t)cl->nentries * (size_t)ninp * sizeof(double));
          }
        }
        if (EXIT_SUCCESS == ok) {
          cl->raw_outputs = (double*)malloc(
            (size_t)cl->nentries * (size_t)nout * sizeof(double));
          if (NULL == cl->raw_outputs) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read(&src, end,
              cl->raw_outputs, (size_t)cl->nentries * (size_t)nout * sizeof(double));
          }
        }
        if (EXIT_SUCCESS == ok) {
          cl->coeffs = (double*)calloc(
            (size_t)nout * (size_t)(cl->maxorder + 1), sizeof(double));
          if (NULL == cl->coeffs) ok = EXIT_FAILURE;
          for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
            ok = internal_libxs_predict_read(&src, end,
              cl->coeffs + (size_t)j * (cl->maxorder + 1),
              (size_t)(cl->order[j] + 1) * sizeof(double));
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok && src < end && model->decompose == LIBXS_PREDICT_RF) {
      uint16_t rf_ntrees = 0, rf_nouts = 0;
      int j;
      ok = internal_libxs_predict_read(&src, end, &rf_ntrees, 2);
      if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &rf_nouts, 2);
      if (EXIT_SUCCESS == ok && rf_ntrees > 0 && rf_nouts > 0) {
        internal_libxs_predict_rf_t* rf = (internal_libxs_predict_rf_t*)calloc(
          1, sizeof(internal_libxs_predict_rf_t));
        if (NULL != rf) {
          const int total_trees = (int)rf_ntrees * (int)rf_nouts;
          rf->ntrees = (int)rf_ntrees;
          rf->noutputs = (int)rf_nouts;
          rf->label_offset = (int*)malloc((size_t)rf_nouts * sizeof(int));
          rf->trees = (internal_libxs_predict_rf_tree_t*)calloc(
            (size_t)total_trees, sizeof(internal_libxs_predict_rf_tree_t));
          if (NULL != rf->label_offset && NULL != rf->trees) {
            int ti;
            for (j = 0; j < (int)rf_nouts && EXIT_SUCCESS == ok; ++j) {
              int16_t off = 0;
              ok = internal_libxs_predict_read(&src, end, &off, 2);
              rf->label_offset[j] = (int)off;
            }
            for (ti = 0; ti < total_trees && EXIT_SUCCESS == ok; ++ti) {
              uint16_t nn = 0;
              int k;
              ok = internal_libxs_predict_read(&src, end, &nn, 2);
              if (EXIT_SUCCESS == ok && nn > 0) {
                rf->trees[ti].nodes = (internal_libxs_predict_rf_node_t*)malloc(
                  (size_t)nn * sizeof(internal_libxs_predict_rf_node_t));
                rf->trees[ti].nnodes = (int)nn;
                if (NULL == rf->trees[ti].nodes) ok = EXIT_FAILURE;
                for (k = 0; k < (int)nn && EXIT_SUCCESS == ok; ++k) {
                  int16_t f = 0, l = 0, r = 0;
                  uint8_t lab = 0;
                  ok = internal_libxs_predict_read(&src, end, &f, 2);
                  if (EXIT_SUCCESS == ok) {
                    ok = internal_libxs_predict_read(&src, end,
                      &rf->trees[ti].nodes[k].threshold, 8);
                  }
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &l, 2);
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &r, 2);
                  if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read(&src, end, &lab, 1);
                  rf->trees[ti].nodes[k].feature = (int)f;
                  rf->trees[ti].nodes[k].left = (int)l;
                  rf->trees[ti].nodes[k].right = (int)r;
                  rf->trees[ti].nodes[k].label = (int)lab;
                }
              }
            }
          }
          else ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) model->rf = rf;
          else {
            if (NULL != rf->trees) {
              int ti;
              for (ti = 0; ti < total_trees; ++ti) free(rf->trees[ti].nodes);
              free(rf->trees);
            }
            free(rf->label_offset);
            free(rf);
          }
        }
      }
    }
    if (EXIT_SUCCESS == ok) {
      model->built = 1;
    }
    else if (NULL != model) {
      libxs_predict_destroy(model);
      model = NULL;
    }
  }
  return model;
}


LIBXS_API_INLINE const char* internal_libxs_predict_detect_delims(const char* line)
{
  const char* result = " ";
  if (NULL != strchr(line, ';')) result = ";";
  else if (NULL != strchr(line, ',')) result = ",";
  else if (NULL != strchr(line, '\t')) result = "\t";
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_resolve_col(
  const char* name, const char* header, const char* delims)
{
  int col = 0;
  const char* p = header;
  char* endptr = NULL;
  const long idx = strtol(name, &endptr, 10);
  if (endptr != name && '\0' == *endptr) return (int)idx;
  if (NULL == header) return -1;
  while ('\0' != *p) {
    const char* field = p;
    size_t flen;
    while ('\0' != *p && NULL == strchr(delims, *p)) ++p;
    flen = (size_t)(p - field);
    if (flen == strlen(name)) {
      const char* hit = libxs_stristrn(field, name, flen);
      if (hit == field) return col;
    }
    if ('\0' != *p) ++p;
    ++col;
  }
  return -1;
}


LIBXS_API_INLINE int internal_libxs_predict_parse_row(
  const char* line, const char* delims, const int idx[], int nidx, double vals[])
{
  int filled = 0, col = 0, ok = 1, i;
  const char* p = line;
  while ('\0' != *p && filled < nidx && 0 != ok) {
    const char* field = p;
    while ('\0' != *p && NULL == strchr(delims, *p)) ++p;
    for (i = 0; i < nidx && 0 != ok; ++i) {
      if (col == idx[i]) {
        char* endptr = NULL;
        const double v = strtod(field, &endptr);
        if (endptr == field || (endptr != p && '\0' != *endptr
          && NULL == strchr(delims, *endptr)))
        {
          ok = 0;
        }
        else {
          vals[i] = v;
          ++filled;
        }
      }
    }
    if ('\0' != *p) ++p;
    ++col;
  }
  return (0 != ok && filled >= nidx) ? 1 : 0;
}


LIBXS_API_INLINE int internal_libxs_predict_tokenize(
  const char spec[], char buf[], const char* tokens[], int maxtokens)
{
  int n = 0;
  const char* p = spec;
  char* dst = buf;
  while ('\0' != *p && n < maxtokens) {
    while (' ' == *p || '\t' == *p) ++p;
    if ('\0' == *p) break;
    tokens[n] = dst;
    while ('\0' != *p && ',' != *p) {
      *dst++ = *p++;
    }
    while (dst > tokens[n] && (' ' == dst[-1] || '\t' == dst[-1])) --dst;
    *dst++ = '\0';
    ++n;
    if (',' == *p) ++p;
  }
  return n;
}


LIBXS_API int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char inputs[], const char outputs[])
{
  int result = -1;
  FILE* file;
  const int ninputs = model->ninputs;
  const int noutputs = model->noutputs;
  LIBXS_ASSERT(NULL != model && NULL != filename);
  file = fopen(filename, "r");
  if (NULL != file) {
    char line[4096];
    double vals[128];
    double* inp = vals;
    double* outp = vals + ninputs;
    const char* sep = delims;
    const char* input_tokens[64];
    const char* output_tokens[64];
    char tokbuf[2048];
    int idx[128];
    int i, resolved = 1;
    int ni = 0, no = 0;
    LIBXS_ASSERT(ninputs + noutputs <= 128);
    if (NULL != inputs) {
      ni = internal_libxs_predict_tokenize(inputs, tokbuf, input_tokens, 64);
    }
    if (NULL != outputs) {
      no = internal_libxs_predict_tokenize(outputs,
        tokbuf + sizeof(tokbuf) / 2, output_tokens, 64);
    }
    LIBXS_ASSERT((NULL == inputs || ni == ninputs)
      && (NULL == outputs || no == noutputs));
    result = 0;
    while (NULL != fgets(line, (int)sizeof(line), file) && '#' == line[0]) {}
    if (NULL == inputs || NULL == outputs) {
      for (i = 0; i < ninputs; ++i) {
        idx[i] = (NULL != inputs)
          ? (int)strtol(input_tokens[i], NULL, 10) : i;
      }
      for (i = 0; i < noutputs; ++i) {
        idx[ninputs + i] = (NULL != outputs)
          ? (int)strtol(output_tokens[i], NULL, 10) : ninputs + i;
      }
      if (NULL == sep && '\0' != line[0]) {
        size_t len = strlen(line);
        if (0 < len && '\n' == line[len - 1]) line[--len] = '\0';
        if (0 < len && '\r' == line[len - 1]) line[--len] = '\0';
        sep = internal_libxs_predict_detect_delims(line);
      }
      if (NULL == sep) sep = ",";
      resolved = 0;
    }
    if (0 != resolved && NULL == sep && '\0' != line[0]) {
      size_t len = strlen(line);
      if (0 < len && '\n' == line[len - 1]) line[--len] = '\0';
      if (0 < len && '\r' == line[len - 1]) line[--len] = '\0';
      sep = internal_libxs_predict_detect_delims(line);
      { int ncols = 1, nextra = 0;
        const char* cp = line;
        while ('\0' != *cp) { if (NULL != strchr(sep, *cp)) ++ncols; ++cp; }
        for (i = 0; i < ninputs && 0 != resolved; ++i) {
          idx[i] = internal_libxs_predict_resolve_col(input_tokens[i], line, sep);
          if (0 > idx[i]) resolved = 0;
        }
        for (i = 0; i < noutputs && 0 != resolved; ++i) {
          idx[ninputs + i] = internal_libxs_predict_resolve_col(
            output_tokens[i], line, sep);
          if (0 > idx[ninputs + i]) {
            idx[ninputs + i] = ncols + nextra;
            ++nextra;
          }
        }
      }
      if (0 == resolved) {
        rewind(file);
        for (i = 0; i < ninputs; ++i) {
          idx[i] = (NULL != inputs)
            ? (int)strtol(input_tokens[i], NULL, 10) : i;
        }
        for (i = 0; i < noutputs; ++i) {
          idx[ninputs + i] = (NULL != outputs)
            ? (int)strtol(output_tokens[i], NULL, 10) : ninputs + i;
        }
      }
    }
    else if (0 != resolved) {
      if (NULL == sep) sep = ";";
      for (i = 0; i < ninputs; ++i) {
        idx[i] = (NULL != inputs)
          ? (int)strtol(input_tokens[i], NULL, 10) : i;
      }
      for (i = 0; i < noutputs; ++i) {
        idx[ninputs + i] = (NULL != outputs)
          ? (int)strtol(output_tokens[i], NULL, 10) : ninputs + i;
      }
    }
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len;
      if ('#' == line[0]) continue;
      len = strlen(line);
      while (0 < len && ('\n' == line[len - 1] || '\r' == line[len - 1])) line[--len] = '\0';
      if (0 != internal_libxs_predict_parse_row(line, sep, idx, ninputs, inp)
        && 0 != internal_libxs_predict_parse_row(line, sep, idx + ninputs, noutputs, outp))
      {
        if (EXIT_SUCCESS == libxs_predict_push(NULL, model, inp, outp)) {
          ++result;
        }
      }
    }
    fclose(file);
  }
  return result;
}
