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
#include <libxs_rng.h>
#include <libxs_str.h>
#include <libxs_malloc.h>
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
#  define LIBXS_PREDICT_KNN 5
#endif


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
  int* interp_pos;
  int* order;
  int* reliable;
  int* mode;
  int* sorted_idx;
  int* kd_idx;
  double* sorted_dist;
  double dmax;
  int nentries;
  int maxorder;
} internal_libxs_predict_cluster_t;

LIBXS_EXTERN_C struct libxs_predict_t {
  internal_libxs_predict_entry_t* entries;
  internal_libxs_predict_cluster_t* clusters;
  int* assignments;
  double* eval_buf;
  int ninputs, noutputs;
  int nentries, capacity;
  int nclusters;
  int built;
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
      free(cl->interp_pos);
      free(cl->order);
      free(cl->reliable);
      free(cl->mode);
      free(cl->sorted_idx);
      free(cl->kd_idx);
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


LIBXS_API_INLINE void internal_libxs_predict_kmeans(libxs_predict_t* model, int nclusters)
{
  const int m = model->ninputs;
  const int p = model->nentries;
  double* centroids = (double*)calloc((size_t)nclusters * (size_t)m, sizeof(double));
  double* comp = (double*)calloc((size_t)nclusters * (size_t)m, sizeof(double));
  int* counts = (int*)calloc((size_t)nclusters, sizeof(int));
  double* dists = (double*)malloc((size_t)p * sizeof(double));
  if (NULL != centroids && NULL != counts && NULL != comp && NULL != dists) {
    int c, i, j, iter;
    /* k-means++ initialization */
    memcpy(centroids, model->entries[libxs_rng_u32((unsigned int)p)].inputs,
      (size_t)m * sizeof(double));
    for (c = 1; c < nclusters; ++c) {
      double total = 0, cumul = 0, threshold;
      for (i = 0; i < p; ++i) {
        double mind = libxs_dist2(model->entries[i].inputs, centroids, m);
        int cc;
        for (cc = 1; cc < c; ++cc) {
          const double d = libxs_dist2(
            model->entries[i].inputs, centroids + (size_t)cc * m, m);
          if (d < mind) mind = d;
        }
        dists[i] = mind;
        total += mind;
      }
      threshold = libxs_rng_f64() * total;
      for (i = 0; i < p; ++i) {
        cumul += dists[i];
        if (cumul >= threshold) {
          memcpy(centroids + (size_t)c * m,
            model->entries[i].inputs, (size_t)m * sizeof(double));
          i = p;
        }
      }
    }
    /* Lloyd iterations with Kahan-compensated centroid accumulation */
    for (iter = 0; iter < LIBXS_PREDICT_MAXITER; ++iter) {
      int changed = 0;
      for (i = 0; i < p; ++i) {
        double best = libxs_dist2(model->entries[i].inputs, centroids, m);
        int bestc = 0;
        for (c = 1; c < nclusters; ++c) {
          const double d = libxs_dist2(
            model->entries[i].inputs, centroids + (size_t)c * m, m);
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
            libxs_kahan_sum(model->entries[i].inputs[j], &cen[j], &cmp[j]);
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
  free(dists);
  free(comp);
  free(centroids);
  free(counts);
}


LIBXS_API_INLINE double internal_libxs_predict_local_error(
  const libxs_predict_t* model, const internal_libxs_predict_cluster_t* cl,
  int pos, int output_j)
{
  double result = cl->errors[output_j];
  const int nc = cl->nentries;
  const int radius = LIBXS_MIN(4, nc / 2);
  if (radius >= 2) {
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
  int best_k = 0;
  if (NULL != cl->kd_pts && NULL != cl->kd_idx) {
    const int hit = libxs_kdtree_nearest(
      cl->kd_pts, cl->kd_idx, NULL, nc, m, m, inputs, DBL_MAX);
    if (0 <= hit) best_k = hit;
  }
  else {
    double best = libxs_dist2(inputs, model->entries[cl->sorted_idx[0]].inputs, m);
    int k;
    for (k = 1; k < nc; ++k) {
      const double d = libxs_dist2(inputs, model->entries[cl->sorted_idx[k]].inputs, m);
      if (d < best) { best = d; best_k = k; }
    }
  }
  return (double)best_k;
}


LIBXS_API_INLINE double internal_libxs_predict_classify(
  const internal_libxs_predict_cluster_t* cl, const double* kd_pts,
  const int* kd_idx, int nc, int m, const double* inputs, int output_j, int nouts)
{
  double candidates[LIBXS_PREDICT_KNN];
  double best_val = cl->raw_outputs[output_j];
  unsigned char used_buf[256];
  unsigned char* used = (nc <= 256) ? used_buf : (unsigned char*)calloc((size_t)nc, 1);
  int nfound = 0, best_count = 0, i, exact = 0;
  if (NULL != used) {
    if (used != used_buf) memset(used, 0, (size_t)nc);
    while (nfound < LIBXS_PREDICT_KNN && nfound < nc && 0 == exact) {
      const int hit = libxs_kdtree_nearest(
        kd_pts, kd_idx, used, nc, m, m, inputs, DBL_MAX);
      if (0 > hit) nfound = LIBXS_PREDICT_KNN;
      else {
        candidates[nfound] = cl->raw_outputs[(size_t)hit * nouts + output_j];
        if (0 == nfound && 0.0 == libxs_dist2(inputs, kd_pts + (size_t)hit * m, m)) {
          best_val = candidates[0];
          exact = 1;
        }
        used[hit] = 1;
        ++nfound;
      }
    }
    if (used != used_buf) free(used);
    if (0 == exact) {
      for (i = 0; i < nfound; ++i) {
        int cnt = 0, ii;
        for (ii = 0; ii < nfound; ++ii) {
          if (candidates[ii] == candidates[i]) ++cnt;
        }
        if (cnt > best_count) { best_count = cnt; best_val = candidates[i]; }
      }
    }
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
    free(model);
  }
}


LIBXS_API int libxs_predict_push(
  libxs_lock_t* lock, libxs_predict_t* model, const double inputs[], const double outputs[])
{
  int result = EXIT_SUCCESS;
  if (NULL == model || NULL == inputs || NULL == outputs) {
    result = EXIT_FAILURE;
  }
  else {
    const int m = model->ninputs, n = model->noutputs;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
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
    if (EXIT_SUCCESS == result) {
      internal_libxs_predict_entry_t* e = &model->entries[model->nentries];
      e->inputs = (double*)malloc((size_t)m * sizeof(double));
      e->outputs = (double*)malloc((size_t)n * sizeof(double));
      if (NULL != e->inputs && NULL != e->outputs) {
        memcpy(e->inputs, inputs, (size_t)m * sizeof(double));
        memcpy(e->outputs, outputs, (size_t)n * sizeof(double));
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


typedef struct internal_libxs_predict_quality_ctx_t {
  libxs_predict_t* model;
  int nclusters;
} internal_libxs_predict_quality_ctx_t;

LIBXS_API_INLINE double internal_libxs_predict_quality_fn(
  double quality, const void* data)
{
  const internal_libxs_predict_quality_ctx_t* ctx =
    (const internal_libxs_predict_quality_ctx_t*)data;
  double total_err = 1e30;
  if (EXIT_SUCCESS == libxs_predict_build(ctx->model, ctx->nclusters, quality)) {
    const int p = ctx->model->nentries;
    const int n = ctx->model->noutputs;
    int i, j;
    total_err = 0;
    for (i = 0; i < p; ++i) {
      double outputs[128];
      libxs_predict_eval(NULL, ctx->model,
        ctx->model->entries[i].inputs, outputs, NULL, 1);
      for (j = 0; j < n; ++j) {
        total_err += LIBXS_DELTA(outputs[j], ctx->model->entries[i].outputs[j]);
      }
    }
  }
  return total_err;
}


LIBXS_API int libxs_predict_build(libxs_predict_t* model, int nclusters, double quality)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || 0 >= model->nentries) {
    result = EXIT_FAILURE;
  }
  else if (quality < 0.0) {
    internal_libxs_predict_quality_ctx_t ctx;
    double best_quality = 0.8;
    ctx.model = model;
    ctx.nclusters = nclusters;
    libxs_gss_min(internal_libxs_predict_quality_fn, &ctx,
      0.1, 1.0, &best_quality, 20);
    result = libxs_predict_build(model, nclusters, best_quality);
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    int c, i;
    if (quality > 1.0) quality = 1.0;
    internal_libxs_predict_free_clusters(model);
    if (0 >= nclusters) {
      nclusters = (int)(sqrt((double)p) + 0.5);
      if (nclusters < 1) nclusters = 1;
    }
    if (nclusters > p) nclusters = p;
    model->assignments = (int*)calloc((size_t)p, sizeof(int));
    model->clusters = (internal_libxs_predict_cluster_t*)calloc(
      (size_t)nclusters, sizeof(internal_libxs_predict_cluster_t));
    model->eval_buf = (double*)malloc((size_t)n * 2 * sizeof(double) + (size_t)n * sizeof(int));
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
      cl->reliable = (int*)malloc((size_t)n * sizeof(int));
      cl->mode = (int*)malloc((size_t)n * sizeof(int));
      if (NULL == cl->sorted_idx || NULL == cl->sorted_dist
        || NULL == cl->order || NULL == cl->reliable || NULL == cl->mode)
      {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        int ki = 0;
        for (i = 0; i < p; ++i) {
          if (model->assignments[i] == c) {
            cl->sorted_idx[ki] = i;
            cl->sorted_dist[ki] = sqrt(
              libxs_dist2(model->entries[i].inputs, cl->centroid, m));
            ++ki;
          }
        }
        libxs_sort(cl->sorted_dist, nc, sizeof(double), libxs_cmp_f64,
          NULL);
        { double* dist_copy = (double*)malloc((size_t)nc * sizeof(double));
          int* idx_copy = (int*)malloc((size_t)nc * sizeof(int));
          if (NULL != dist_copy && NULL != idx_copy) {
            for (k = 0; k < nc; ++k) { dist_copy[k] = 0; idx_copy[k] = 0; }
            ki = 0;
            for (i = 0; i < p; ++i) {
              if (model->assignments[i] == c) {
                dist_copy[ki] = sqrt(
                  libxs_dist2(model->entries[i].inputs, cl->centroid, m));
                idx_copy[ki] = i;
                ++ki;
              }
            }
            for (k = 0; k < nc; ++k) {
              int best = 0, kk;
              for (kk = 1; kk < nc; ++kk) {
                if (dist_copy[kk] < dist_copy[best]) best = kk;
              }
              cl->sorted_idx[k] = idx_copy[best];
              cl->sorted_dist[k] = dist_copy[best];
              dist_copy[best] = DBL_MAX;
            }
          }
          else result = EXIT_FAILURE;
          free(dist_copy);
          free(idx_copy);
        }
        if (EXIT_SUCCESS == result) {
          cl->dmax = cl->sorted_dist[nc - 1];
          if (cl->dmax <= 0.0) cl->dmax = 1.0;
          cl->kd_pts = (double*)malloc((size_t)nc * (size_t)m * sizeof(double));
          cl->kd_idx = (int*)malloc((size_t)nc * sizeof(int));
          if (NULL != cl->kd_pts && NULL != cl->kd_idx) {
            for (k = 0; k < nc; ++k) {
              memcpy(cl->kd_pts + (size_t)k * m,
                model->entries[cl->sorted_idx[k]].inputs,
                (size_t)m * sizeof(double));
              cl->kd_idx[k] = k;
            }
            libxs_kdtree_build(cl->kd_pts, cl->kd_idx, nc, m, m);
          }
        }
      }
      if (EXIT_SUCCESS == result) {
        maxorder = LIBXS_MIN(nc - 1, LIBXS_FPRINT_MAXORDER);
        {
          const int qorder = (int)(quality * maxorder + 0.5);
          maxorder = LIBXS_MAX(qorder, 1);
          maxorder = LIBXS_MIN(maxorder, nc - 1);
        }
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
        const double decay_threshold = 0.01 + 0.99 * quality;
        const int ndistinct_thresh = (int)(sqrt((double)nc) + 0.5);
        for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
          int ndistinct = 0, d;
          double prev;
          double* seq = cl->raw_outputs + j;
          double buf_local[256];
          double* buf = (nc <= 256) ? buf_local
            : (double*)malloc((size_t)nc * sizeof(double));
          libxs_fprint_t fp;
          if (NULL == buf) { result = EXIT_FAILURE; continue; }
          for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)k * n + j];
          libxs_sort(buf, nc, sizeof(double), libxs_cmp_f64, NULL);
          prev = buf[0]; ndistinct = 1;
          for (k = 1; k < nc; ++k) {
            if (buf[k] != prev) { ++ndistinct; prev = buf[k]; }
          }
          for (k = 0; k < nc; ++k) {
            buf[k] = cl->raw_outputs[(size_t)k * n + j];
          }
          { const size_t stride = (size_t)n;
            libxs_fprint(&fp, LIBXS_DATATYPE_F64, seq, 1, &shape, &stride,
              LIBXS_FPRINT_MAXORDER, -1);
          }
          cl->reliable[j] = 0;
          if (ndistinct <= ndistinct_thresh || 0 == fp.l2[0]
            || libxs_fprint_decay(&fp) >= 1.0)
          {
            cl->mode[j] = 1;
            cl->order[j] = 0;
          }
          else {
            int trunc_order = maxorder;
            int* perm_j = (int*)malloc((size_t)nc * sizeof(int));
            cl->mode[j] = 0;
            cl->reliable[j] = 1;
            if (NULL == cl->interp_pos) {
              cl->interp_pos = (int*)calloc((size_t)n * (size_t)nc, sizeof(int));
            }
            if (NULL != perm_j && NULL != cl->interp_pos) {
              libxs_sort_smooth(LIBXS_SORT_GREEDY, nc, 1, buf, nc,
                LIBXS_DATATYPE_F64, perm_j);
              for (k = 0; k < nc; ++k) {
                cl->interp_pos[(size_t)j * nc + perm_j[k]] = k;
              }
              for (k = 0; k < nc; ++k) buf[k] = cl->raw_outputs[(size_t)perm_j[k] * n + j];
            }
            free(perm_j);
            { const size_t shape_j = (size_t)nc;
              libxs_fprint(&fp, LIBXS_DATATYPE_F64, buf, 1, &shape_j, NULL,
                LIBXS_FPRINT_MAXORDER, -1);
            }
            for (d = 1; d <= trunc_order; ++d) {
              if (fp.l2[d] >= decay_threshold * fp.l2[0]) {
                trunc_order = LIBXS_MAX(d - 1, 1);
                d = maxorder + 1;
              }
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
          if (buf != buf_local) free(buf);
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


LIBXS_API void libxs_predict_eval(libxs_lock_t* lock, const libxs_predict_t* model,
  const double inputs[], double outputs[], libxs_predict_info_t* info, int nblend)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built && NULL != inputs);
  {
    const int m = model->ninputs, n = model->noutputs;
    double *vals, *errs, best_dist;
    int *rels, c, j, best_c = 0;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    vals = model->eval_buf;
    errs = vals + n;
    rels = (int*)(errs + n);
    if (nblend <= 0) nblend = 0;
    if (nblend > model->nclusters) nblend = model->nclusters;
    best_dist = libxs_dist2(inputs, model->clusters[0].centroid, m);
    for (c = 1; c < model->nclusters; ++c) {
      const double d = libxs_dist2(inputs, model->clusters[c].centroid, m);
      if (d < best_dist) { best_dist = d; best_c = c; }
    }
    if (nblend <= 1) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[best_c];
      const int nearest = (int)internal_libxs_predict_position(model, cl, inputs);
      for (j = 0; j < n; ++j) {
        if (0 != cl->mode[j]) {
          vals[j] = internal_libxs_predict_classify(
            cl, cl->kd_pts, cl->kd_idx, cl->nentries, m, inputs, j, n);
          errs[j] = 0;
          rels[j] = 0;
        }
        else {
          const int pos = (NULL != cl->interp_pos)
            ? cl->interp_pos[(size_t)j * cl->nentries + nearest] : nearest;
          const double t = (double)pos;
          const int d = cl->order[j];
          const double* cj = cl->coeffs + (size_t)j * (cl->maxorder + 1);
          double val = 0;
          int k;
          for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
          vals[j] = val;
          errs[j] = (NULL != info)
            ? internal_libxs_predict_local_error(model, cl, pos, j)
            : cl->errors[j];
          rels[j] = cl->reliable[j];
        }
      }
      if (NULL != info) info->cluster = best_c;
    }
    else {
      typedef struct { double dist; int idx; } dc_t;
      dc_t dc_buf[64];
      dc_t* dists = (model->nclusters <= 64)
        ? dc_buf : (dc_t*)malloc((size_t)model->nclusters * sizeof(dc_t));
      double wsum = 0;
      int b;
      LIBXS_ASSERT(NULL != dists);
      for (c = 0; c < model->nclusters; ++c) {
        dists[c].dist = sqrt(libxs_dist2(inputs, model->clusters[c].centroid, m));
        dists[c].idx = c;
      }
      for (b = 0; b < nblend; ++b) {
        int minj = b;
        for (c = b + 1; c < model->nclusters; ++c) {
          if (dists[c].dist < dists[minj].dist) minj = c;
        }
        if (minj != b) { dc_t tmp = dists[b]; dists[b] = dists[minj]; dists[minj] = tmp; }
      }
      memset(vals, 0, (size_t)n * sizeof(double));
      memset(errs, 0, (size_t)n * sizeof(double));
      memset(rels, 0, (size_t)n * sizeof(int));
      for (b = 0; b < nblend; ++b) {
        const double w = (dists[b].dist > 0) ? (1.0 / dists[b].dist) : 1e30;
        wsum += w;
      }
      if (wsum <= 0) wsum = 1.0;
      for (b = 0; b < nblend; ++b) {
        const int ci = dists[b].idx;
        const internal_libxs_predict_cluster_t* cl = &model->clusters[ci];
        const double dq = dists[b].dist;
        const int nearest = (int)internal_libxs_predict_position(model, cl, inputs);
        const double w = ((dq > 0) ? (1.0 / dq) : 1e30) / wsum;
        for (j = 0; j < n; ++j) {
          if (0 != cl->mode[j]) {
            vals[j] += w * internal_libxs_predict_classify(
              cl, cl->kd_pts, cl->kd_idx, cl->nentries, m, inputs, j, n);
          }
          else {
            const int pos = (NULL != cl->interp_pos)
              ? cl->interp_pos[(size_t)j * cl->nentries + nearest] : nearest;
            const double t = (double)pos;
            const int d = cl->order[j];
            const double* cj = cl->coeffs + (size_t)j * (cl->maxorder + 1);
            double val = 0;
            int k;
            for (k = 0; k <= d; ++k) val += cj[k] * libxs_binom(t, k);
            vals[j] += w * val;
            errs[j] += w * ((NULL != info)
              ? internal_libxs_predict_local_error(model, cl, pos, j)
              : cl->errors[j]);
            if (0 != cl->reliable[j]) rels[j] = 1;
          }
        }
      }
      if (NULL != info) info->cluster = -1;
      if (dists != dc_buf) free(dists);
    }
    if (NULL != outputs) {
      memcpy(outputs, vals, (size_t)n * sizeof(double));
    }
    if (NULL != info) {
      info->values = vals;
      info->error = errs;
      info->reliable = rels;
      info->noutputs = n;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_predict_query(
  const libxs_predict_t* model, int* nclusters, int* nentries, double* compression)
{
  LIBXS_ASSERT(NULL != model && 0 != model->built);
  if (NULL != nclusters) *nclusters = model->nclusters;
  if (NULL != nentries) *nentries = model->nentries;
  if (NULL != compression) {
    const double raw = (double)model->nentries * (model->ninputs + model->noutputs);
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
    *compression = (compressed > 0) ? (raw / compressed) : 0;
  }
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
    required += sizeof(uint32_t) + 4 * sizeof(uint16_t);
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      required += (size_t)model->ninputs * sizeof(double);
      required += sizeof(double);
      required += sizeof(uint16_t) + sizeof(uint8_t);
      required += (size_t)model->noutputs * 3;
      required += (size_t)model->noutputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double);
      required += (size_t)cl->nentries * (size_t)model->noutputs * sizeof(uint16_t);
      for (j = 0; j < model->noutputs; ++j) {
        required += (size_t)(cl->order[j] + 1) * sizeof(double);
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
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        int k;
        WRITE_BLK(cl->centroid, (size_t)model->ninputs * sizeof(double));
        WRITE_F64(cl->dmax);
        WRITE_U16(cl->nentries);
        WRITE_U8(cl->maxorder);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->order[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->reliable[j]);
        for (j = 0; j < model->noutputs; ++j) WRITE_U8(cl->mode[j]);
        WRITE_BLK(cl->errors, (size_t)model->noutputs * sizeof(double));
        WRITE_BLK(cl->kd_pts, (size_t)cl->nentries * (size_t)model->ninputs * sizeof(double));
        WRITE_BLK(cl->raw_outputs, (size_t)cl->nentries * (size_t)model->noutputs * sizeof(double));
        for (k = 0; k < cl->nentries * model->noutputs; ++k) {
          const int v = (NULL != cl->interp_pos) ? cl->interp_pos[k] : 0;
          WRITE_U16(v);
        }
        for (j = 0; j < model->noutputs; ++j) {
          WRITE_BLK(cl->coeffs + (size_t)j * (cl->maxorder + 1),
            (size_t)(cl->order[j] + 1) * sizeof(double));
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
      model->nclusters = (int)nclust;
      model->clusters = (internal_libxs_predict_cluster_t*)calloc(
        (size_t)nclust, sizeof(internal_libxs_predict_cluster_t));
      model->eval_buf = (double*)malloc(
        (size_t)nout * 2 * sizeof(double) + (size_t)nout * sizeof(int));
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
        cl->reliable = (int*)malloc((size_t)nout * sizeof(int));
        cl->mode = (int*)malloc((size_t)nout * sizeof(int));
        cl->errors = (double*)malloc((size_t)nout * sizeof(double));
        if (NULL == cl->centroid || NULL == cl->order || NULL == cl->reliable
          || NULL == cl->mode || NULL == cl->errors) ok = EXIT_FAILURE;
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
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->order[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->reliable[j] = (int)v;
        }
        for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
          uint8_t v = 0;
          ok = internal_libxs_predict_read(&src, end, &v, 1);
          cl->mode[j] = (int)v;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read(&src, end,
            cl->errors, (size_t)nout * sizeof(double));
        }
        if (EXIT_SUCCESS == ok) {
          cl->kd_pts = (double*)malloc(
            (size_t)cl->nentries * (size_t)ninp * sizeof(double));
          cl->kd_idx = (int*)malloc((size_t)cl->nentries * sizeof(int));
          if (NULL == cl->kd_pts || NULL == cl->kd_idx) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            ok = internal_libxs_predict_read(&src, end,
              cl->kd_pts, (size_t)cl->nentries * (size_t)ninp * sizeof(double));
          }
          if (EXIT_SUCCESS == ok) {
            int kk;
            for (kk = 0; kk < cl->nentries; ++kk) cl->kd_idx[kk] = kk;
            libxs_kdtree_build(cl->kd_pts, cl->kd_idx, cl->nentries, (int)ninp, (int)ninp);
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
          cl->interp_pos = (int*)malloc(
            (size_t)cl->nentries * (size_t)nout * sizeof(int));
          if (NULL == cl->interp_pos) ok = EXIT_FAILURE;
          if (EXIT_SUCCESS == ok) {
            int k;
            for (k = 0; k < cl->nentries * (int)nout && EXIT_SUCCESS == ok; ++k) {
              uint16_t v = 0;
              ok = internal_libxs_predict_read(&src, end, &v, 2);
              cl->interp_pos[k] = (int)v;
            }
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


LIBXS_API int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char* inputs[], int ninputs,
  const char* outputs[], int noutputs)
{
  int result = -1;
  FILE* file;
  LIBXS_ASSERT(NULL != model && NULL != filename);
  LIBXS_ASSERT(NULL != inputs && NULL != outputs);
  LIBXS_ASSERT(ninputs == model->ninputs && noutputs == model->noutputs);
  file = fopen(filename, "r");
  if (NULL != file) {
    char line[4096];
    double vals[128];
    double* inp = vals;
    double* outp = vals + ninputs;
    const char* sep = delims;
    int idx[128];
    int i, resolved = 1;
    LIBXS_ASSERT(ninputs + noutputs <= 128);
    result = 0;
    if (NULL == sep && NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len = strlen(line);
      if (0 < len && '\n' == line[len - 1]) line[--len] = '\0';
      if (0 < len && '\r' == line[len - 1]) line[--len] = '\0';
      sep = internal_libxs_predict_detect_delims(line);
      for (i = 0; i < ninputs && 0 != resolved; ++i) {
        idx[i] = internal_libxs_predict_resolve_col(inputs[i], line, sep);
        if (0 > idx[i]) resolved = 0;
      }
      for (i = 0; i < noutputs && 0 != resolved; ++i) {
        idx[ninputs + i] = internal_libxs_predict_resolve_col(outputs[i], line, sep);
        if (0 > idx[ninputs + i]) resolved = 0;
      }
      if (0 == resolved) {
        rewind(file);
        for (i = 0; i < ninputs; ++i) {
          idx[i] = (int)strtol(inputs[i], NULL, 10);
        }
        for (i = 0; i < noutputs; ++i) {
          idx[ninputs + i] = (int)strtol(outputs[i], NULL, 10);
        }
      }
    }
    else {
      if (NULL == sep) sep = ";";
      for (i = 0; i < ninputs; ++i) {
        idx[i] = (int)strtol(inputs[i], NULL, 10);
      }
      for (i = 0; i < noutputs; ++i) {
        idx[ninputs + i] = (int)strtol(outputs[i], NULL, 10);
      }
    }
    while (NULL != fgets(line, (int)sizeof(line), file)) {
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
