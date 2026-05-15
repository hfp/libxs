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


typedef struct internal_libxs_predict_entry_t {
  double* inputs;
  double* outputs;
} internal_libxs_predict_entry_t;

typedef struct internal_libxs_predict_cluster_t {
  double* centroid;
  double* coeffs;
  double* errors;
  double* kd_pts;
  int* order;
  int* reliable;
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
      free(cl->order);
      free(cl->reliable);
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


LIBXS_API int libxs_predict_build(libxs_predict_t* model, int nclusters, double quality)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || 0 >= model->nentries) {
    result = EXIT_FAILURE;
  }
  else {
    const int p = model->nentries;
    const int m = model->ninputs;
    const int n = model->noutputs;
    int c, i;
    if (quality < 0.0) quality = 0.0;
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
      if (NULL == cl->sorted_idx || NULL == cl->sorted_dist
        || NULL == cl->order || NULL == cl->reliable)
      {
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS == result) {
        int ki = 0;
        int* entry_map = (int*)malloc((size_t)nc * sizeof(int));
        double* outmat = (double*)malloc((size_t)nc * (size_t)n * sizeof(double));
        int* smooth_perm = (int*)malloc((size_t)nc * sizeof(int));
        if (NULL == entry_map || NULL == outmat || NULL == smooth_perm) {
          free(entry_map); free(outmat); free(smooth_perm);
          result = EXIT_FAILURE;
        }
        else {
          for (i = 0; i < p; ++i) {
            if (model->assignments[i] == c) {
              entry_map[ki] = i;
              ++ki;
            }
          }
          for (k = 0; k < nc; ++k) {
            for (j = 0; j < n; ++j) {
              outmat[(size_t)j * nc + k] = model->entries[entry_map[k]].outputs[j];
            }
          }
          libxs_sort_smooth(LIBXS_SORT_GREEDY, nc, n, outmat, nc,
            LIBXS_DATATYPE_F64, smooth_perm);
          for (k = 0; k < nc; ++k) {
            cl->sorted_idx[k] = entry_map[smooth_perm[k]];
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
          free(entry_map);
          free(outmat);
          free(smooth_perm);
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
        if (NULL == cl->coeffs || NULL == cl->errors) {
          result = EXIT_FAILURE;
        }
      }
      if (EXIT_SUCCESS == result) {
        int pool_seq = 0, pool_buf = 0;
        double* seq = (double*)libxs_malloc(internal_libxs_default_pool,
          (size_t)nc * sizeof(double), LIBXS_MALLOC_AUTO);
        double* buf = (double*)libxs_malloc(internal_libxs_default_pool,
          2 * (size_t)nc * sizeof(double), LIBXS_MALLOC_AUTO);
        if (NULL != seq) pool_seq = 1;
        else seq = (double*)malloc((size_t)nc * sizeof(double));
        if (NULL != buf) pool_buf = 1;
        else buf = (double*)malloc(2 * (size_t)nc * sizeof(double));
        if (NULL == seq || NULL == buf) {
          result = EXIT_FAILURE;
        }
        else {
          const size_t shape = (size_t)nc;
          const double decay_threshold = 0.01 + 0.99 * quality;
          for (j = 0; j < n && EXIT_SUCCESS == result; ++j) {
            int trunc_order = maxorder;
            int d;
            libxs_fprint_t fp;
            for (k = 0; k < nc; ++k) {
              seq[k] = model->entries[cl->sorted_idx[k]].outputs[j];
            }
            libxs_fprint(&fp, LIBXS_DATATYPE_F64, seq, 1, &shape, NULL,
              LIBXS_FPRINT_MAXORDER, -1);
            cl->reliable[j] = 0;
            if (0 < fp.l2[0]) {
              const double decay = libxs_fprint_decay(&fp);
              if (decay < 1.0) {
                cl->reliable[j] = 1;
                for (d = 1; d <= trunc_order; ++d) {
                  if (fp.l2[d] >= decay_threshold * fp.l2[0]) {
                    trunc_order = LIBXS_MAX(d - 1, 1);
                    d = maxorder + 1;
                  }
                }
              }
            }
            cl->order[j] = trunc_order;
            memcpy(buf, seq, (size_t)nc * sizeof(double));
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
        }
        if (0 != pool_seq) libxs_free(seq); else free(seq);
        if (0 != pool_buf) libxs_free(buf); else free(buf);
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
      const double t = internal_libxs_predict_position(model, cl, inputs);
      const int pos = (int)(t + 0.5);
      for (j = 0; j < n; ++j) {
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
        const double t = internal_libxs_predict_position(model, cl, inputs);
        const int pos = (int)(t + 0.5);
        const double w = ((dq > 0) ? (1.0 / dq) : 1e30) / wsum;
        for (j = 0; j < n; ++j) {
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


LIBXS_API int libxs_predict_save(const libxs_predict_t* model, void* buffer, size_t* size)
{
  int result = EXIT_SUCCESS;
  if (NULL == model || 0 == model->built || NULL == size) {
    result = EXIT_FAILURE;
  }
  else {
    size_t required = 0;
    int c, j;
    required += 5 * sizeof(uint32_t);
    for (c = 0; c < model->nclusters; ++c) {
      const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
      required += (size_t)model->ninputs * sizeof(double);
      required += sizeof(double);
      required += 2 * sizeof(uint32_t);
      required += (size_t)model->noutputs * sizeof(int32_t);
      required += (size_t)model->noutputs * sizeof(int32_t);
      required += (size_t)model->noutputs * sizeof(double);
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
#define WRITE_U32(V) \
  do { \
    const uint32_t v_ = (uint32_t)(V); \
    memcpy(dst, &v_, sizeof(uint32_t)); \
    dst += sizeof(uint32_t); \
  } while (0)
#define WRITE_F64(V) \
  do { \
    const double v_ = (V); \
    memcpy(dst, &v_, sizeof(double)); \
    dst += sizeof(double); \
  } while (0)
#define WRITE_BLK(PTR, SZ) \
  do { \
    memcpy(dst, (PTR), (SZ)); \
    dst += (SZ); \
  } while (0)
      WRITE_U32(LIBXS_PREDICT_MAGIC);
      WRITE_U32(LIBXS_PREDICT_VERSION);
      WRITE_U32(model->ninputs);
      WRITE_U32(model->noutputs);
      WRITE_U32(model->nclusters);
      for (c = 0; c < model->nclusters; ++c) {
        const internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        WRITE_BLK(cl->centroid, (size_t)model->ninputs * sizeof(double));
        WRITE_F64(cl->dmax);
        WRITE_U32(cl->nentries);
        WRITE_U32(cl->maxorder);
        WRITE_BLK(cl->order, (size_t)model->noutputs * sizeof(int32_t));
        WRITE_BLK(cl->reliable, (size_t)model->noutputs * sizeof(int32_t));
        WRITE_BLK(cl->errors, (size_t)model->noutputs * sizeof(double));
        for (j = 0; j < model->noutputs; ++j) {
          WRITE_BLK(
            cl->coeffs + (size_t)j * (cl->maxorder + 1), (size_t)(cl->order[j] + 1) * sizeof(double));
        }
      }
#undef WRITE_U32
#undef WRITE_F64
#undef WRITE_BLK
      *size = (size_t)(dst - (unsigned char*)buffer);
    }
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_read_u32(
  const unsigned char** src, const unsigned char* end, uint32_t* val)
{
  int result = EXIT_SUCCESS;
  if (*src + sizeof(uint32_t) > end) {
    result = EXIT_FAILURE;
  }
  else {
    memcpy(val, *src, sizeof(uint32_t));
    *src += sizeof(uint32_t);
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_read_f64(
  const unsigned char** src, const unsigned char* end, double* val)
{
  int result = EXIT_SUCCESS;
  if (*src + sizeof(double) > end) {
    result = EXIT_FAILURE;
  }
  else {
    memcpy(val, *src, sizeof(double));
    *src += sizeof(double);
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_read_blk(
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
  if (NULL != buffer && size >= 5 * sizeof(uint32_t)) {
    const unsigned char* src = (const unsigned char*)buffer;
    const unsigned char* end = src + size;
    uint32_t magic = 0, version = 0, ninp = 0, nout = 0, nclust = 0;
    int ok = EXIT_SUCCESS;
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read_u32(&src, end, &magic);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read_u32(&src, end, &version);
    if (EXIT_SUCCESS == ok && (magic != LIBXS_PREDICT_MAGIC || version != LIBXS_PREDICT_VERSION)) {
      ok = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read_u32(&src, end, &ninp);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read_u32(&src, end, &nout);
    if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read_u32(&src, end, &nclust);
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
    {
      int c;
      for (c = 0; c < (int)nclust && EXIT_SUCCESS == ok; ++c) {
        internal_libxs_predict_cluster_t* cl = &model->clusters[c];
        int j;
        cl->centroid = (double*)malloc((size_t)ninp * sizeof(double));
        cl->order = (int*)malloc((size_t)nout * sizeof(int));
        cl->reliable = (int*)malloc((size_t)nout * sizeof(int));
        cl->errors = (double*)malloc((size_t)nout * sizeof(double));
        if (NULL == cl->centroid || NULL == cl->order || NULL == cl->reliable ||
            NULL == cl->errors)
          ok = EXIT_FAILURE;
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_blk(
            &src, end, cl->centroid, (size_t)ninp * sizeof(double));
        }
        if (EXIT_SUCCESS == ok) ok = internal_libxs_predict_read_f64(&src, end, &cl->dmax);
        if (EXIT_SUCCESS == ok) {
          uint32_t ne;
          ok = internal_libxs_predict_read_u32(&src, end, &ne);
          if (EXIT_SUCCESS == ok) cl->nentries = (int)ne;
        }
        if (EXIT_SUCCESS == ok) {
          uint32_t mo;
          ok = internal_libxs_predict_read_u32(&src, end, &mo);
          if (EXIT_SUCCESS == ok) cl->maxorder = (int)mo;
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_blk(
            &src, end, cl->order, (size_t)nout * sizeof(int32_t));
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_blk(
            &src, end, cl->reliable, (size_t)nout * sizeof(int32_t));
        }
        if (EXIT_SUCCESS == ok) {
          ok = internal_libxs_predict_read_blk(
            &src, end, cl->errors, (size_t)nout * sizeof(double));
        }
        if (EXIT_SUCCESS == ok) {
          cl->coeffs = (double*)calloc(
            (size_t)nout * (size_t)(cl->maxorder + 1), sizeof(double));
          if (NULL == cl->coeffs) ok = EXIT_FAILURE;
          for (j = 0; j < (int)nout && EXIT_SUCCESS == ok; ++j) {
            ok = internal_libxs_predict_read_blk(&src, end,
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
  const int inputs_idx[], int ninputs,
  const int outputs_idx[], int noutputs)
{
  int result = -1;
  FILE* file;
  LIBXS_ASSERT(NULL != model && NULL != filename);
  LIBXS_ASSERT(NULL != inputs_idx && NULL != outputs_idx);
  LIBXS_ASSERT(ninputs == model->ninputs && noutputs == model->noutputs);
  file = fopen(filename, "r");
  if (NULL != file) {
    char line[4096];
    double vals[128];
    double* inp = vals;
    double* outp = vals + ninputs;
    const char* sep = delims;
    LIBXS_ASSERT(ninputs + noutputs <= 128);
    result = 0;
    if (NULL == sep && NULL != fgets(line, (int)sizeof(line), file)) {
      sep = internal_libxs_predict_detect_delims(line);
      if (0 != internal_libxs_predict_parse_row(line, sep, inputs_idx, ninputs, inp)
        && 0 != internal_libxs_predict_parse_row(line, sep, outputs_idx, noutputs, outp))
      {
        if (EXIT_SUCCESS == libxs_predict_push(NULL, model, inp, outp)) {
          ++result;
        }
      }
    }
    else if (NULL == sep) {
      sep = ";";
    }
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      if (0 != internal_libxs_predict_parse_row(line, sep, inputs_idx, ninputs, inp)
        && 0 != internal_libxs_predict_parse_row(line, sep, outputs_idx, noutputs, outp))
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
