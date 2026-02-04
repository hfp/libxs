/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_hist.h>


struct libxs_hist_t {
  libxs_hist_update_t* update;
  double *vals, min, max;
  int *buckets, nbuckets, nqueue, nvals, n;
};


LIBXS_API void libxs_hist_create(libxs_hist_t** hist,
  int nbuckets, int nqueue, int nvals, const libxs_hist_update_t update[])
{
  libxs_hist_t* h = (libxs_hist_t*)malloc(sizeof(libxs_hist_t));
  LIBXS_ASSERT(NULL != hist && 0 < nbuckets && 0 < nqueue && 0 < nvals && NULL != update);
  if (NULL != h) {
    h->vals = (double*)malloc(sizeof(double) * LIBXS_MAX(nbuckets, nqueue) * nvals);
    h->update = (libxs_hist_update_t*)malloc(sizeof(libxs_hist_update_t) * nvals);
    h->buckets = (int*)calloc(nbuckets, sizeof(int));
    if (NULL != h->vals && NULL != h->buckets && NULL != h->update) {
      union {
        int raw;
        float value;
      } inf = {0};
#  if defined(INFINITY) && /*overflow warning*/ !defined(_CRAYC)
      inf.value = (float)(INFINITY);
#  else
      inf.raw = 0x7F800000;
#  endif
      h->min = +inf.value;
      h->max = -inf.value;
      h->nbuckets = nbuckets;
      h->nqueue = nqueue;
      h->nvals = nvals;
      /* if update[] is NULL, libxs_hist_avg is assumed */
      for (h->n = 0; h->n < nvals; ++h->n) h->update[h->n] = update[h->n];
      h->n = 0;
    }
    else {
      free(h->buckets);
      free(h->vals);
      free(h);
      h = NULL;
    }
  }
  *hist = h;
}


LIBXS_API void libxs_hist_destroy(libxs_hist_t* hist)
{
  if (NULL != hist) {
    free(hist->buckets);
    free(hist->update);
    free(hist->vals);
    free(hist);
  }
}


LIBXS_API void libxs_hist_set(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, libxs_hist_t* hist, const double vals[])
{
  if (NULL != hist) {
    int i, j, k;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (hist->nqueue <= hist->n) {
      const double *values, w = hist->max - hist->min;
      const int* buckets;
      if (hist->nqueue == hist->n) {
        libxs_hist_get(NULL /*lock*/, hist, &buckets, NULL /*nbuckets*/, NULL /*range*/, &values, NULL /*nvals*/);
      }
      for (i = 1; i <= hist->nbuckets; ++i) {
        const double q = hist->min + i * w / hist->nbuckets;
        if (vals[0] <= q || hist->nbuckets == i) {
          for (k = 0, j = (i - 1) * hist->nvals; k < hist->nvals; ++k) {
            if (0 != hist->buckets[i - 1]) {
              if (NULL != hist->update[k]) {
                const libxs_hist_update_t update = hist->update[k];
                update(hist->vals + (j + k), vals + k);
              }
              else libxs_hist_avg(hist->vals + (j + k), vals + k);
            }
            else hist->vals[j + k] = vals[k]; /* initialize */
          }
          ++hist->buckets[i - 1];
          break;
        }
      }
    }
    else { /* fill-phase */
      if (hist->min > vals[0]) hist->min = vals[0];
      if (hist->max < vals[0]) hist->max = vals[0];
      for (k = 0, j = hist->nvals * hist->n; k < hist->nvals; ++k) {
        hist->vals[j + k] = vals[k];
      }
    }
    ++hist->n; /* count number of accumulated values */
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_hist_get(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, const libxs_hist_t* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals)
{
  int *b = NULL, m = 0, n = 0, i, j, k;
  double *v = NULL, r[] = {0, 0};
  LIBXS_ASSERT(NULL != buckets || NULL != range || NULL != vals);
  if (NULL != hist) {
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (hist->n <= hist->nqueue) {
      const double w = hist->max - hist->min;
      if (hist->n < hist->nbuckets) ((libxs_hist_t*)hist)->nbuckets = hist->n;
      for (i = 1, j = 0; i <= hist->nbuckets; j = hist->nvals * i++) {
        const double p = hist->min + (i - 1) * w / hist->nbuckets, q = hist->min + i * w / hist->nbuckets;
        for (n = 0, m = 0; n < hist->n; m = ++n * hist->nvals) {
          if (0 == hist->buckets[n] && (p < hist->vals[m] || 1 == i) && (hist->vals[m] <= q || hist->nbuckets == i)) {
            if (j != m) {
              if (0 != hist->buckets[i - 1]) { /* accumulate */
                for (k = 0; k < hist->nvals; ++k) {
                  (NULL != hist->update[k] ? hist->update[k] : libxs_hist_avg)(hist->vals + (j + k), hist->vals + (m + k));
                }
              }
              else { /* initialize/swap */
                for (k = 0; k < hist->nvals; ++k) {
                  const double value = hist->vals[m + k];
                  hist->vals[m + k] = hist->vals[j + k];
                  hist->vals[j + k] = value;
                }
              }
            }
            ++hist->buckets[i - 1];
          }
        }
      }
      ((libxs_hist_t*)hist)->nqueue = 0;
    }
    if (0 < hist->n) {
      r[0] = hist->min;
      r[1] = hist->max;
      b = hist->buckets;
      n = hist->nbuckets;
      v = hist->vals;
      m = hist->nvals;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
  if (NULL != nbuckets) *nbuckets = n;
  if (NULL != buckets) *buckets = b;
  if (NULL != nvals) *nvals = m;
  if (NULL != vals) *vals = v;
  if (NULL != range) {
    range[0] = r[0];
    range[1] = r[1];
  }
}


LIBXS_API void libxs_hist_print(FILE* stream, const libxs_hist_t* hist, const char title[],
  const int prec[], const libxs_hist_adjust_t adjust[])
{
  int nbuckets = 0, nvals = 0, i = 1, j = 0, k;
  const int* buckets = NULL;
  const double* vals = NULL;
  double range[2];
  libxs_hist_get(NULL /*lock*/, hist, &buckets, &nbuckets, range, &vals, &nvals);
  if (NULL != stream && NULL != buckets && 0 < nbuckets && NULL != vals && 0 < nvals) {
    const double w = range[1] - range[0];
    if (NULL != title) fprintf(stream, "%s pid=%u\n", title, libxs_pid());
    for (; i <= nbuckets; j = nvals * i++) {
      const double q = range[0] + i * w / nbuckets, r = (i != nbuckets ? q : LIBXS_MAX(q, vals[j]));
      const int c = buckets[i - 1];
      if (NULL != prec) fprintf(stream, "\t#%i <= %.*f: %i", i, prec[0], r, c);
      else fprintf(stream, "\t#%i <= %f: %i", i, r, c);
      if (0 != c) {
        fprintf(stream, " ->");
        for (k = 0; k < nvals; ++k) {
          double value;
          if (NULL == adjust || NULL == adjust[k]) value = vals[j + k];
          else value = adjust[k](vals[j + k], c);
          if (NULL != prec) fprintf(stream, " %.*f", prec[k], value);
          else fprintf(stream, " %f", value);
        }
      }
      fprintf(stream, "\n");
    }
  }
}


LIBXS_API void libxs_hist_avg(double* dst, const double* src)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  *dst = 0.5 * (*dst + *src);
}


LIBXS_API void libxs_hist_add(double* dst, const double* src)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  *dst += *src;
}
