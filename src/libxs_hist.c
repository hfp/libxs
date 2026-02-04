/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_hist.h>


typedef struct libxs_hist_t {
  libxs_hist_update_t* update;
  double *vals, min, max;
  int *buckets, nbuckets, nqueue, nvals, n;
} libxs_hist_t;


LIBXS_API void libxs_hist_create(void** hist,
  int nbuckets, int nqueue, int nvals, const libxs_hist_update_t update[])
{
  libxs_hist_t* h = (libxs_hist_t*)malloc(sizeof(libxs_hist_t));
  assert(NULL != hist && 0 < nbuckets && 0 < nqueue && 0 < nvals && NULL != update);
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


LIBXS_API void libxs_hist_destroy(void* hist)
{
  if (NULL != hist) {
    libxs_hist_t* const h = (libxs_hist_t*)hist;
    free(h->buckets);
    free(h->update);
    free(h->vals);
    free(h);
  }
}


LIBXS_API void libxs_hist_set(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, void* hist, const double vals[])
{
  if (NULL != hist) {
    libxs_hist_t* const h = (libxs_hist_t*)hist;
    int i, j, k;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (h->nqueue <= h->n) {
      const double *values, w = h->max - h->min;
      const int* buckets;
      if (h->nqueue == h->n) {
        libxs_hist_get(NULL /*lock*/, hist, &buckets, NULL /*nbuckets*/, NULL /*range*/, &values, NULL /*nvals*/);
      }
      for (i = 1; i <= h->nbuckets; ++i) {
        const double q = h->min + i * w / h->nbuckets;
        if (vals[0] <= q || h->nbuckets == i) {
          for (k = 0, j = (i - 1) * h->nvals; k < h->nvals; ++k) {
            if (0 != h->buckets[i - 1]) {
              if (NULL != h->update[k]) {
                const libxs_hist_update_t update = h->update[k];
                update(h->vals + (j + k), vals + k);
              }
              else libxs_hist_avg(h->vals + (j + k), vals + k);
            }
            else h->vals[j + k] = vals[k]; /* initialize */
          }
          ++h->buckets[i - 1];
          break;
        }
      }
    }
    else { /* fill-phase */
      if (h->min > vals[0]) h->min = vals[0];
      if (h->max < vals[0]) h->max = vals[0];
      for (k = 0, j = h->nvals * h->n; k < h->nvals; ++k) {
        h->vals[j + k] = vals[k];
      }
    }
    ++h->n; /* count number of accumulated values */
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_hist_get(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, void* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals)
{
  int *b = NULL, m = 0, n = 0, i, j, k;
  double *v = NULL, r[] = {0, 0};
  assert(NULL != buckets || NULL != range || NULL != vals);
  if (NULL != hist) {
    libxs_hist_t* const h = (libxs_hist_t*)hist;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (h->n <= h->nqueue) {
      const double w = h->max - h->min;
      if (h->n < h->nbuckets) h->nbuckets = h->n;
      for (i = 1, j = 0; i <= h->nbuckets; j = h->nvals * i++) {
        const double p = h->min + (i - 1) * w / h->nbuckets, q = h->min + i * w / h->nbuckets;
        for (n = 0, m = 0; n < h->n; m = ++n * h->nvals) {
          if (0 == h->buckets[n] && (p < h->vals[m] || 1 == i) && (h->vals[m] <= q || h->nbuckets == i)) {
            if (j != m) {
              if (0 != h->buckets[i - 1]) { /* accumulate */
                for (k = 0; k < h->nvals; ++k) {
                  (NULL != h->update[k] ? h->update[k] : libxs_hist_avg)(h->vals + (j + k), h->vals + (m + k));
                }
              }
              else { /* initialize/swap */
                for (k = 0; k < h->nvals; ++k) {
                  const double value = h->vals[m + k];
                  h->vals[m + k] = h->vals[j + k];
                  h->vals[j + k] = value;
                }
              }
            }
            ++h->buckets[i - 1];
          }
        }
      }
      h->nqueue = 0;
    }
    if (0 < h->n) {
      r[0] = h->min;
      r[1] = h->max;
      b = h->buckets;
      n = h->nbuckets;
      v = h->vals;
      m = h->nvals;
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


LIBXS_API void libxs_hist_print(FILE* stream, void* hist, const char title[], const int prec[],
  const libxs_hist_adjust_t adjust[])
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
  assert(NULL != dst && NULL != src);
  *dst = 0.5 * (*dst + *src);
}


LIBXS_API void libxs_hist_add(double* dst, const double* src)
{
  assert(NULL != dst && NULL != src);
  *dst += *src;
}
