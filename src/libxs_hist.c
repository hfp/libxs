/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_hist.h>
#include <libxs_mem.h>
#include <stdarg.h>


LIBXS_EXTERN_C struct libxs_hist_t {
  libxs_hist_update_t* update;
  double *vals, min, max;
  int *buckets, nbuckets, nqueue, nvals, n;
};


LIBXS_API libxs_hist_t* libxs_hist_create(int nbuckets, int nvals,
  const libxs_hist_update_t update[])
{
  libxs_hist_t* h = (libxs_hist_t*)malloc(sizeof(libxs_hist_t));
  LIBXS_ASSERT(0 < nbuckets && 0 < nvals && NULL != update);
  if (NULL != h) {
    const int nqueue = 16 * nbuckets;
    h->vals = (double*)malloc(sizeof(double) * LIBXS_MAX(nbuckets, nqueue) * nvals);
    h->update = (libxs_hist_update_t*)malloc(sizeof(libxs_hist_update_t) * nvals);
    h->buckets = (int*)calloc(LIBXS_MAX(nbuckets, nqueue), sizeof(int));
    if (NULL != h->vals && NULL != h->buckets && NULL != h->update) {
      const union { uint32_t raw; float value; } inf = { 0x7F800000U };
      h->min = +inf.value;
      h->max = -inf.value;
      h->nbuckets = nbuckets;
      h->nqueue = nqueue;
      h->nvals = nvals;
      for (h->n = 0; h->n < nvals; ++h->n) h->update[h->n] = update[h->n];
      h->n = 0;
    }
    else {
      free(h->buckets);
      free(h->update);
      free(h->vals);
      free(h);
      h = NULL;
    }
  }
  return h;
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


LIBXS_API void libxs_hist_push(libxs_lock_t* lock, libxs_hist_t* hist, const double vals[])
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
              else libxs_hist_update_avg(hist->vals + (j + k), vals + k);
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


LIBXS_API void libxs_hist_get(libxs_lock_t* lock, const libxs_hist_t* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals)
{
  /* C "mutable": lazy commit mutates internal state via cast (safe: always heap-allocated) */
  libxs_hist_t* const h = (libxs_hist_t*)(uintptr_t)hist;
  int *b = NULL, m = 0, n = 0, i, j, k;
  double *v = NULL, r[] = {0, 0};
  LIBXS_ASSERT(NULL != buckets || NULL != range || NULL != vals);
  if (NULL != h) {
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (h->n <= h->nqueue) {
      const double w = h->max - h->min;
      if (h->n < h->nbuckets) h->nbuckets = h->n;
      for (i = 1, j = 0; i <= h->nbuckets; j = h->nvals * i++) {
        const double p = h->min + (i - 1) * w / h->nbuckets, q = h->min + i * w / h->nbuckets;
        for (n = 0, m = 0; n < h->n; m = ++n * h->nvals) {
          if (0 == h->buckets[n] && (p < h->vals[m] || 1 == i) && (h->vals[m] <= q || h->nbuckets == i)) {
            if (j != m) {
              if (0 != h->buckets[i - 1]) { /* accumulate: arithmetic sum */
                for (k = 0; k < h->nvals; ++k) {
                  h->vals[j + k] += h->vals[m + k];
                }
              }
              else { /* initialize/swap */
                for (k = 0; k < h->nvals; ++k) {
                  LIBXS_VALUE_SWAP(h->vals[m + k], h->vals[j + k]);
                }
              }
            }
            ++h->buckets[i - 1];
          }
        }
      }
      /* normalize committed sums: arithmetic mean for avg (or default) */
      for (i = 0, j = 0; i < h->nbuckets; j = ++i * h->nvals) {
        if (1 < h->buckets[i]) {
          for (k = 0; k < h->nvals; ++k) {
            if (NULL == h->update[k] || libxs_hist_update_avg == h->update[k]) {
              h->vals[j + k] /= h->buckets[i];
            }
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


LIBXS_API void libxs_hist_get_percentile(libxs_lock_t* lock, const libxs_hist_t* hist,
  double percentile, double vals[])
{
  const double* v = NULL;
  const int* buckets = NULL;
  double range[2];
  int nbuckets = 0, m = 0, i;
  LIBXS_ASSERT(NULL != vals);
  libxs_hist_get(lock, hist, &buckets, &nbuckets, range, &v, &m);
  if (NULL != buckets && 0 < nbuckets) {
    const double w = range[1] - range[0];
    int total = 0, cumulative = 0;
    if (0 > percentile) percentile = 0;
    if (1 < percentile) percentile = 1;
    for (i = 0; i < nbuckets; ++i) total += buckets[i];
    if (0 < total) {
      const double target = percentile * total;
      for (i = 0; i < nbuckets; ++i) {
        cumulative += buckets[i];
        if (target <= cumulative) {
          const double fraction = (0 < buckets[i])
            ? (1.0 - (cumulative - target) / buckets[i]) : 0.5;
          const int ia = i * m, ib = (fraction < 0.5 && 0 < i)
            ? (i - 1) * m : ((fraction >= 0.5 && i + 1 < nbuckets)
            ? (i + 1) * m : ia);
          int k = 1;
          const double t = (ia != ib
            ? (fraction < 0.5 ? 0.5 + fraction : fraction - 0.5) : 0);
          vals[0] = range[0] + (i + fraction) * w / nbuckets;
          for (; k < m; ++k) {
            /* cast mutes false positive OOB warning */
            ((double*)(void*)vals)[k] = v[ia + k] + t * (v[ib + k] - v[ia + k]);
          }
          break;
        }
      }
    }
  }
}


LIBXS_API void libxs_hist_get_median(libxs_lock_t* lock, const libxs_hist_t* hist,
  double vals[])
{
  libxs_hist_get_percentile(lock, hist, 0.5, vals);
}


LIBXS_API void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist, const int prec[],
  const char fmt[], ...)
{
  int nbuckets = 0, nvals = 0, i = 1, j = 0, k;
  const int* buckets = NULL;
  const double* vals = NULL;
  double range[2];
  libxs_hist_get(NULL /*lock*/, hist, &buckets, &nbuckets, range, &vals, &nvals);
  if (NULL != ostream && NULL != buckets && 0 < nbuckets && NULL != vals && 0 < nvals) {
    const double w = range[1] - range[0];
    if (NULL != fmt) {
      va_list args;
      va_start(args, fmt);
      vfprintf(ostream, fmt, args);
      va_end(args);
    }
    for (; i <= nbuckets; j = nvals * i++) {
      const double q = range[0] + i * w / nbuckets;
      const int c = buckets[i - 1];
      if (NULL != prec) {
        if (0 > prec[0]) continue;
        fprintf(ostream, "\t#%i <= %.*f: %i", i, prec[0], q, c);
      }
      else fprintf(ostream, "\t#%i <= %f: %i", i, q, c);
      if (0 != c) {
        fprintf(ostream, " ->");
        for (k = 0; k < nvals; ++k) {
          const double value = vals[j + k];
          if (NULL != prec && 0 > prec[k]) continue;
          if (NULL != prec) fprintf(ostream, " %.*f", prec[k], value);
          else fprintf(ostream, " %f", value);
        }
      }
      fprintf(ostream, "\n");
    }
  }
}


LIBXS_API void libxs_hist_update_avg(double* dst, const double* src)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  *dst = 0.5 * (*dst + *src);
}


LIBXS_API void libxs_hist_update_add(double* dst, const double* src)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  *dst += *src;
}
