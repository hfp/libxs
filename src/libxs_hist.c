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


LIBXS_API_INTERN void internal_libxs_hist_rebin(libxs_hist_t* h, double new_min, double new_max);
LIBXS_API_INTERN void internal_libxs_hist_rebin(libxs_hist_t* h, double new_min, double new_max)
{
  const double old_w = h->max - h->min, new_w = new_max - new_min;
  const int nb = h->nbuckets, nv = h->nvals;
  int i, k, start, end, step;
  LIBXS_ASSERT(0 < new_w);
  if (new_min < h->min) {
    start = nb - 1; end = -1; step = -1;
  }
  else {
    start = 0; end = nb; step = 1;
  }
  for (i = start; i != end; i += step) {
    if (0 < h->buckets[i]) {
      const double mid = h->min + (i + 0.5) * old_w / nb;
      int ni = (int)((mid - new_min) * nb / new_w);
      if (ni < 0) ni = 0;
      if (ni >= nb) ni = nb - 1;
      if (ni != i) {
        const int nj = ni * nv, oj = i * nv;
        if (0 < h->buckets[ni]) {
          const double ca = h->buckets[i], cb = h->buckets[ni];
          for (k = 0; k < nv; ++k) {
            const libxs_hist_update_t update = h->update[k];
            if (NULL == update || libxs_hist_update_avg == update) {
              h->vals[nj + k] = (cb * h->vals[nj + k] + ca * h->vals[oj + k]) / (ca + cb);
            }
            else if (libxs_hist_update_min == update) {
              if (h->vals[oj + k] < h->vals[nj + k]) h->vals[nj + k] = h->vals[oj + k];
            }
            else if (libxs_hist_update_max == update) {
              if (h->vals[oj + k] > h->vals[nj + k]) h->vals[nj + k] = h->vals[oj + k];
            }
            else {
              h->vals[nj + k] += h->vals[oj + k];
            }
          }
          h->buckets[ni] += h->buckets[i];
        }
        else {
          for (k = 0; k < nv; ++k) h->vals[nj + k] = h->vals[oj + k];
          h->buckets[ni] = h->buckets[i];
        }
        h->buckets[i] = 0;
      }
    }
  }
  h->min = new_min;
  h->max = new_max;
}


LIBXS_API void libxs_hist_push(libxs_lock_t* lock, libxs_hist_t* hist, const double vals[])
{
  if (NULL != hist) {
    int i, j, k;
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (hist->nqueue <= hist->n) {
      double w;
      if (hist->nqueue == hist->n) {
        libxs_hist_info_t info;
        libxs_hist_get(NULL /*lock*/, hist, &info);
      }
      if (vals[0] < hist->min || vals[0] > hist->max) {
        internal_libxs_hist_rebin(hist,
          vals[0] < hist->min ? vals[0] : hist->min,
          vals[0] > hist->max ? vals[0] : hist->max);
      }
      w = hist->max - hist->min;
      for (i = 1; i <= hist->nbuckets; ++i) {
        const double q = hist->min + i * w / hist->nbuckets;
        if (vals[0] <= q || hist->nbuckets == i) {
          ++hist->buckets[i - 1];
          for (k = 0, j = (i - 1) * hist->nvals; k < hist->nvals; ++k) {
            if (1 < hist->buckets[i - 1]) {
              if (NULL != hist->update[k]) {
                const libxs_hist_update_t update = hist->update[k];
                update(hist->vals + (j + k), vals + k, hist->buckets[i - 1]);
              }
              else libxs_hist_update_avg(hist->vals + (j + k), vals + k, hist->buckets[i - 1]);
            }
            else hist->vals[j + k] = vals[k];
          }
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
    if (hist->n < INT_MAX) ++hist->n;
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_hist_get(libxs_lock_t* lock, const libxs_hist_t* hist,
  libxs_hist_info_t* info)
{
  /* C "mutable": lazy commit mutates internal state via cast (safe: always heap-allocated) */
  libxs_hist_t* const h = (libxs_hist_t*)(uintptr_t)hist;
  int i, j, k, n, m;
  LIBXS_ASSERT(NULL != info);
  info->buckets = NULL;
  info->vals = NULL;
  info->range[0] = 0;
  info->range[1] = 0;
  info->nbuckets = 0;
  info->nvals = 0;
  info->nsamples = 0;
  if (NULL != h) {
    if (NULL != lock) LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, lock);
    if (0 < h->n && h->n <= h->nqueue) {
      const double w = h->max - h->min;
      if (h->n < h->nbuckets) h->nbuckets = h->n;
      /* buckets[0..nbuckets-1] are bucket counts; buckets[nbuckets..nqueue-1]
       * are zero-initialized flags consumed left-to-right per queued sample.
       * Swap-on-first-assign relocates displaced values to flagged positions. */
      for (i = 1, j = 0; i <= h->nbuckets; j = h->nvals * i++) {
        const double p = h->min + (i - 1) * w / h->nbuckets, q = h->min + i * w / h->nbuckets;
        for (n = 0, m = 0; n < h->n; m = ++n * h->nvals) {
          if (0 == h->buckets[n] && (p < h->vals[m] || 1 == i) && (h->vals[m] <= q || h->nbuckets == i)) {
            ++h->buckets[i - 1];
            if (j != m) {
              if (1 < h->buckets[i - 1]) {
                for (k = 0; k < h->nvals; ++k) {
                  const libxs_hist_update_t update = h->update[k];
                  if (NULL != update) update(h->vals + (j + k), h->vals + (m + k), h->buckets[i - 1]);
                  else libxs_hist_update_avg(h->vals + (j + k), h->vals + (m + k), h->buckets[i - 1]);
                }
              }
              else {
                for (k = 0; k < h->nvals; ++k) {
                  LIBXS_VALUE_SWAP(h->vals[m + k], h->vals[j + k]);
                }
              }
            }
          }
        }
      }
      h->nqueue = 0;
    }
    if (0 < h->n) {
      info->buckets = h->buckets;
      info->vals = h->vals;
      info->range[0] = h->min;
      info->range[1] = h->max;
      info->nbuckets = h->nbuckets;
      info->nvals = h->nvals;
      info->nsamples = h->n;
    }
    if (NULL != lock) LIBXS_LOCK_RELEASE(LIBXS_LOCK, lock);
  }
}


LIBXS_API void libxs_hist_get_percentile(libxs_lock_t* lock, const libxs_hist_t* hist,
  double percentile, double vals[])
{
  libxs_hist_info_t info;
  int i;
  LIBXS_ASSERT(NULL != vals);
  libxs_hist_get(lock, hist, &info);
  if (NULL != info.buckets && 0 < info.nbuckets) {
    const double w = info.range[1] - info.range[0];
    int total = 0, cumulative = 0;
    if (0 > percentile) percentile = 0;
    if (1 < percentile) percentile = 1;
    for (i = 0; i < info.nbuckets; ++i) total += info.buckets[i];
    if (0 < total) {
      const double target = percentile * total;
      for (i = 0; i < info.nbuckets; ++i) {
        cumulative += info.buckets[i];
        if (target <= cumulative) {
          const double fraction = (0 < info.buckets[i])
            ? (1.0 - (cumulative - target) / info.buckets[i]) : 0.5;
          const int ia = i * info.nvals, ib = (fraction < 0.5 && 0 < i)
            ? (i - 1) * info.nvals : ((fraction >= 0.5 && i + 1 < info.nbuckets)
            ? (i + 1) * info.nvals : ia);
          int k = 1;
          const double t = (ia != ib
            ? (fraction < 0.5 ? 0.5 + fraction : fraction - 0.5) : 0);
          vals[0] = info.range[0] + (i + fraction) * w / info.nbuckets;
          LIBXS_PRAGMA_DIAG_PUSH()
          LIBXS_PRAGMA_DIAG_OFF("-Warray-bounds")
          LIBXS_PRAGMA_DIAG_OFF("-Warray-bounds=")
          for (; k < info.nvals; ++k) {
            vals[k] = info.vals[ia + k] + t * (info.vals[ib + k] - info.vals[ia + k]);
          }
          LIBXS_PRAGMA_DIAG_POP()
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
  libxs_hist_info_t info;
  int i = 1, j = 0, k;
  libxs_hist_get(NULL /*lock*/, hist, &info);
  if (NULL != ostream && NULL != info.buckets && 0 < info.nbuckets && NULL != info.vals && 0 < info.nvals) {
    const double w = info.range[1] - info.range[0];
    if (NULL != fmt) {
      va_list args;
      va_start(args, fmt);
      vfprintf(ostream, fmt, args);
      va_end(args);
    }
    for (; i <= info.nbuckets; j = info.nvals * i++) {
      const double q = info.range[0] + i * w / info.nbuckets;
      const int c = info.buckets[i - 1];
      if (NULL != prec) {
        if (0 > prec[0]) continue;
        fprintf(ostream, "\t#%i <= %.*f: %i", i, prec[0], q, c);
      }
      else fprintf(ostream, "\t#%i <= %f: %i", i, q, c);
      if (0 != c) {
        fprintf(ostream, " ->");
        for (k = 0; k < info.nvals; ++k) {
          const double value = info.vals[j + k];
          if (NULL != prec && 0 > prec[k]) continue;
          if (NULL != prec) fprintf(ostream, " %.*f", prec[k], value);
          else fprintf(ostream, " %f", value);
        }
      }
      fprintf(ostream, "\n");
    }
  }
}


LIBXS_API void libxs_hist_update_avg(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src && 1 < count);
  *dst += (*src - *dst) / count;
}


LIBXS_API void libxs_hist_update_add(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  LIBXS_UNUSED(count);
  *dst += *src;
}


LIBXS_API void libxs_hist_update_min(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  LIBXS_UNUSED(count);
  if (*src < *dst) *dst = *src;
}


LIBXS_API void libxs_hist_update_max(double* dst, const double* src, int count)
{
  LIBXS_ASSERT(NULL != dst && NULL != src);
  LIBXS_UNUSED(count);
  if (*src > *dst) *dst = *src;
}
