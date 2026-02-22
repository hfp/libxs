/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_HIST_H
#define LIBXS_HIST_H

#include "libxs_sync.h"


/** Lock-type used for this domain. */
typedef LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_hist_lock_t;

/** Opaque histogram type. */
LIBXS_EXTERN_C typedef struct libxs_hist_t libxs_hist_t;

/** Per-value update function: accumulates src into dst. */
LIBXS_EXTERN_C typedef void (*libxs_hist_update_t)(double* /*dst*/, const double* /*src*/);

/** Create histogram: nbuckets resolution, nqueue for statistical accuracy, nvals per entry. */
LIBXS_API void libxs_hist_create(libxs_hist_t** hist, int nbuckets, int nqueue, int nvals,
  const libxs_hist_update_t update[]);

/** Destroy histogram (NULL is accepted). */
LIBXS_API void libxs_hist_destroy(libxs_hist_t* hist);

/** Insert values; vals[0] determines the bucket (lock can be NULL). */
LIBXS_API void libxs_hist_set(libxs_hist_lock_t* lock, libxs_hist_t* hist, const double vals[]);

/** Query statistics; commits queued items if pending (lock can be NULL). */
LIBXS_API void libxs_hist_get(libxs_hist_lock_t* lock, const libxs_hist_t* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals);

/** Adjust function: transforms value given the bucket's count. */
typedef double (*libxs_hist_adjust_t)(double /*value*/, int count);

/** Print histogram to ostream (NULL ostream is accepted). */
LIBXS_API void libxs_hist_print(FILE* ostream, const libxs_hist_t* hist, const char title[],
  const int prec[], const libxs_hist_adjust_t adjust[]);

/** Update function (libxs_hist_update_t): sliding average, arithmetic at commit. */
LIBXS_API void libxs_hist_avg(double* dst, const double* src);
/** Update function (libxs_hist_update_t).*/
LIBXS_API void libxs_hist_add(double* dst, const double* src);

#endif /*LIBXS_HIST_H*/
