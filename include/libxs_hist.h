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


LIBXS_EXTERN_C typedef struct libxs_hist_t libxs_hist_t;

LIBXS_EXTERN_C typedef void (*libxs_hist_update_t)(double* /*dst*/, const double* /*src*/);

LIBXS_API void libxs_hist_create(libxs_hist_t** hist,
  int nbuckets, int nqueue, int nvals, const libxs_hist_update_t update[]);

LIBXS_API void libxs_hist_destroy(libxs_hist_t* hist);

LIBXS_API void libxs_hist_set(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, libxs_hist_t* hist, const double vals[]);

LIBXS_API void libxs_hist_get(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, const libxs_hist_t* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals);

typedef double (*libxs_hist_adjust_t)(double /*value*/, int count);

LIBXS_API void libxs_hist_print(FILE* stream, const libxs_hist_t* hist, const char title[],
  const int prec[], const libxs_hist_adjust_t adjust[]);

/** Update function (libxs_hist_update_t).*/
LIBXS_API void libxs_hist_avg(double* dst, const double* src);
/** Update function (libxs_hist_update_t).*/
LIBXS_API void libxs_hist_add(double* dst, const double* src);

#endif /*LIBXS_HIST_H*/
