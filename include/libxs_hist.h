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


typedef void (*libxs_hist_update_fn)(double* /*dst*/, const double* /*src*/);
typedef double (*libxs_hist_adjust_fn)(double /*value*/, int count);

LIBXS_API void libxs_hist_create(void** hist,
  int nbuckets, int nqueue, int nvals, const libxs_hist_update_fn update[]);

LIBXS_API void libxs_hist_avg(double* dst, const double* src);

LIBXS_API void libxs_hist_add(double* dst, const double* src);

LIBXS_API void libxs_hist_set(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, void* hist, const double vals[]);

LIBXS_API void libxs_hist_get(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock, void* hist,
  const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals);

LIBXS_API void libxs_hist_print(FILE* stream, void* hist, const char title[], const int prec[],
  const libxs_hist_adjust_fn adjust[]);

LIBXS_API void libxs_hist_free(void* hist);

#endif /*LIBXS_HIST_H*/
