/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_TAS_H
#define LIBXS_TAS_H

#include "libxs_gemm.h"


/**
 * Plan of one column-major GEMM, C := alpha * op(A) * op(B) + beta * C, that is
 * tall and skinny (TAS): bound by memory bandwidth rather than by compute, the
 * GEMM runs along its longest dimension as a constant-stride batch of panels
 * plus one remainder. Along K, each chunk writes a partial C of its own, and
 * the partial results are summed in a fixed order, i.e., the result does not
 * depend on the number of tasks. Any other GEMM is planned as one call (dim is
 * zero). A plan is caller-owned, holds the scratch of the partial results, and
 * serves one call at a time; release it with libxs_tas_release.
 */
typedef struct libxs_tas_t {
  /** The whole GEMM. */
  libxs_gemm_shape_t shape;
  /** Configs of a full panel and of the remainder (rest.shape.m is zero if none). */
  libxs_gemm_config_t panel, rest;
  /** Element strides between panels, and element offsets of the remainder. */
  int stride_a, stride_b, stride_c;
  size_t rest_a, rest_b, rest_c;
  /** Number of full panels, and number of partial C (zero unless along K). */
  int npanels, nscratch;
  /** Dimension the plan runs along ('M', 'N', or 'K'), zero for one call. */
  char dim;
  /** Count of finished tasks followed by the partial C (along K), NULL otherwise. */
  void* scratch;
} libxs_tas_t;


/**
 * Plan a GEMM (runtime, explicit backend selection), F64 or F32. The panel is
 * the extent of a panel along M or N, or of a chunk along K; zero takes
 * LIBXS_TAS_PANEL or LIBXS_TAS_CHUNK. A GEMM is planned along its longest
 * dimension if its arithmetic intensity (FLOPs per Byte) is below LIBXS_TAS_MAX,
 * and if this yields at least two panels. Panels are dispatched with
 * LIBXS_GEMM_FLAG_STREAM and LIBXS_GEMM_FLAG_REUSE, a plan of one call without
 * hints. Returns nonzero if plan was populated, zero otherwise.
 */
LIBXS_API int libxs_tas_dispatch_rt(libxs_tas_t* plan,
  const libxs_gemm_shape_t* shape, int panel,
  const libxs_gemm_backend_t* backend, void* registry);

/**
 * Plan a GEMM (see libxs_tas_dispatch_rt), with the backend selected at the
 * caller's compile time (see libxs_gemm_backend_init).
 */
LIBXS_API_INLINE int libxs_tas_dispatch(libxs_tas_t* plan,
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta, int panel,
  void* LIBXS_ARGDEF(registry, NULL))
{
  libxs_gemm_shape_t shape;
  libxs_gemm_backend_t be;
  LIBXS_MEMZERO(&shape);
  shape.datatype = datatype;
  shape.transa = transa; shape.transb = transb;
  shape.m = m; shape.n = n; shape.k = k;
  shape.lda = lda; shape.ldb = ldb; shape.ldc = ldc;
  if (LIBXS_DATATYPE_F64 == datatype) {
    shape.alpha = (NULL != alpha ? *(const double*)alpha : 1.0);
    shape.beta  = (NULL != beta  ? *(const double*)beta  : 0.0);
  }
  else if (LIBXS_DATATYPE_F32 == datatype) {
    shape.alpha = (NULL != alpha ? (double)*(const float*)alpha : 1.0);
    shape.beta  = (NULL != beta  ? (double)*(const float*)beta  : 0.0);
  }
  libxs_gemm_backend_init(&be);
  return libxs_tas_dispatch_rt(plan, &shape, panel, &be, registry);
}

/** Run the planned GEMM; a, b, and c are the operands of the whole GEMM. */
LIBXS_API void libxs_tas(const libxs_tas_t* plan,
  const void* a, const void* b, void* c);

/**
 * Per-thread form of libxs_tas. Every tid in [0, ntasks) must call, since
 * along K the last task to finish sums the partial results into C.
 */
LIBXS_API void libxs_tas_task(const libxs_tas_t* plan,
  const void* a, const void* b, void* c,
  int tid, int ntasks);

/**
 * Number of tasks worth splitting libxs_tas_task into, given nthreads
 * available (see libxs_gemm_ntasks). Returns 1 for a plan of one call.
 */
LIBXS_API int libxs_tas_ntasks(const libxs_tas_t* plan, int nthreads);

/** Release the scratch held by a plan. Safe to call with NULL. */
LIBXS_API void libxs_tas_release(libxs_tas_t* plan);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_TAS_H*/
