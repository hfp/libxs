/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * Bridge between LIBXS Ozaki (CPU) and LIBXSTREAM Ozaki (OpenCL).
 * Compiled once (precision-independent); wraps LIBXSTREAM types
 * behind an opaque void* handle so that ozaki.c needs no OpenCL
 * or LIBXSTREAM headers.
 */
#include "ozaki_opencl.h"


typedef struct ozaki_ocl_handle_t {
  libxs_lock_t lock;
  ozaki_context_t ctx;
  libxstream_stream_t* stream;
} ozaki_ocl_handle_t;


void* ozaki_ocl_create(
  int use_double, int kind, int verbosity, int tm, int tn, int ndecomp, int ozflags, int oztrim, int ozgroups, int maxk, int profiling)
{
  ozaki_ocl_handle_t* h = NULL;
  int ndevices = 0;
  if (EXIT_SUCCESS == libxstream_init() &&
      EXIT_SUCCESS == libxstream_device_count(&ndevices)
      /* MPI: avoid activating a particular device */
      && 0 < ndevices)
  {
    h = (ozaki_ocl_handle_t*)calloc(1, sizeof(*h));
  }
  if (NULL != h) {
    if (EXIT_SUCCESS != ozaki_init(&h->ctx, tm, tn, use_double, kind, verbosity, ndecomp, ozflags, oztrim, ozgroups, maxk, profiling)) {
      free(h);
      h = NULL;
    }
    /* Refuse the handle if fp64 was requested but the device only supports fp32.
     * Silently downgrading would cause a type mismatch: the host passes double
     * arrays while the OpenCL kernels operate on float, leading to wrong results
     * and potential memory corruption. */
    else if (use_double && !h->ctx.use_double) {
      ozaki_destroy(&h->ctx);
      free(h);
      h = NULL;
    }
    else if (EXIT_SUCCESS != libxstream_stream_create(
                               &h->stream, "ozaki_wrap", profiling ? LIBXSTREAM_STREAM_PROFILING : LIBXSTREAM_STREAM_DEFAULT))
    {
      ozaki_destroy(&h->ctx);
      free(h);
      h = NULL;
    }
  }
  return h;
}


void ozaki_ocl_release(void* handle)
{
  ozaki_ocl_handle_t* h = (ozaki_ocl_handle_t*)handle;
  if (NULL != h) {
    ozaki_destroy(&h->ctx);
    if (NULL != h->stream) libxstream_stream_destroy(h->stream);
    free(h);
  }
}


int ozaki_ocl_gemm(void* handle, char transa, char transb, int M, int N, int K, double alpha, const void* a, int lda, const void* b,
  int ldb, double beta, void* c, int ldc, libxs_hist_t* hist, int profile)
{
  int result = EXIT_FAILURE;
  ozaki_ocl_handle_t* h = (ozaki_ocl_handle_t*)handle;
  if (NULL != h) {
    const unsigned int mxcsr = LIBXS_MXCSR_GET();
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &h->lock);
    result = ozaki_gemm(&h->ctx, h->stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc, hist, profile, 0);
    /* BLAS API is synchronous: caller expects result in c upon return.
     * Must sync all streams including persistent helper streams used
     * for preprocessing (stream_a, stream_b) to prevent race conditions. */
    libxstream_stream_sync(h->stream);
    if (NULL != h->ctx.stream_a) libxstream_stream_sync(h->ctx.stream_a);
    if (NULL != h->ctx.stream_b) libxstream_stream_sync(h->ctx.stream_b);
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &h->lock);
    LIBXS_MXCSR_SET(mxcsr);
  }
  return result;
}


int ozaki_ocl_gemm_complex(void* handle, char transa, char transb, int M, int N, int K, const double* alpha, const void* a, int lda,
  const void* b, int ldb, const double* beta, void* c, int ldc)
{
  int result = EXIT_FAILURE;
  ozaki_ocl_handle_t* h = (ozaki_ocl_handle_t*)handle;
  if (NULL != h) {
    const unsigned int mxcsr = LIBXS_MXCSR_GET();
    LIBXS_LOCK_ACQUIRE(LIBXS_LOCK, &h->lock);
    result = ozaki_gemm_complex(&h->ctx, h->stream, transa, transb, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
    /* BLAS API is synchronous: caller expects result in c upon return.
     * Must sync all streams including persistent helper streams used
     * for preprocessing (stream_a, stream_b) to prevent race conditions. */
    libxstream_stream_sync(h->stream);
    if (NULL != h->ctx.stream_a) libxstream_stream_sync(h->ctx.stream_a);
    if (NULL != h->ctx.stream_b) libxstream_stream_sync(h->ctx.stream_b);
    LIBXS_LOCK_RELEASE(LIBXS_LOCK, &h->lock);
    LIBXS_MXCSR_SET(mxcsr);
  }
  return result;
}


int ozaki_ocl_supports_gemm_complex(void* handle)
{
  const ozaki_ocl_handle_t* h = (const ozaki_ocl_handle_t*)handle;
  int result = 0;
  if (NULL != h && NULL != h->ctx.kern_zgemm_block_construct_a && NULL != h->ctx.kern_zgemm_block_construct_b_n &&
      NULL != h->ctx.kern_zgemm_block_construct_b_t && NULL != h->ctx.kern_zgemm_block_finalize)
  {
    result = 1;
  }
  return result;
}


void ozaki_ocl_invalidate_cache(void* handle, const void* a, const void* b)
{
  ozaki_ocl_handle_t* h = (ozaki_ocl_handle_t*)handle;
  if (NULL != h) {
    ozaki_invalidate_cache(&h->ctx, a, b);
  }
}


void ozaki_ocl_finalize(void)
{
  libxstream_finalize();
}
