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
  ozaki_context_t ctx;
  libxstream_stream_t* stream;
} ozaki_ocl_handle_t;


void* ozaki_ocl_create(int use_double, int kind, int verbosity,
  int nslices, int batch_k, int ozflags, int oztrim,
  ozaki_host_preprocess_fn host_a, ozaki_host_preprocess_fn host_b,
  int host_bm, int host_bn, int host_bk)
{
  ozaki_ocl_handle_t* h = NULL;
  int ndevices = 0;
  if (EXIT_SUCCESS == libxstream_init()
      && EXIT_SUCCESS == libxstream_device_count(&ndevices)
      && 0 < ndevices
      && EXIT_SUCCESS == libxstream_device_set_active(0))
  {
    h = (ozaki_ocl_handle_t*)calloc(1, sizeof(*h));
  }
  if (NULL != h) {
    if (EXIT_SUCCESS != ozaki_init(&h->ctx, 0, 0, 0,
          use_double, kind, verbosity, nslices, batch_k,
          ozflags, oztrim))
    {
      free(h); h = NULL;
    }
    /* Refuse the handle if fp64 was requested but the device only supports fp32.
     * Silently downgrading would cause a type mismatch: the host passes double
     * arrays while the OpenCL kernels operate on float, leading to wrong results
     * and potential memory corruption. */
    else if (use_double && !h->ctx.use_double) {
      ozaki_destroy(&h->ctx);
      free(h); h = NULL;
    }
    else if (EXIT_SUCCESS != libxstream_stream_create(
               &h->stream, "ozaki_wrap",
               NULL != h->ctx.hist ? LIBXSTREAM_STREAM_PROFILING : LIBXSTREAM_STREAM_DEFAULT))
    {
      ozaki_destroy(&h->ctx);
      free(h); h = NULL;
    }
    else {
      /* Set host preprocessing callbacks if block sizes match.
       * The GPU side auto-selects block sizes; only wire up the host
       * callbacks when the CPU-side BLOCK_M/N/K are compatible. */
      if (NULL != host_a && h->ctx.bm == host_bm && h->ctx.bk == host_bk) {
        h->ctx.host_preprocess_a = host_a;
      }
      if (NULL != host_b && h->ctx.bn == host_bn && h->ctx.bk == host_bk) {
        h->ctx.host_preprocess_b = host_b;
      }
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


int ozaki_ocl_gemm(void* handle, char transa, char transb,
  int M, int N, int K, double alpha, const void* a, int lda,
  const void* b, int ldb, double beta, void* c, int ldc)
{
  int result = EXIT_FAILURE;
  ozaki_ocl_handle_t* h = (ozaki_ocl_handle_t*)handle;
  if (NULL != h) {
    result = ozaki_gemm(&h->ctx, h->stream,
      transa, transb, M, N, K,
      alpha, a, lda, b, ldb, beta, c, ldc);
    /* BLAS API is synchronous: caller expects result in c upon return. */
    libxstream_stream_sync(h->stream);
  }
  return result;
}


void ozaki_ocl_finalize(void) {
  libxstream_finalize();
}
