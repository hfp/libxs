/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

/**
 * Bridge between LIBXS Ozaki (CPU) and LIBXSTREAM Ozaki (GPU).
 * Compiled once (precision-independent); wraps LIBXSTREAM types
 * behind an opaque void* handle so that ozaki.c needs no OpenCL
 * or LIBXSTREAM headers.
 */
#include "ozaki_opencl.h"


typedef struct ozaki_gpu_handle_t {
  ozaki_context_t ctx;
  libxstream_stream_t* stream;
} ozaki_gpu_handle_t;


void* ozaki_gpu_create(int use_double, int kind, int verbosity,
  int nslices, int batch_k, int ozflags, int oztrim)
{
  ozaki_gpu_handle_t* h;
  int ndevices = 0;
  if (EXIT_SUCCESS != libxstream_init()
      || EXIT_SUCCESS != libxstream_get_ndevices(&ndevices)
      || 0 >= ndevices
      || EXIT_SUCCESS != libxstream_set_active_device(0))
  {
    return NULL;
  }
  h = (ozaki_gpu_handle_t*)calloc(1, sizeof(*h));
  if (NULL == h) return NULL;
  if (EXIT_SUCCESS != ozaki_init(&h->ctx, 0, 0, 0,
        use_double, kind, verbosity, nslices, batch_k,
        ozflags, oztrim))
  {
    free(h);
    return NULL;
  }
  /* Refuse the handle if fp64 was requested but the device only supports fp32.
   * Silently downgrading would cause a type mismatch: the host passes double
   * arrays while the GPU kernels operate on float, leading to wrong results
   * and potential memory corruption. */
  if (use_double && !h->ctx.use_double) {
    ozaki_destroy(&h->ctx);
    free(h);
    return NULL;
  }
  if (EXIT_SUCCESS != libxstream_stream_create(
        &h->stream, "ozaki_wrap", -1))
  {
    ozaki_destroy(&h->ctx);
    free(h);
    return NULL;
  }
  return h;
}


void ozaki_gpu_release(void* handle)
{
  ozaki_gpu_handle_t* h = (ozaki_gpu_handle_t*)handle;
  if (NULL != h) {
    ozaki_destroy(&h->ctx);
    if (NULL != h->stream) libxstream_stream_destroy(h->stream);
    free(h);
  }
}


int ozaki_gpu_dgemm(void* handle, char transa, char transb,
  int M, int N, int K, double alpha, const void* a, int lda,
  const void* b, int ldb, double beta, void* c, int ldc)
{
  ozaki_gpu_handle_t* h = (ozaki_gpu_handle_t*)handle;
  if (NULL == h) return EXIT_FAILURE;
  return ozaki_gemm(&h->ctx, h->stream,
    transa, transb, M, N, K,
    alpha, a, lda, b, ldb, beta, c, ldc);
}


void ozaki_gpu_finalize(void) {
  libxstream_finalize();
}
