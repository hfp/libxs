/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxs_bgemm.h>
#include <libxs.h>
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXS_BGEMM_MAX_NTHREADS)
# define LIBXS_BGEMM_MAX_NTHREADS 512
#endif
#if !defined(LIBXS_BGEMM_PREFETCH)
# define LIBXS_BGEMM_PREFETCH
#endif


typedef union LIBXS_RETARGETABLE libxs_bgemm_lock {
  volatile int instance, pad[16];
} libxs_bgemm_lock;

struct LIBXS_RETARGETABLE libxs_bgemm_handle {
  union { double d; float s; int w; } alpha, beta;
#if defined(LIBXS_BGEMM_PREFETCH)
  libxs_xmmfunction kernel_pf;
#endif
  libxs_xmmfunction kernel;
  void* buffer;
  libxs_bgemm_lock* locks;
  libxs_blasint m, n, k, bm, bn, bk;
  libxs_blasint b_m1, b_n1, b_k1, b_k2;
  libxs_blasint mb, nb, kb;
  libxs_gemm_precision precision;
  libxs_bgemm_order order;
  int typesize;
  int flags;
};


LIBXS_API_DEFINITION libxs_bgemm_handle* libxs_bgemm_handle_create(libxs_gemm_precision precision,
  libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint bm, libxs_blasint bn, libxs_blasint bk,
  const void* alpha, const void* beta, const int* gemm_flags, const libxs_bgemm_order* order)
{
  libxs_bgemm_handle handle, *result = 0;
  libxs_gemm_descriptor descriptor = { 0 };
  static int error_once = 0;

  if (0 < m && 0 < n && 0 < k && 0 < bm && 0 < bn && 0 < bk) {
    memset(&handle, 0, sizeof(handle));
    handle.flags = (0 == gemm_flags ? LIBXS_FLAGS : *gemm_flags);

    switch (precision) {
      case LIBXS_GEMM_PRECISION_F64: {
        handle.alpha.d = (0 != alpha ? *((const double*)alpha) : LIBXS_ALPHA);
        handle.beta.d = (0 != beta ? *((const double*)beta) : LIBXS_BETA);
        assert(LIBXS_FEQ(1, handle.alpha.d) && LIBXS_FEQ(1, handle.beta.d)/*TODO*/);
        LIBXS_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
          handle.alpha.d, handle.beta.d, LIBXS_PREFETCH_NONE);
        handle.typesize = 8;
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
        handle.alpha.s = (0 != alpha ? *((const float*)alpha) : LIBXS_ALPHA);
        handle.beta.s = (0 != beta ? *((const float*)beta) : LIBXS_BETA);
        assert(LIBXS_FEQ(1, handle.alpha.s) && LIBXS_FEQ(1, handle.beta.s)/*TODO*/);
        LIBXS_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
          handle.alpha.s, handle.beta.s, LIBXS_PREFETCH_NONE);
        handle.typesize = 4;
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
        handle.alpha.w = (0 != alpha ? *((const int*)alpha) : LIBXS_ALPHA);
        handle.beta.w = (0 != beta ? *((const int*)beta) : LIBXS_BETA);
        assert(LIBXS_FEQ(1, handle.alpha.w) && LIBXS_FEQ(1, handle.beta.w)/*TODO*/);
        LIBXS_GEMM_DESCRIPTOR(descriptor, precision, handle.flags, bm, bn, bk, bm/*lda*/, bk/*ldb*/, bm/*ldc*/,
          handle.alpha.w, handle.beta.w, LIBXS_PREFETCH_NONE);
        handle.typesize = 2;
      } break;
      default: ;
    }

    if (0 < handle.typesize) {
      handle.mb = m / bm; handle.nb = n / bn; handle.kb = k / bk;

      if (0 == (m % bm) && 0 == (n % bn) && 0 == (k % bk)) { /* check for valid block-size */
        const libxs_blasint sm = m / handle.mb, sn = n / handle.nb, size = sm * sn;
        handle.b_m1 = 1; handle.b_n1 = 1; handle.b_k1 = 1; handle.b_k2 = 1;
        assert(0 == (m % handle.b_m1) && 0 == (n % handle.b_n1) && 0 == (k % handle.b_k1));
        assert(0 == ((k / handle.b_k1 / handle.b_k2) % bk));
        assert(0 == ((n / handle.b_n1) % bn));
        assert(0 == ((m / handle.b_m1) % bm));
        handle.kernel = libxs_xmmdispatch(&descriptor);
#if defined(LIBXS_BGEMM_PREFETCH)
        descriptor.prefetch = LIBXS_PREFETCH_AL2BL2_VIA_C;
        handle.kernel_pf = libxs_xmmdispatch(&descriptor);
#endif
        if (0 != handle.kernel.smm
#if defined(LIBXS_BGEMM_PREFETCH)
         && 0 != handle.kernel_pf.smm
#endif
        ) { /* TODO: allow NULL-kernels and implement a BLAS fallback */
          result = (libxs_bgemm_handle*)malloc(sizeof(libxs_bgemm_handle));
          handle.buffer = libxs_aligned_malloc(LIBXS_BGEMM_MAX_NTHREADS * bm * bn * handle.typesize, LIBXS_ALIGNMENT);
          handle.locks = (libxs_bgemm_lock*)libxs_aligned_malloc(size * sizeof(libxs_bgemm_lock), LIBXS_ALIGNMENT);

          if (0 != result && 0 != handle.buffer && 0 != handle.locks) {
            handle.precision = precision;
            handle.m = m; handle.n = n; handle.k = k; handle.bm = bm; handle.bn = bn; handle.bk = bk;
            memset(handle.locks, 0, size * sizeof(libxs_bgemm_lock));
            handle.order = (0 == order ? LIBXS_BGEMM_ORDER_JIK : *order);
            *result = handle;
          }
          else {
            libxs_free(handle.buffer);
            libxs_free(handle.locks);
            free(result);
            result = 0;
          }
        }
        else if (0 != libxs_verbosity /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS: BGEMM kernel failed to generate!\n");
        }
      }
      else {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS: BGEMM block-size is invalid!\n");
        }
      }
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS: BGEMM precision is not supported!\n");
    }
  }
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS: BGEMM arguments for libxs_bgemm_handle_create are invalid!\n");
  }

  return result;
}


LIBXS_API_DEFINITION void libxs_bgemm_handle_destroy(const libxs_bgemm_handle* handle)
{
  if (0 != handle) {
    libxs_free(handle->buffer);
    libxs_free(handle->locks);
    free((libxs_bgemm_handle*)handle);
  }
}


LIBXS_API_DEFINITION int libxs_bgemm_copyin_a(const libxs_bgemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxs_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXS_UNUSED(ild);

    switch (handle->precision) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxs_bgemm_copyin_a.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxs_bgemm_copyin_a.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE short
#       include "template/libxs_bgemm_copyin_a.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      default: {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_DEFINITION int libxs_bgemm_copyin_b(const libxs_bgemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxs_blasint ild = (0 == ld ? handle->k : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->k); LIBXS_UNUSED(ild);

    switch (handle->precision) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxs_bgemm_copyin_b.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxs_bgemm_copyin_b.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE short
#       include "template/libxs_bgemm_copyin_b.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      default: {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS: BGEMM precision of matrix B is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_DEFINITION int libxs_bgemm_copyin_c(const libxs_bgemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
    const libxs_blasint ild = (0 == ld ? handle->m : *ld);
    /* TODO: support leading dimension for the source buffer */
    assert(ild >= handle->m); LIBXS_UNUSED(ild);

    switch (handle->precision) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE double
#       include "template/libxs_bgemm_copyin_c.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxs_bgemm_copyin_c.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE int
#       include "template/libxs_bgemm_copyin_c.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      default: {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_INLINE void internal_bgemm_order(libxs_bgemm_order order,
  libxs_blasint w_i, libxs_blasint nw_i, libxs_blasint nw_j, libxs_blasint nw_k,
  libxs_blasint* i2, libxs_blasint* j2, libxs_blasint* k2)
{
  switch (order) {
    case LIBXS_BGEMM_ORDER_JIK: {
      *j2 = (w_i / (nw_i * nw_k));
      *i2 = (w_i - (*j2) * (nw_i * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXS_BGEMM_ORDER_IJK: {
      *i2 = (w_i / (nw_j * nw_k));
      *j2 = (w_i - (*i2) * (nw_j * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXS_BGEMM_ORDER_JKI: {
      *j2 = (w_i / (nw_k * nw_i));
      *k2 = (w_i - (*j2) * (nw_k * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXS_BGEMM_ORDER_IKJ: {
      *i2 = (w_i / (nw_k * nw_j));
      *k2 = (w_i - (*i2) * (nw_k * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    case LIBXS_BGEMM_ORDER_KJI: {
      *k2 = (w_i / (nw_j * nw_i));
      *j2 = (w_i - (*k2) * (nw_j * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXS_BGEMM_ORDER_KIJ: {
      *k2 = (w_i / (nw_i * nw_j));
      *i2 = (w_i - (*k2) * (nw_i * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    default: assert(0/*should never happen*/);
  }
}


LIBXS_API_DEFINITION void libxs_bgemm(const libxs_bgemm_handle* handle,
  const void* a, const void* b, void* c, int tid, int nthreads)
{
  static int error_once = 0;
#if !defined(NDEBUG) /* intentionally no errror check in release build */
  if (0 != handle && 0 != a && 0 != b && 0 != c && 0 <= tid && tid < nthreads)
#endif
  {
    switch (handle->precision) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE_AB double
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE_C  double
#       include "template/libxs_bgemm.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE_AB
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE_C
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE_AB float
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE_C  float
#       include "template/libxs_bgemm.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE_AB
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE_C
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE_AB short
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE_C  int
#       include "template/libxs_bgemm.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE_AB
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE_C
      } break;
      default: if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS: BGEMM precision is not supported!\n");
      }
    }
  }
#if !defined(NDEBUG) /* intentionally no errror check in release build */
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS: BGEMM arguments for libxs_bgemm are invalid!\n");
  }
#endif
}

