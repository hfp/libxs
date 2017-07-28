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
#include "libxs_gemm.h"
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


typedef union LIBXS_RETARGETABLE libxs_bgemm_lock {
  volatile int instance, pad[16];
} libxs_bgemm_lock;

struct LIBXS_RETARGETABLE libxs_bgemm_handle {
  union { double d; float s; int w; } alpha, beta;
  libxs_xmmfunction kernel_pf;
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


LIBXS_API_DEFINITION libxs_bgemm_handle* libxs_bgemm_handle_create(
  libxs_gemm_precision precision, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* bm, const libxs_blasint* bn, const libxs_blasint* bk,
  const libxs_blasint* b_m1, const libxs_blasint* b_n1, const libxs_blasint* b_k1, const libxs_blasint* b_k2,
  const void* alpha, const void* beta, const int* gemm_flags,
  const libxs_gemm_prefetch_type* strategy,
  const libxs_bgemm_order* order)
{
  const char *const env_m = getenv("LIBXS_BGEMM_M"), *const env_n = getenv("LIBXS_BGEMM_N"), *const env_k = getenv("LIBXS_BGEMM_K");
  const libxs_blasint mm = LIBXS_MIN(0 == bm ? ((0 == env_m || 0 == *env_m) ? 32 : atoi(env_m)) : *bm, m);
  const libxs_blasint kk = LIBXS_MIN(0 == bk ? ((0 == env_k || 0 == *env_k) ? mm : atoi(env_k)) : *bk, k);
  const libxs_blasint nn = LIBXS_MIN(0 == bn ? ((0 == env_n || 0 == *env_n) ? kk : atoi(env_n)) : *bn, n);
  libxs_bgemm_handle handle, *result = 0;
  libxs_gemm_descriptor descriptor = { 0 };
  static int error_once = 0;

  if (0 < m && 0 < n && 0 < k && 0 < mm && 0 < nn && 0 < kk) {
    memset(&handle, 0, sizeof(handle));

    if (EXIT_SUCCESS == libxs_gemm_descriptor_init(&descriptor,
      precision, mm, nn, kk, &mm/*lda*/, &kk/*ldb*/, &mm/*ldc*/,
      alpha, beta, gemm_flags, 0/*prefetch*/))
    {
      handle.typesize = LIBXS_TYPESIZE(precision);
      handle.mb = m / mm; handle.nb = n / nn; handle.kb = k / kk;
      assert(0 < handle.typesize);

      if (0 == (m % mm) && 0 == (n % nn) && 0 == (k % kk) &&
          0 == (m % *b_m1) && 0 == (n % *b_n1) && 0 == (k % *b_k1) &&
          0 == ((k / *b_k1 / *b_k2) % kk) && 0 == ((n / *b_n1) % nn) && 0 == ((m / *b_m1) % mm)) { /* check for valid block-size */
        const libxs_gemm_prefetch_type prefetch = (0 == strategy ? ((libxs_gemm_prefetch_type)LIBXS_PREFETCH) : *strategy);
        handle.b_m1 = *b_m1; handle.b_n1 = *b_n1;
        handle.b_k1 = *b_k1; handle.b_k2 = *b_k2;
        handle.kernel = libxs_xmmdispatch(&descriptor);
        if (0 != handle.kernel.smm && LIBXS_PREFETCH_NONE != prefetch && LIBXS_PREFETCH_SIGONLY != prefetch) {
          if (LIBXS_PREFETCH_AUTO == prefetch) { /* automatically chosen */
            /* TODO: more sophisticated strategy perhaps according to CPUID */
            const char *const env_p = getenv("LIBXS_BGEMM_PREFETCH");
            const int uid = ((0 == env_p || 0 == *env_p) ? 7/*LIBXS_PREFETCH_AL2BL2_VIA_C*/ : atoi(env_p));
            descriptor.prefetch = (unsigned short)libxs_gemm_uid2prefetch(uid);
          }
          else { /* user-defined */
            descriptor.prefetch = (unsigned short)prefetch;
          }
          handle.kernel_pf = libxs_xmmdispatch(&descriptor);
        }
        if (0 != handle.kernel.smm && (LIBXS_PREFETCH_NONE == descriptor.prefetch || 0 != handle.kernel_pf.smm)) {
          const size_t tls_size = ((mm * nn * handle.typesize + LIBXS_CACHELINE_SIZE - 1) & ~(LIBXS_CACHELINE_SIZE - 1)) * LIBXS_BGEMM_MAX_NTHREADS;
          const libxs_blasint size_locks = handle.mb * handle.nb * sizeof(libxs_bgemm_lock);
          handle.locks = (libxs_bgemm_lock*)libxs_aligned_malloc(size_locks, LIBXS_ALIGNMENT);
          handle.buffer = libxs_aligned_malloc(tls_size, LIBXS_ALIGNMENT);
          result = (libxs_bgemm_handle*)malloc(sizeof(libxs_bgemm_handle));

          if (0 != result && 0 != handle.buffer && 0 != handle.locks) {
            handle.precision = precision;
            handle.m = m; handle.n = n; handle.k = k; handle.bm = mm; handle.bn = nn; handle.bk = kk;
            memset(handle.locks, 0, size_locks);
            handle.order = (0 == order ? LIBXS_BGEMM_ORDER_JIK : *order);
            *result = handle;
          }
          else {
            if (0 != libxs_verbosity /* library code is expected to be mute */
             && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXS ERROR: BGEMM handle allocation failed!\n");
            }
            libxs_free(handle.buffer);
            libxs_free(handle.locks);
            free(result);
            result = 0;
          }
        }
        else if (0 != libxs_verbosity /* library code is expected to be mute */
              && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: BGEMM kernel generation failed!\n");
        }
      }
      else {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: BGEMM block-size is invalid!\n");
        }
      }
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: BGEMM precision is not supported!\n");
    }
  }
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_bgemm_handle_create!\n");
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
          fprintf(stderr, "LIBXS ERROR: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: BGEMM-handle cannot be NULL!\n");
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
          fprintf(stderr, "LIBXS ERROR: BGEMM precision of matrix B is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: BGEMM-handle cannot be NULL!\n");
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
          fprintf(stderr, "LIBXS ERROR: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: BGEMM-handle cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API_DEFINITION int libxs_bgemm_copyout_c(const libxs_bgemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
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
#       include "template/libxs_bgemm_copyout_c.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE float
#       include "template/libxs_bgemm_copyout_c.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BGEMM_TEMPLATE_REAL_TYPE int
#       include "template/libxs_bgemm_copyout_c.tpl.c"
#       undef  LIBXS_BGEMM_TEMPLATE_REAL_TYPE
      } break;
      default: {
        if (0 != libxs_verbosity /* library code is expected to be mute */
         && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: BGEMM precision of matrix A is not supported!\n");
        }
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    if (0 != libxs_verbosity /* library code is expected to be mute */
     && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: BGEMM-handle cannot be NULL!\n");
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
        fprintf(stderr, "LIBXS ERROR: BGEMM precision is not supported!\n");
      }
    }
  }
#if !defined(NDEBUG) /* intentionally no errror check in release build */
  else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_bgemm!\n");
  }
#endif
}
