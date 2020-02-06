/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxs_blocked_gemm_types.h"
#include <libxs.h>


LIBXS_API libxs_blocked_gemm_handle* libxs_blocked_gemm_handle_create(/*unsigned*/ int nthreads,
  libxs_gemm_precision iprec, libxs_gemm_precision oprec, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* bm, const libxs_blasint* bn, const libxs_blasint* bk,
  const libxs_blasint* b_m1, const libxs_blasint* b_n1, const libxs_blasint* b_k1, const libxs_blasint* b_k2,
  const void* alpha, const void* beta, const int* gemm_flags,
  const libxs_gemm_prefetch_type* prefetch,
  const libxs_blocked_gemm_order* order)
{
  const char *const env_m = getenv("LIBXS_BLOCKED_GEMM_M"), *const env_n = getenv("LIBXS_BLOCKED_GEMM_N"), *const env_k = getenv("LIBXS_BLOCKED_GEMM_K");
  const libxs_blasint mm = LIBXS_MIN(0 == bm ? ((NULL == env_m || 0 == *env_m) ? 32 : atoi(env_m)) : *bm, m);
  const libxs_blasint kk = LIBXS_MIN(0 == bk ? ((NULL == env_k || 0 == *env_k) ? mm : atoi(env_k)) : *bk, k);
  const libxs_blasint nn = LIBXS_MIN(0 == bn ? ((NULL == env_n || 0 == *env_n) ? kk : atoi(env_n)) : *bn, n);
  libxs_blocked_gemm_handle* result = 0;
  static int error_once = 0;

  if (0 < m && 0 < n && 0 < k && 0 < mm && 0 < nn && 0 < kk && 0 < nthreads) {
    libxs_blocked_gemm_handle handle;
    memset(&handle, 0, sizeof(handle));
    if (0 == (m % mm) && 0 == (n % nn) && 0 == (k % kk) &&
        0 == (m % *b_m1) && 0 == (n % *b_n1) && 0 == (k % *b_k1) &&
        0 == ((k / *b_k1 / *b_k2) % kk) && 0 == ((n / *b_n1) % nn) && 0 == ((m / *b_m1) % mm))
    { /* check for valid block-size */
      libxs_gemm_descriptor* desc;
      libxs_descriptor_blob blob;
      if (0 == prefetch) { /* auto-prefetch */
        /* TODO: more sophisticated strategy perhaps according to CPUID */
        const libxs_gemm_prefetch_type prefetch_default = LIBXS_GEMM_PREFETCH_AL2BL2_VIA_C;
        const char *const env_p = getenv("LIBXS_BLOCKED_GEMM_PREFETCH");
        desc = libxs_gemm_descriptor_init2(&blob, iprec, oprec, mm, nn, kk, mm/*lda*/, kk/*ldb*/, mm/*ldc*/,
          alpha, beta, 0 == gemm_flags ? LIBXS_GEMM_FLAG_NONE : *gemm_flags,
          (NULL == env_p || 0 == *env_p) ? prefetch_default : libxs_gemm_uid2prefetch(atoi(env_p)));
      }
      else { /* user-defined */
        desc = libxs_gemm_descriptor_init2(&blob, iprec, oprec, mm, nn, kk, mm/*lda*/, kk/*ldb*/, mm/*ldc*/,
          alpha, beta, 0 == gemm_flags ? LIBXS_GEMM_FLAG_NONE : *gemm_flags, *prefetch);
      }
      if (0 != desc) {
        handle.mb = m / mm; handle.nb = n / nn; handle.kb = k / kk;
        if (LIBXS_GEMM_PREFETCH_NONE != desc->prefetch) {
          handle.kernel_pf = libxs_xmmdispatch(desc);
          desc->prefetch = LIBXS_GEMM_PREFETCH_NONE;
          handle.kernel = libxs_xmmdispatch(desc);
        }
        else { /* no prefetch */
          handle.kernel = libxs_xmmdispatch(desc);
          handle.kernel_pf.xmm = 0;
        }
      }
      if (0 != handle.kernel.xmm) {
        const size_t tls_size = LIBXS_UP2((size_t)mm * nn * LIBXS_TYPESIZE(oprec), LIBXS_CACHELINE) * nthreads;
        const size_t size_locks = (size_t)handle.mb * (size_t)handle.nb * sizeof(libxs_blocked_gemm_lock);
        handle.locks = (libxs_blocked_gemm_lock*)libxs_aligned_malloc(size_locks, LIBXS_CACHELINE);
        handle.buffer = libxs_aligned_malloc(tls_size, LIBXS_CACHELINE);
        result = (libxs_blocked_gemm_handle*)malloc(sizeof(libxs_blocked_gemm_handle));

        if (224 <= nthreads
#if !defined(__MIC__)
          && LIBXS_X86_AVX512_MIC <= libxs_target_archid
          && LIBXS_X86_AVX512_CORE > libxs_target_archid
#endif
          )
        {
          handle.barrier = libxs_barrier_create(nthreads / 4, 4);
        }
        else {
          handle.barrier = libxs_barrier_create(nthreads / 2, 2);
        }
        if (0 != result && 0 != handle.barrier && 0 != handle.buffer && 0 != handle.locks) {
          handle.m = m; handle.n = n; handle.k = k; handle.bm = mm; handle.bn = nn; handle.bk = kk;
          handle.b_m1 = *b_m1; handle.b_n1 = *b_n1; handle.b_k1 = *b_k1; handle.b_k2 = *b_k2;
          handle.iprec = iprec; handle.oprec = oprec;
          memset(handle.locks, 0, size_locks);
          handle.order = (0 == order ? LIBXS_BLOCKED_GEMM_ORDER_JIK : *order);
          handle.nthreads = nthreads;
          *result = handle;
        }
        else {
          if (0 != libxs_verbosity /* library code is expected to be mute */
            && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXS ERROR: BGEMM handle allocation failed!\n");
          }
          libxs_barrier_release(handle.barrier);
          libxs_free(handle.buffer);
          libxs_free(handle.locks);
          free(result);
          result = 0;
        }
      }
      else if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: unsupported BGEMM kernel requested!\n");
      }
    }
    else if (0 != libxs_verbosity /* library code is expected to be mute */
      && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXS ERROR: BGEMM block-size is invalid!\n");
    }
  }
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_blocked_gemm_handle_create!\n");
  }

  return result;
}


LIBXS_API void libxs_blocked_gemm_handle_destroy(const libxs_blocked_gemm_handle* handle)
{
  if (0 != handle) {
    libxs_barrier_release(handle->barrier);
    libxs_free(handle->buffer);
    libxs_free(handle->locks);
    free((libxs_blocked_gemm_handle*)handle);
  }
}


LIBXS_API int libxs_blocked_gemm_copyin_a(const libxs_blocked_gemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxs_blasint ild = (0 == ld ? handle->m : *ld);
    assert(ild >= handle->m);
#else
    LIBXS_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxs_blocked_gemm_copyin_a.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxs_blocked_gemm_copyin_a.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxs_blocked_gemm_copyin_a.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
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


LIBXS_API int libxs_blocked_gemm_copyin_b(const libxs_blocked_gemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxs_blasint ild = (0 == ld ? handle->k : *ld);
    assert(ild >= handle->k);
#else
    LIBXS_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxs_blocked_gemm_copyin_b.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxs_blocked_gemm_copyin_b.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxs_blocked_gemm_copyin_b.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
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


LIBXS_API int libxs_blocked_gemm_copyin_c(const libxs_blocked_gemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxs_blasint ild = (0 == ld ? handle->m : *ld);
    assert(ild >= handle->m);
#else
    LIBXS_UNUSED(ld);
#endif
    switch (handle->oprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxs_blocked_gemm_copyin_c.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxs_blocked_gemm_copyin_c.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE int
#       include "template/libxs_blocked_gemm_copyin_c.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
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


LIBXS_API int libxs_blocked_gemm_copyout_c(const libxs_blocked_gemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxs_blasint ild = (0 == ld ? handle->m : *ld);
    assert(ild >= handle->m);
#else
    LIBXS_UNUSED(ld);
#endif
    switch (handle->oprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxs_blocked_gemm_copyout_c.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxs_blocked_gemm_copyout_c.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE int
#       include "template/libxs_blocked_gemm_copyout_c.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
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


LIBXS_API int libxs_blocked_gemm_convert_b_to_a(const libxs_blocked_gemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxs_blasint ild = (0 == ld ? handle->k : *ld);
    assert(ild >= handle->k);
#else
    LIBXS_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxs_blocked_gemm_convert_b_to_a.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxs_blocked_gemm_convert_b_to_a.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxs_blocked_gemm_convert_b_to_a.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
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


LIBXS_API int libxs_blocked_gemm_transpose_b(const libxs_blocked_gemm_handle* handle, const void* src, const libxs_blasint* ld, void* dst)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  if (0 != handle) {
#if 0 /* TODO: support leading dimension for the source buffer */
    const libxs_blasint ild = (0 == ld ? handle->k : *ld);
    assert(ild >= handle->k);
#else
    LIBXS_UNUSED(ld);
#endif
    switch (handle->iprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE double
#       include "template/libxs_blocked_gemm_transpose_b.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE float
#       include "template/libxs_blocked_gemm_transpose_b.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE short
#       include "template/libxs_blocked_gemm_transpose_b.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE
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


LIBXS_API_INLINE void internal_bgemm_order(libxs_blocked_gemm_order order,
  libxs_blasint w_i, libxs_blasint nw_i, libxs_blasint nw_j, libxs_blasint nw_k,
  libxs_blasint* i2, libxs_blasint* j2, libxs_blasint* k2)
{
  switch (order) {
    case LIBXS_BLOCKED_GEMM_ORDER_JIK: {
      *j2 = (w_i / (nw_i * nw_k));
      *i2 = (w_i - (*j2) * (nw_i * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXS_BLOCKED_GEMM_ORDER_IJK: {
      *i2 = (w_i / (nw_j * nw_k));
      *j2 = (w_i - (*i2) * (nw_j * nw_k)) / nw_k;
      *k2 = (w_i % nw_k);
    } break;
    case LIBXS_BLOCKED_GEMM_ORDER_JKI: {
      *j2 = (w_i / (nw_k * nw_i));
      *k2 = (w_i - (*j2) * (nw_k * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXS_BLOCKED_GEMM_ORDER_IKJ: {
      *i2 = (w_i / (nw_k * nw_j));
      *k2 = (w_i - (*i2) * (nw_k * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    case LIBXS_BLOCKED_GEMM_ORDER_KJI: {
      *k2 = (w_i / (nw_j * nw_i));
      *j2 = (w_i - (*k2) * (nw_j * nw_i)) / nw_i;
      *i2 = (w_i % nw_i);
    } break;
    case LIBXS_BLOCKED_GEMM_ORDER_KIJ: {
      *k2 = (w_i / (nw_i * nw_j));
      *i2 = (w_i - (*k2) * (nw_i * nw_j)) / nw_j;
      *j2 = (w_i % nw_j);
    } break;
    default: assert(0/*should never happen*/);
  }
}

LIBXS_API void libxs_blocked_gemm_st(const libxs_blocked_gemm_handle* handle, const void* a, const void* b, void* c,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  static int error_once = 0;
#if defined(LIBXS_BLOCKED_GEMM_CHECKS)
  if (0 != handle && 0 != a && 0 != b && 0 != c && start_thread <= tid && 0 <= tid)
#endif
  {
    const int ltid = tid - start_thread;
    if (handle->nthreads > 1) {
      libxs_barrier_init(handle->barrier, ltid);
    }
    switch (handle->iprec) {
      case LIBXS_GEMM_PRECISION_F64: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_AB double
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_C  double
#       include "template/libxs_blocked_gemm.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_AB
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_C
      } break;
      case LIBXS_GEMM_PRECISION_F32: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_AB float
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_C  float
#       include "template/libxs_blocked_gemm.tpl.c"
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_AB
#       undef  LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_C
      } break;
      case LIBXS_GEMM_PRECISION_I16: {
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_AB short
#       define LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_C  int
#       include "template/libxs_blocked_gemm.tpl.c"
#       undef LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_C
#       undef LIBXS_BLOCKED_GEMM_TEMPLATE_TYPE_AB
      } break;
      default: if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: BGEMM precision is not supported!\n");
      }
    }
    if (handle->nthreads > 1) {
      libxs_barrier_wait(handle->barrier, ltid);
    }
  }
#if defined(LIBXS_BLOCKED_GEMM_CHECKS)
  else if (0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: invalid arguments for libxs_blocked_gemm!\n");
  }
#endif
}

