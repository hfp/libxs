/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_GEMM_H
#define LIBXS_GEMM_H

#include "libxs.h"
#include "libxs_reg.h"

#if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) \
  && defined(LIBXS_PLATFORM_X86)
# include <mkl.h>
#endif

#if defined(__LIBXSMM)
# include <libxsmm.h>
#endif


/** Standard Fortran BLAS dgemm signature (e.g., dgemm_). */
typedef void (*libxs_gemm_dblas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* a, const int* lda,
                       const double* b, const int* ldb,
  const double* beta,        double* c, const int* ldc);

/** Standard Fortran BLAS sgemm signature (e.g., sgemm_). */
typedef void (*libxs_gemm_sblas_t)(
  const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* a, const int* lda,
                      const float* b, const int* ldb,
  const float* beta,        float* c, const int* ldc);

/**
 * MKL JIT dgemm kernel signature (mkl_jit_get_dgemm_ptr).
 * Shape, alpha, beta, and transpose info are baked into the jitter handle.
 */
typedef void (*libxs_gemm_djit_t)(void* jitter,
  const double* a, const double* b, double* c);

/**
 * MKL JIT sgemm kernel signature (mkl_jit_get_sgemm_ptr).
 * Shape, alpha, beta, and transpose info are baked into the jitter handle.
 */
typedef void (*libxs_gemm_sjit_t)(void* jitter,
  const float* a, const float* b, float* c);

/**
 * XGEMM parameter struct, layout-compatible with libxsmm_gemm_param.
 * op: 4 void pointers (op state), a/b/c: 6 void pointers each (matrix arg).
 * Only a.primary, b.primary, c.primary are used for plain GEMM.
 */
typedef struct libxs_gemm_param_t {
  void* op[4]; const void* a[6]; const void* b[6]; void* c[6];
} libxs_gemm_param_t;

/** Opaque XGEMM kernel: void(const libxs_gemm_param_t*). */
typedef void (*libxs_gemm_xfn_t)(const void*);

/**
 * MKL-compatible jit_create_dgemm signature.
 * Return 0 on success (same as MKL_JIT_SUCCESS).
 */
typedef int (*libxs_jit_create_dgemm_t)(void** jitter,
  int layout, int transa, int transb, int m, int n, int k,
  double alpha, int lda, int ldb, double beta, int ldc);

/** MKL-compatible jit_get_dgemm_ptr signature. */
typedef void* (*libxs_jit_get_dgemm_t)(void* jitter);

/**
 * MKL-compatible jit_create_sgemm signature.
 * Return 0 on success (same as MKL_JIT_SUCCESS).
 */
typedef int (*libxs_jit_create_sgemm_t)(void** jitter,
  int layout, int transa, int transb, int m, int n, int k,
  float alpha, int lda, int ldb, float beta, int ldc);

/** MKL-compatible jit_get_sgemm_ptr signature. */
typedef void* (*libxs_jit_get_sgemm_t)(void* jitter);

/**
 * LIBXSMM-compatible dispatch (scalar args, no struct dependency).
 * flags: bit 0 = transpose A, bit 1 = transpose B, bit 2 = beta==0.
 * Returns kernel function pointer, or NULL on failure.
 */
typedef libxs_gemm_xfn_t (*libxs_xgemm_dispatch_t)(
  int datatype, int flags, int m, int n, int k,
  int lda, int ldb, int ldc);

/** Backend function pointers for GEMM dispatch. */
typedef struct libxs_gemm_backend_t {
  libxs_jit_create_dgemm_t jit_create_dgemm;
  libxs_jit_get_dgemm_t   jit_get_dgemm;
  libxs_jit_create_sgemm_t jit_create_sgemm;
  libxs_jit_get_sgemm_t   jit_get_sgemm;
  libxs_xgemm_dispatch_t  xgemm_dispatch;
  libxs_gemm_dblas_t dgemm_blas;
  libxs_gemm_sblas_t sgemm_blas;
} libxs_gemm_backend_t;

/** Flags controlling GEMM batch synchronization (bitfield). */
typedef enum libxs_gemm_flags_t {
  LIBXS_GEMM_FLAGS_DEFAULT = 0,
  LIBXS_GEMM_FLAG_NOLOCK = 1
} libxs_gemm_flags_t;

/**
 * GEMM shape: problem geometry, transpose flags, and scalar
 * coefficients.  Alpha/beta are stored as double regardless of
 * datatype (float values are promoted without loss).
 * Also serves as registry key when caching dispatched
 * configurations (all-int fields avoid padding).
 */
typedef struct libxs_gemm_shape_t {
  libxs_data_t datatype;
  char transa, transb;
  int m, n, k, lda, ldb, ldc;
  double alpha, beta;
} libxs_gemm_shape_t;

/**
 * Configuration supplying GEMM kernels. Pass NULL to batch functions
 * to use the built-in default kernel (auto-vectorized, no BLAS dependency).
 * Users can supply their own kernels. Kernel selection priority:
 *   1. JIT kernel (dgemm_jit/sgemm_jit + jitter) if non-NULL,
 *   2. XGEMM kernel (xgemm) if non-NULL,
 *   3. BLAS kernel (dgemm/sgemm) if non-NULL,
 *   4. built-in default kernel.
 * Only the function pointers matching the datatype need to be set.
 * By default (flags=0), _task variants synchronize C-matrix updates.
 * Set LIBXS_GEMM_FLAG_NOLOCK if no duplicate C pointers exist.
 * The shape member is populated by libxs_gemm_dispatch.
 */
typedef struct libxs_gemm_config_t {
  libxs_gemm_dblas_t dgemm_blas;
  libxs_gemm_sblas_t sgemm_blas;
  libxs_gemm_djit_t dgemm_jit;
  libxs_gemm_sjit_t sgemm_jit;
  libxs_gemm_xfn_t xgemm;
  void* jitter;
  libxs_gemm_flags_t flags;
  libxs_gemm_shape_t shape;
} libxs_gemm_config_t;

/**
 * Runtime GEMM dispatch with double-dispatch support.
 * shape: full problem shape (used as registry key and stored in config).
 * kernel_shape: actual kernel dimensions to dispatch (may differ from
 *   shape, e.g., tight ldc for scratch). NULL means same as shape.
 *   If kernel_shape differs, the kernel is looked up/registered under
 *   its own key, avoiding redundant code generation.
 * backend: backend function pointers (MKL JIT, LIBXSMM, BLAS). NULL
 *   means no backends (built-in default only).
 * Returns pointer to cached config (registry-owned), NULL on failure.
 */
LIBXS_API libxs_gemm_config_t* libxs_gemm_dispatch_rt(
  const libxs_gemm_shape_t* shape,
  const libxs_gemm_shape_t* kernel_shape,
  const libxs_gemm_backend_t* backend,
  void* registry);

/**
 * Process a batch of GEMMs given arrays of pointers to matrices.
 * C_i := alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchsize).
 * Shape, alpha, beta, datatype, and transpose info come from config->shape.
 */
LIBXS_API void libxs_gemm_batch(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config);

/** Per-thread form of libxs_gemm_batch. */
LIBXS_API void libxs_gemm_batch_task(
  const void* a_array[], const void* b_array[], void* c_array[],
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);

/**
 * Process a batch of GEMMs given index arrays into contiguous buffers.
 * C_i := alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchsize).
 * Each stride_a[i], stride_b[i], stride_c[i] is an element-offset from
 * the respective base pointer (a, b, c). index_base selects the indexing
 * convention: 0 for zero-based (C), 1 for one-based (Fortran).
 * index_stride is the Byte-stride used to walk the stride arrays
 * (e.g. sizeof(int) for packed int arrays).
 * index_stride=0 selects constant-stride mode: each stride array
 * points to a single element, and batch i uses offset stride[0]*i
 * (equivalent to the strided batch pattern).
 * Shape, alpha, beta, datatype, and transpose info come from config->shape.
 */
LIBXS_API void libxs_gemm_index(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config);

/** Per-thread form of libxs_gemm_index. */
LIBXS_API void libxs_gemm_index_task(
  const void* a, const int stride_a[],
  const void* b, const int stride_b[],
        void* c, const int stride_c[],
  int index_stride, int index_base,
  int batchsize, const libxs_gemm_config_t* config,
  int tid, int ntasks);

/**
 * Check whether a dispatched config holds a usable kernel.
 * Returns nonzero if libxs_gemm_call would succeed, zero otherwise.
 */
LIBXS_API_INLINE int libxs_gemm_ready(const libxs_gemm_config_t* config) {
  return NULL != config
    && (  (NULL != config->dgemm_jit && NULL != config->jitter)
       || (NULL != config->sgemm_jit && NULL != config->jitter)
       ||  NULL != config->xgemm);
}

/**
 * Dispatch a GEMM kernel and return a pointer to the configuration.
 * On registry hit, returns pointer to cached config (zero-cost).
 * On miss, JIT-compiles (MKL JIT > LIBXSMM > fallthrough), stores
 * in registry, and returns pointer to stored config.
 * Returns NULL on failure (unsupported datatype, NULL registry).
 * The returned pointer is owned by the registry and valid until
 * the registry is destroyed.
 * If registry is NULL, uses the internal global registry.
 */
#if defined(LIBXSMM_H)
static LIBXS_INLINE libxs_gemm_xfn_t internal_libxs_xgemm_dispatch(
  int datatype, int flags, int m, int n, int k,
  int lda, int ldb, int ldc)
{
  libxsmm_gemm_shape gemm_shape;
  libxsmm_bitfield xflags = LIBXSMM_GEMM_FLAG_NONE;
  libxsmm_datatype xtype;
  libxsmm_gemmfunction xresult;
  if (0 != (flags & 1)) xflags |= LIBXSMM_GEMM_FLAG_TRANS_A;
  if (0 != (flags & 2)) xflags |= LIBXSMM_GEMM_FLAG_TRANS_B;
  if (0 != (flags & 4)) xflags |= LIBXSMM_GEMM_FLAG_BETA_0;
  xtype = (LIBXS_DATATYPE_F64 == datatype)
    ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
  gemm_shape.m = m; gemm_shape.n = n; gemm_shape.k = k;
  gemm_shape.lda = lda; gemm_shape.ldb = ldb; gemm_shape.ldc = ldc;
  gemm_shape.a_in_type = xtype; gemm_shape.b_in_type = xtype;
  gemm_shape.comp_type = xtype; gemm_shape.out_type = xtype;
  xresult = libxsmm_dispatch_gemm(gemm_shape, xflags,
    LIBXSMM_GEMM_PREFETCH_NONE);
  return (libxs_gemm_xfn_t)xresult;
}
#endif
LIBXS_API_INLINE libxs_gemm_config_t* libxs_gemm_dispatch(
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta,
  void* LIBXS_ARGDEF(registry, NULL))
{
  libxs_gemm_shape_t shape;
  libxs_gemm_backend_t be;
  memset(&shape, 0, sizeof(shape));
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
  memset(&be, 0, sizeof(be));
#if defined(mkl_jit_create_dgemm)
  be.jit_create_dgemm = (libxs_jit_create_dgemm_t)mkl_jit_create_dgemm;
  be.jit_get_dgemm = (libxs_jit_get_dgemm_t)mkl_jit_get_dgemm_ptr;
  be.jit_create_sgemm = (libxs_jit_create_sgemm_t)mkl_jit_create_sgemm;
  be.jit_get_sgemm = (libxs_jit_get_sgemm_t)mkl_jit_get_sgemm_ptr;
#endif
#if defined(LIBXSMM_H)
  be.xgemm_dispatch = internal_libxs_xgemm_dispatch;
#endif
#if defined(__MKL) || defined(MKL_H)
  be.dgemm_blas = (libxs_gemm_dblas_t)dgemm;
  be.sgemm_blas = (libxs_gemm_sblas_t)sgemm;
#elif defined(__BLAS)
  { extern void LIBXS_FSYMBOL(dgemm)(
      const char*, const char*,
      const int*, const int*, const int*,
      const double*, const double*, const int*,
      const double*, const int*,
      const double*, double*, const int*);
    extern void LIBXS_FSYMBOL(sgemm)(
      const char*, const char*,
      const int*, const int*, const int*,
      const float*, const float*, const int*,
      const float*, const int*,
      const float*, float*, const int*);
    be.dgemm_blas = LIBXS_FSYMBOL(dgemm);
    be.sgemm_blas = LIBXS_FSYMBOL(sgemm);
  }
#endif
  return libxs_gemm_dispatch_rt(&shape, NULL, &be, registry);
}

/**
 * Call the GEMM kernel previously dispatched into config.
 * Priority: JIT > XGEMM > EXIT_FAILURE.
 * Returns EXIT_SUCCESS if a kernel was called, EXIT_FAILURE otherwise.
 */
LIBXS_API_INLINE int libxs_gemm_call(
  const libxs_gemm_config_t* config,
  const void* a, const void* b, void* c)
{
  int result = EXIT_FAILURE;
  if (0 != libxs_gemm_ready(config)) {
    if (NULL != config->dgemm_jit) {
      config->dgemm_jit(config->jitter, a, b, c);
    }
    else if (NULL != config->sgemm_jit) {
      config->sgemm_jit(config->jitter, a, b, c);
    }
    else {
      libxs_gemm_param_t xparam;
      memset(&xparam, 0, sizeof(xparam));
      xparam.a[0] = a;
      xparam.b[0] = b;
      xparam.c[0] = c;
      config->xgemm(&xparam);
    }
    result = EXIT_SUCCESS;
  }
  return result;
}

/**
 * Release resources acquired by libxs_gemm_dispatch (e.g., MKL jitter).
 * Safe to call even if dispatch was not used or returned zero.
 */
LIBXS_API_INLINE void libxs_gemm_release(libxs_gemm_config_t* config) {
  if (NULL != config) {
#if defined(mkl_jit_create_dgemm)
    if (NULL != config->jitter) {
      mkl_jit_destroy(config->jitter);
      config->jitter = NULL;
    }
#endif
    config->xgemm = NULL;
    config->dgemm_jit = NULL;
    config->sgemm_jit = NULL;
  }
}

/**
 * Release all GEMM configs cached in a registry (e.g., MKL jitter),
 * then destroy the registry itself. Safe to call with NULL.
 */
LIBXS_API_INLINE void libxs_gemm_release_registry(libxs_registry_t* registry) {
  if (NULL != registry) {
    const void* key = NULL;
    size_t cursor = 0;
    libxs_gemm_config_t* config = (libxs_gemm_config_t*)
      libxs_registry_begin(registry, &key, &cursor);
    while (NULL != config) {
      libxs_gemm_release(config);
      config = (libxs_gemm_config_t*)
        libxs_registry_next(registry, &key, &cursor);
    }
    libxs_registry_destroy(registry);
  }
}

/**
 * Dispatch a GEMM config suitable for libxs_syr2k.
 * backend: optional backend pointers (NULL = built-in only).
 * Returns pointer to cached config (registry-owned), or NULL.
 */
LIBXS_API libxs_gemm_config_t* libxs_syr2k_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldb, int ldc,
  const libxs_gemm_backend_t* LIBXS_ARGDEF(backend, NULL),
  void* LIBXS_ARGDEF(registry, NULL));

/**
 * Dispatch a GEMM config suitable for libxs_syrk.
 * Equivalent to libxs_syr2k_dispatch with ldb=lda.
 */
LIBXS_API libxs_gemm_config_t* libxs_syrk_dispatch(
  libxs_data_t datatype, int n, int k, int lda, int ldc,
  const libxs_gemm_backend_t* LIBXS_ARGDEF(backend, NULL),
  void* LIBXS_ARGDEF(registry, NULL));

/**
 * Symmetric rank-2k update: C := alpha*(A*B^T + B*A^T) + beta*C.
 * Only the triangle specified by uplo ('U' or 'L') is written.
 * Scratch is managed internally (TLS buffer, grown as needed).
 * Returns EXIT_SUCCESS on success.
 */
LIBXS_API int libxs_syr2k(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c);

/** Per-thread form of libxs_syr2k. */
LIBXS_API int libxs_syr2k_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, const void* b, void* c,
  int tid, int ntasks);

/**
 * Symmetric rank-k update: C := alpha*A*A^T + beta*C.
 * Only the triangle specified by uplo ('U' or 'L') is written.
 * Scratch is managed internally (TLS buffer, grown as needed).
 * Returns EXIT_SUCCESS on success.
 */
LIBXS_API int libxs_syrk(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c);

/** Per-thread form of libxs_syrk. */
LIBXS_API int libxs_syrk_task(
  const libxs_gemm_config_t* config, char uplo,
  double alpha, double beta,
  const void* a, void* c,
  int tid, int ntasks);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_GEMM_H*/
