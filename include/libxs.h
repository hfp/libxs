/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_H
#define LIBXS_H

#include "libxs_config.h"

/**
 * Strings to denote the version of LIBXS (libxs_config.h).
 * LIBXS_VERSION: Name of the version (stringized version numbers).
 * LIBXS_BRANCH:  Name of the branch this version is derived from.
 */
#define LIBXS_VERSION LIBXS_CONFIG_VERSION
#define LIBXS_BRANCH  LIBXS_CONFIG_BRANCH

/**
 * Semantic version according to https://semver.org/ (see also libxs_config.h).
 * LIBXS_VERSION_MAJOR:  Major version derived from the most recent RCS-tag.
 * LIBXS_VERSION_MINOR:  Minor version derived from the most recent RCS-tag.
 * LIBXS_VERSION_UPDATE: Update number derived from the most recent RCS-tag.
 * LIBXS_VERSION_PATCH:  Patch number based on distance to most recent RCS-tag.
 */
#define LIBXS_VERSION_MAJOR  LIBXS_CONFIG_VERSION_MAJOR
#define LIBXS_VERSION_MINOR  LIBXS_CONFIG_VERSION_MINOR
#define LIBXS_VERSION_UPDATE LIBXS_CONFIG_VERSION_UPDATE
#define LIBXS_VERSION_PATCH  LIBXS_CONFIG_VERSION_PATCH

/**
 * The following interfaces shall be explicitly included,
 * i.e., separate from libxs.h:
 * - libxs_intrinsics_x86.h
 * - libxs_cpuid.h
 * - libxs_sync.h
 * - libxs_mhd.h
*/
#include "libxs_dnn_convolution.h"
#include "libxs_dnn_fullyconnected.h"
#include "libxs_dnn_fusedbatchnorm.h"
#include "libxs_dnn_fusedgroupnorm.h"
#include "libxs_dnn_pooling.h"
#include "libxs_dnn_rnncell.h"
#include "libxs_dnn_softmaxloss.h"
#include "libxs_dnn_optimizer.h"
#include "libxs_generator.h"
#include "libxs_frontend.h"
#include "libxs_fsspmdm.h"
#include "libxs_malloc.h"
#include "libxs_spmdm.h"
#include "libxs_cpuid.h"
#include "libxs_timer.h"
#include "libxs_math.h"
#include "libxs_rng.h"


/** Initialize the library; pay for setup cost at a specific point. */
LIBXS_API void libxs_init(void);
/** De-initialize the library and free internal memory (optional). */
LIBXS_API void libxs_finalize(void);

/**
 * Returns the architecture and instruction set extension as determined by the CPUID flags, as set
 * by the libxs_get_target_arch* functions, or as set by the LIBXS_TARGET environment variable.
 */
LIBXS_API int libxs_get_target_archid(void);
/** Set target architecture (id: see libxs_typedefs.h) for subsequent code generation (JIT). */
LIBXS_API void libxs_set_target_archid(int id);

/**
 * Returns the name of the target architecture as determined by the CPUID flags, as set by the
 * libxs_get_target_arch* functions, or as set by the LIBXS_TARGET environment variable.
 */
LIBXS_API const char* libxs_get_target_arch(void);
/** Set target architecture (arch="0|sse|snb|hsw|knl|knm|skx|clx|cpx", NULL/"0": CPUID). */
LIBXS_API void libxs_set_target_arch(const char* arch);

/** Get the level of verbosity. */
LIBXS_API int libxs_get_verbosity(void);
/**
 * Set the level of verbosity (0: off, positive value: verbosity level,
 * negative value: maximum verbosity, which also dumps JIT-code)
 */
LIBXS_API void libxs_set_verbosity(int level);

/** Get the default prefetch strategy. */
LIBXS_API libxs_gemm_prefetch_type libxs_get_gemm_auto_prefetch(void);
/** Set the default prefetch strategy. */
LIBXS_API void libxs_set_gemm_auto_prefetch(libxs_gemm_prefetch_type strategy);

/** Get information about the matrix multiplication kernel. */
LIBXS_API int libxs_get_mmkernel_info(libxs_xmmfunction kernel, libxs_mmkernel_info* info);
/** Get information about the matrix eltwise kernel. */
LIBXS_API int libxs_get_meltwkernel_info(libxs_xmeltwfunction kernel, libxs_meltwkernel_info* info);

/** Receive information about JIT-generated code (kernel or registry entry). */
LIBXS_API int libxs_get_kernel_info(const void* kernel, libxs_kernel_info* info);
/** Get information about the code registry. */
LIBXS_API int libxs_get_registry_info(libxs_registry_info* info);
/** Enumerate registry by kind (e.g., LIBXS_KERNEL_KIND_USER); can be NULL (no such kind). */
LIBXS_API void* libxs_get_registry_begin(libxs_kernel_kind kind, const void** key);
/** Receive next (or NULL) based on given entry (see libxs_get_registry_begin). */
LIBXS_API void* libxs_get_registry_next(const void* regentry, const void** key);

/**
 * Register user-defined key-value; value can be queried (libxs_xdispatch).
 * Since the key-type is unknown to LIBXS, the key must be binary reproducible,
 * i.e., a structured type (can be padded) must be initialized like a binary blob
 * (memset) followed by an element-wise initialization. The size of the
 * key is limited (see documentation). The given value is copied by LIBXS and
 * can be initialized prior to registration or whenever queried. Registered data
 * is released when the program terminates but can be also released if needed
 * (libxs_xrelease), .e.g., in case of a larger value reusing the same key.
 */
LIBXS_API void* libxs_xregister(const void* key, size_t key_size,
  size_t value_size, const void* value_init);
/** Query user-defined value from LIBXS's code registry. */
LIBXS_API void* libxs_xdispatch(const void* key, size_t key_size);
/** Remove key-value pair from code registry and release memory. */
LIBXS_API void libxs_xrelease(const void* key, size_t key_size);

LIBXS_API libxs_gemm_shape libxs_create_gemm_shape( const libxs_blasint m, const libxs_blasint n, const libxs_blasint k,
                                                          const libxs_blasint lda, const libxs_blasint ldb, const libxs_blasint ldc,
                                                          const libxs_datatype a_in_type, const libxs_datatype b_in_type, const libxs_datatype out_type, const libxs_datatype comp_type );
LIBXS_API libxs_gemm_batch_reduce_config libxs_create_gemm_batch_reduce_config( const libxs_gemm_batch_reduce_type br_type,
                                                                                      const unsigned long long br_stride_a_hint, const unsigned long long br_stride_b_hint,
                                                                                      const unsigned char br_unroll_hint );
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (descriptor form). */
LIBXS_API libxs_xmmfunction libxs_xmmdispatch(const libxs_gemm_descriptor* descriptor);
/** Query or JIT-generate SMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
LIBXS_API libxs_gemmfunction libxs_dispatch_gemm_v2( const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags,
                                                        const libxs_bitfield prefetch_flags );
/** Query or JIT-generate BRGEMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
LIBXS_API libxs_gemmfunction libxs_dispatch_brgemm_v2( const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags,
                                                          const libxs_bitfield prefetch_flags, const libxs_gemm_batch_reduce_config brgemm_config );
/** Query or JIT-generate BRGEMM-kernel with fusion, general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
LIBXS_API libxs_gemmfunction_ext libxs_dispatch_brgemm_ext_v2( const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags,
                                                                  const libxs_bitfield prefetch_flags, const libxs_gemm_batch_reduce_config brgemm_config,
                                                                  const libxs_gemm_ext_unary_argops unary_argops, const libxs_gemm_ext_binary_postops binary_postops );
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction libxs_dmmdispatch_v2( const libxs_blasint m, const libxs_blasint n, const libxs_blasint k,
                                                     const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
                                                     const int* basicflags );
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction libxs_smmdispatch_v2( const libxs_blasint m, const libxs_blasint n, const libxs_blasint k,
                                                     const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
                                                     const int* basicflags );

/* @TODO: deprecate all these GEMM dispatchers
   <---- START HERE ----> */
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction libxs_dmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction libxs_smmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (bf16 inputs, fp32-accumulate) */
LIBXS_API libxs_bsmmfunction libxs_bsmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs) */
LIBXS_API libxs_bmmfunction libxs_bmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (low/short-precision, int-accumulate) */
LIBXS_API libxs_wimmfunction libxs_wimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (low/char-precision, int-accumulate) */
LIBXS_API libxs_ssbimmfunction libxs_ssbimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
LIBXS_API libxs_usbimmfunction libxs_usbimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
LIBXS_API libxs_subimmfunction libxs_subimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
LIBXS_API libxs_uubimmfunction libxs_uubimmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate SMM-kernel; returns NULL if it does not exist or if JIT is not supported (low/char-precision, int-accumulate, int8 outputs) */
LIBXS_API libxs_sububmmfunction libxs_sububmmdispatch(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction_reducebatch_addr libxs_dmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction_reducebatch_addr libxs_smmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXS_API libxs_bsmmfunction_reducebatch_addr libxs_bsmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXS_API libxs_bmmfunction_reducebatch_addr libxs_bmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXS_API libxs_wimmfunction_reducebatch_addr libxs_wimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_ssbimmfunction_reducebatch_addr libxs_ssbimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_usbimmfunction_reducebatch_addr libxs_usbimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_subimmfunction_reducebatch_addr libxs_subimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_uubimmfunction_reducebatch_addr libxs_uubimmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXS_API libxs_sububmmfunction_reducebatch_addr libxs_sububmmdispatch_reducebatch_addr(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction_reducebatch_addr libxs_dmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction_reducebatch_addr libxs_smmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/* Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXS_API libxs_bsmmfunction_reducebatch_addr libxs_bsmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXS_API libxs_bmmfunction_reducebatch_addr libxs_bmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXS_API libxs_wimmfunction_reducebatch_addr libxs_wimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_ssbimmfunction_reducebatch_addr libxs_ssbimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_usbimmfunction_reducebatch_addr libxs_usbimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_subimmfunction_reducebatch_addr libxs_subimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_uubimmfunction_reducebatch_addr libxs_uubimmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXS_API libxs_sububmmfunction_reducebatch_addr libxs_sububmmdispatch_reducebatch_addr_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction_reducebatch_offs libxs_dmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction_reducebatch_offs libxs_smmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXS_API libxs_bsmmfunction_reducebatch_offs libxs_bsmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXS_API libxs_bmmfunction_reducebatch_offs libxs_bmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXS_API libxs_wimmfunction_reducebatch_offs libxs_wimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_ssbimmfunction_reducebatch_offs libxs_ssbimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_usbimmfunction_reducebatch_offs libxs_usbimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_subimmfunction_reducebatch_offs libxs_subimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_uubimmfunction_reducebatch_offs libxs_uubimmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXS_API libxs_sububmmfunction_reducebatch_offs libxs_sububmmdispatch_reducebatch_offs(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction_reducebatch_offs libxs_dmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction_reducebatch_offs libxs_smmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXS_API libxs_bsmmfunction_reducebatch_offs libxs_bsmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXS_API libxs_bmmfunction_reducebatch_offs libxs_bmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXS_API libxs_wimmfunction_reducebatch_offs libxs_wimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_ssbimmfunction_reducebatch_offs libxs_ssbimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_usbimmfunction_reducebatch_offs libxs_usbimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_subimmfunction_reducebatch_offs libxs_subimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_uubimmfunction_reducebatch_offs libxs_uubimmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXS_API libxs_sububmmfunction_reducebatch_offs libxs_sububmmdispatch_reducebatch_offs_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction_reducebatch_strd libxs_dmmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction_reducebatch_strd libxs_smmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXS_API libxs_bsmmfunction_reducebatch_strd libxs_bsmmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXS_API libxs_bmmfunction_reducebatch_strd libxs_bmmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXS_API libxs_wimmfunction_reducebatch_strd libxs_wimmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_ssbimmfunction_reducebatch_strd libxs_ssbimmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_usbimmfunction_reducebatch_strd libxs_usbimmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_subimmfunction_reducebatch_strd libxs_subimmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_uubimmfunction_reducebatch_strd libxs_uubimmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXS_API libxs_sububmmfunction_reducebatch_strd libxs_sububmmdispatch_reducebatch_strd(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (double-precision). */
LIBXS_API libxs_dmmfunction_reducebatch_strd libxs_dmmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (single-precision). */
LIBXS_API libxs_smmfunction_reducebatch_strd libxs_smmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate). */
LIBXS_API libxs_bsmmfunction_reducebatch_strd libxs_bsmmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs). */
LIBXS_API libxs_bmmfunction_reducebatch_strd libxs_bmmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int16 inputs, int32-accumulate). */
LIBXS_API libxs_wimmfunction_reducebatch_strd libxs_wimmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_ssbimmfunction_reducebatch_strd libxs_ssbimmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_usbimmfunction_reducebatch_strd libxs_usbimmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_subimmfunction_reducebatch_strd libxs_subimmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate). */
LIBXS_API libxs_uubimmfunction_reducebatch_strd libxs_uubimmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);
/** Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (int8 inputs, int32-accumulate, int8 outputs). */
LIBXS_API libxs_sububmmfunction_reducebatch_strd libxs_sububmmdispatch_reducebatch_strd_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const int* alpha, const int* beta, const int* flags, const int* prefetch);

/* GEMM + Eltwise fused kernels */
/**
 * Query or JIT-generate reduction kernel; returns NULL if JIT is not supported (bf16 inputs, fp32-accumulate internally, bf16 outputs).
 *
 * This kernel provides the following operation: C = \sum_i A_i * B_i + C_old + colbroadcast(bias) followed by  C_out = Act( C ), the dump of "C" is possible.
 * Also we support elementwise operations on A/B (e.g. decompression of A).
 * */
LIBXS_API libxs_bmmfunction_reducebatch_strd_meltwfused libxs_bmmdispatch_reducebatch_strd_meltwfused(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz);
LIBXS_API libxs_bmmfunction_reducebatch_strd_meltwfused libxs_bmmdispatch_reducebatch_strd_meltwfused_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz);
LIBXS_API libxs_bsmmfunction_reducebatch_strd_meltwfused libxs_bsmmdispatch_reducebatch_strd_meltwfused(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz);
LIBXS_API libxs_bsmmfunction_reducebatch_strd_meltwfused libxs_bsmmdispatch_reducebatch_strd_meltwfused_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint stride_a, libxs_blasint stride_b, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz);

LIBXS_API libxs_bmmfunction_reducebatch_offs_meltwfused libxs_bmmdispatch_reducebatch_offs_meltwfused(libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz);

LIBXS_API libxs_bmmfunction_reducebatch_offs_meltwfused libxs_bmmdispatch_reducebatch_offs_meltwfused_unroll(libxs_blasint m, libxs_blasint n, libxs_blasint k, libxs_blasint unroll_hint,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxs_meltw_operation meltw_op, libxs_datatype meltw_dt, libxs_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz);
/* @TODO deprecation
   <---- END HERE ----> */
/**
 * Process a series of matrix multiplications (batch). See also libxs_gemm_batch/omp.
 * The kind of matrix operands (a, b, c) depend on index_stride:
 * index_stride==0: pointers to pointers of elements, e.g., double** for the C matrices.
 * index_stride!=0: pointer to elements, e.g., const double* for the A and B matrices.
 */
LIBXS_API void libxs_mmbatch(libxs_datatype iprec, libxs_datatype oprec,
  const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const void* alpha, const void* a, const libxs_blasint* lda, const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc,
  /** Determines index-base (usually 0, 1 for one-based indexes); uses the same unit as the strides. */
  libxs_blasint index_base,
  /**
   * Stride used to walk stride_a, stride_b, and stride_c; zero turns stride_* into scalar values.
   * The index_stride is measured in Bytes (sizeof(libxs_blasint) determines packed indexes).
   */
  libxs_blasint index_stride,
  /**
   * Depending on index_stride, the meaning of stride_a, stride_b, and stride_c is different.
   * index_stride==0: stride_a, stride_b, and stride_c are pointers to scalar values.
   * index_stride!=0: stride_* are indexes determining the position of a, b, and c operands.
   */
  const libxs_blasint stride_a[], const libxs_blasint stride_b[], const libxs_blasint stride_c[],
  /**
   * Number of matrix multiplications. If the size is given as a negative value,
   * then internal synchronization is omitted.
   */
  libxs_blasint batchsize,
  /** Thread-ID (TID), and number of threads. */
  /*unsigned*/int tid, /*unsigned*/int ntasks);

/** Process a series of matrix multiplications (batch). See also libxs_mmbatch. */
LIBXS_API void libxs_gemm_batch(libxs_datatype iprec, libxs_datatype oprec,
  const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const void* alpha, const void* a, const libxs_blasint* lda,
                     const void* b, const libxs_blasint* ldb,
   const void* beta,       void* c, const libxs_blasint* ldc,
  libxs_blasint index_base, libxs_blasint index_stride,
  const libxs_blasint stride_a[], const libxs_blasint stride_b[], const libxs_blasint stride_c[],
  libxs_blasint batchsize);

/** Process a series of matrix multiplications (batch) with OpenMP (libxsext). See also libxs_mmbatch. */
LIBXS_APIEXT void libxs_gemm_batch_omp(libxs_datatype iprec, libxs_datatype oprec,
  const char* transa, const char* transb, libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const void* alpha, const void* a, const libxs_blasint* lda,
                     const void* b, const libxs_blasint* ldb,
   const void* beta,       void* c, const libxs_blasint* ldc,
  libxs_blasint index_base, libxs_blasint index_stride,
  const libxs_blasint stride_a[], const libxs_blasint stride_b[], const libxs_blasint stride_c[],
  libxs_blasint batchsize);

/** Unlike libxs_gemm_batch, groups of homogeneous batches are possible (double-precision). */
LIBXS_API void libxs_dgemm_batch(const char transa_array[], const char transb_array[],
  const libxs_blasint m_array[], const libxs_blasint n_array[], const libxs_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxs_blasint lda_array[],
                              const double* b_array[], const libxs_blasint ldb_array[],
   const double beta_array[],       double* c_array[], const libxs_blasint ldc_array[],
  const libxs_blasint* group_count, const libxs_blasint group_size[]);

/** Unlike libxs_gemm_batch, groups of homogeneous batches are possible (single-precision). */
LIBXS_API void libxs_sgemm_batch(const char transa_array[], const char transb_array[],
  const libxs_blasint m_array[], const libxs_blasint n_array[], const libxs_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxs_blasint lda_array[],
                             const float* b_array[], const libxs_blasint ldb_array[],
   const float beta_array[],       float* c_array[], const libxs_blasint ldc_array[],
  const libxs_blasint* group_count, const libxs_blasint group_size[]);

/** Unlike libxs_gemm_batch, groups of homogeneous batches are possible (double-precision). */
LIBXS_APIEXT void libxs_dgemm_batch_omp(const char transa_array[], const char transb_array[],
  const libxs_blasint m_array[], const libxs_blasint n_array[], const libxs_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxs_blasint lda_array[],
                              const double* b_array[], const libxs_blasint ldb_array[],
   const double beta_array[],       double* c_array[], const libxs_blasint ldc_array[],
  const libxs_blasint* group_count, const libxs_blasint group_size[]);

/** Unlike libxs_gemm_batch, groups of homogeneous batches are possible (single-precision). */
LIBXS_APIEXT void libxs_sgemm_batch_omp(const char transa_array[], const char transb_array[],
  const libxs_blasint m_array[], const libxs_blasint n_array[], const libxs_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxs_blasint lda_array[],
                             const float* b_array[], const libxs_blasint ldb_array[],
   const float beta_array[],       float* c_array[], const libxs_blasint ldc_array[],
  const libxs_blasint* group_count, const libxs_blasint group_size[]);

/**
 * This function is a no-op unless LIBXS is built to intercept GEMM calls.
 * Pointer arguments are used to filter intercepted GEMM calls such that
 * non-NULL values match. Otherwise (NULL) the respective argument is
 * considered a "free value", i.e., every value can match; libxsext required.
 */
LIBXS_APIEXT void libxs_mmbatch_begin(libxs_datatype precision, const int* flags,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta);

/** Processes the batch of previously recorded matrix multiplications (libxs_mmbatch_begin); libxsext required. */
LIBXS_APIEXT void libxs_mmbatch_end(void);

/** Code generation routine for matrix-eltwise using a descriptor. */
LIBXS_API libxs_xmeltwfunction libxs_dispatch_meltw( const libxs_meltw_descriptor* descriptor );
LIBXS_API libxs_meltwfunction_opreduce_vecs_idx libxs_dispatch_meltw_opreduce_vecs_idx( const libxs_blasint m, const libxs_blasint* ldi, const libxs_blasint* ldo,
                                                                                              const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype idx_type,
                                                                                              const libxs_meltw_opreduce_vecs_flags flags, const unsigned short bcast_param );
LIBXS_API libxs_meltwfunction_unary libxs_dispatch_meltw_unary( const libxs_blasint m, const libxs_blasint n,
                                                                      const libxs_blasint* ldi, const libxs_blasint* ldo,
                                                                      const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype comp_type,
                                                                      const libxs_meltw_unary_flags flags, const libxs_meltw_unary_type type );
LIBXS_API libxs_meltwfunction_binary libxs_dispatch_meltw_binary( const libxs_blasint m, const libxs_blasint n,
                                                                        const libxs_blasint* ldi, const libxs_blasint* ldi2, const libxs_blasint* ldo,
                                                                        const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype comp_type,
                                                                        const libxs_meltw_binary_flags flags, const libxs_meltw_binary_type type );
LIBXS_API libxs_meltwfunction_ternary libxs_dispatch_meltw_ternary( const libxs_blasint m, const libxs_blasint n,
                                                                          const libxs_blasint* ldi, const libxs_blasint* ldi2, const libxs_blasint* ldi3, const libxs_blasint* ldo,
                                                                          const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype comp_type,
                                                                          const libxs_meltw_ternary_flags flags, const libxs_meltw_ternary_type type );
LIBXS_API libxs_meltw_unary_shape libxs_create_meltw_unary_shape( const libxs_blasint m, const libxs_blasint n,
                                                                        const libxs_blasint ldi, const libxs_blasint ldo,
                                                                        const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype comp_type );
LIBXS_API libxs_meltw_binary_shape libxs_create_meltw_binary_shape( const libxs_blasint m, const libxs_blasint n,
                                                                          const libxs_blasint ldi, const libxs_blasint ldi2, const libxs_blasint ldo,
                                                                          const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype comp_type );
LIBXS_API libxs_meltw_ternary_shape libxs_create_meltw_ternary_shape( const libxs_blasint m, const libxs_blasint n,
                                                                            const libxs_blasint ldi, const libxs_blasint ldi2, const libxs_blasint ldi3, const libxs_blasint ldo,
                                                                            const libxs_datatype in_type, const libxs_datatype out_type, const libxs_datatype comp_type );
LIBXS_API libxs_meltwfunction_unary libxs_dispatch_meltw_unary_v2( const libxs_meltw_unary_type unary_type, const libxs_meltw_unary_shape unary_shape, const libxs_bitfield unary_flags );
LIBXS_API libxs_meltwfunction_binary libxs_dispatch_meltw_binary_v2( const libxs_meltw_binary_type binary_type, const libxs_meltw_binary_shape binary_shape, const libxs_bitfield binary_flags );
LIBXS_API libxs_meltwfunction_ternary libxs_dispatch_meltw_ternary_v2( const libxs_meltw_ternary_type ternary_type, const libxs_meltw_ternary_shape ternary_shape, const libxs_bitfield ternary_flags );

/** matrix equation interface */
LIBXS_API libxs_blasint libxs_matrix_eqn_create(void);
LIBXS_API int libxs_matrix_eqn_push_back_arg( const libxs_blasint idx, const libxs_blasint m, const libxs_blasint n, const libxs_blasint ld,
                                                  const libxs_blasint in_pos, const libxs_blasint offs_in_pos, const libxs_datatype dtype );
LIBXS_API int libxs_matrix_eqn_push_back_unary_op( const libxs_blasint idx, const libxs_meltw_unary_type type, const libxs_meltw_unary_flags flags, const libxs_datatype dtype );
LIBXS_API int libxs_matrix_eqn_push_back_binary_op( const libxs_blasint idx, const libxs_meltw_binary_type type, const libxs_meltw_binary_flags flags, const libxs_datatype dtype );
LIBXS_API int libxs_matrix_eqn_push_back_ternary_op( const libxs_blasint idx, const libxs_meltw_ternary_type type, const libxs_meltw_ternary_flags flags, const libxs_datatype dtype );

LIBXS_API libxs_meqn_arg_shape libxs_create_meqn_arg_shape( const libxs_blasint m, const libxs_blasint n, const libxs_blasint ld, const libxs_datatype type );
LIBXS_API libxs_matrix_arg_attributes libxs_create_matrix_arg_attributes( const libxs_matrix_arg_type type, const libxs_matrix_arg_set_type set_type, const libxs_blasint set_cardinality_hint, const libxs_blasint set_stride_hint );
LIBXS_API libxs_matrix_eqn_arg_metadata libxs_create_matrix_eqn_arg_metadata( const libxs_blasint eqn_idx, const libxs_blasint in_arg_pos );
LIBXS_API int libxs_matrix_eqn_push_back_arg_v2( const libxs_matrix_eqn_arg_metadata arg_metadata, const libxs_meqn_arg_shape arg_shape, libxs_matrix_arg_attributes arg_attr);
LIBXS_API int libxs_matrix_eqn_push_back_unary_op_v2( const libxs_matrix_eqn_op_metadata op_metadata, const libxs_meltw_unary_type type, const libxs_datatype dtype, const libxs_bitfield flags);
LIBXS_API int libxs_matrix_eqn_push_back_binary_op_v2( const libxs_matrix_eqn_op_metadata op_metadata, const libxs_meltw_binary_type type, const libxs_datatype dtype, const libxs_bitfield flags);
LIBXS_API int libxs_matrix_eqn_push_back_ternary_op_v2( const libxs_matrix_eqn_op_metadata op_metadata, const libxs_meltw_ternary_type type, const libxs_datatype dtype, const libxs_bitfield flags);

LIBXS_API void libxs_matrix_eqn_tree_print( const libxs_blasint idx );
LIBXS_API void libxs_matrix_eqn_rpn_print( const libxs_blasint idx );
LIBXS_API libxs_matrix_eqn_function libxs_dispatch_matrix_eqn_desc( const libxs_meqn_descriptor* descriptor );
LIBXS_API libxs_matrix_eqn_function libxs_dispatch_matrix_eqn( const libxs_blasint m, const libxs_blasint n, const libxs_blasint* ldo, const libxs_datatype out_type, const unsigned int eqn_idx );
LIBXS_API libxs_matrix_eqn_function libxs_dispatch_matrix_eqn_v2( const libxs_blasint idx, const libxs_meqn_arg_shape out_shape );

/**
 * Code generation routine for the CSR format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxs_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXS_API libxs_xmmfunction libxs_create_packed_spxgemm_csr(const libxs_gemm_descriptor* descriptor, unsigned int packed_width,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values);
LIBXS_API libxs_gemmfunction libxs_create_packed_spgemm_csr_v2(
  const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags, const libxs_bitfield prefetch_flags, const libxs_blasint packed_width,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values);

/**
 * Code generation routine for the CSC format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxs_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXS_API libxs_xmmfunction libxs_create_packed_spxgemm_csc(const libxs_gemm_descriptor* descriptor, unsigned int packed_width,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values);
LIBXS_API libxs_gemmfunction libxs_create_packed_spgemm_csc_v2(
  const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags, const libxs_bitfield prefetch_flags, const libxs_blasint packed_width,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values);

/**
 * Code generation routine for row-major format B matrix which is multiplied by a dense packed matrix (each element holds a SIMD-width
 * wide vector) and the result is another packed matrix. The memory layout of the SOA matrix is [row][col][packed].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxs_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXS_API libxs_xmmfunction libxs_create_packed_xgemm_ac_rm(const libxs_gemm_descriptor* descriptor, unsigned int packed_width);
LIBXS_API libxs_gemmfunction libxs_create_packed_gemm_ac_rm_v2( const libxs_gemm_shape gemm_shape,
  const libxs_bitfield gemm_flags, const libxs_bitfield prefetch_flags, const libxs_blasint packed_width );

/**
 * Code generation routine for row-major format A matrix which is multiplied by a dense packed matrix (each element holds a SIMD-width
 * wide vector) and the result is another packed matrix. The memory layout of the packed matrix is [row][col][packed].
 * here is no code cache, and user code has to manage the code pointers.
 * Call libxs_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXS_API libxs_xmmfunction libxs_create_packed_xgemm_bc_rm(const libxs_gemm_descriptor* descriptor, unsigned int packed_width);
LIBXS_API libxs_gemmfunction libxs_create_packed_gemm_bc_rm_v2( const libxs_gemm_shape gemm_shape,
  const libxs_bitfield gemm_flags, const libxs_bitfield prefetch_flags, const libxs_blasint packed_width );

/**
 * Code generation routine for the CSR format which multiplies a dense matrix "b" into a dense matrix "c".
 * The sparse matrix "a" is kept in registers.
 * Call libxs_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXS_API libxs_dmmfunction libxs_create_dcsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values);

/**
 * Code generation routine for the CSR format which multiplies a dense matrix "b" into a dense matrix "c".
 * The sparse matrix "a" is kept in registers.
 * Call libxs_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXS_API libxs_smmfunction libxs_create_scsr_reg(const libxs_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const float* values);

LIBXS_API libxs_gemmfunction libxs_create_spgemm_csr_areg_v2( const libxs_gemm_shape gemm_shape,
  const libxs_bitfield gemm_flags, const libxs_bitfield prefetch_flags,
  const libxs_blasint max_N, const unsigned int* row_ptr, const unsigned int* column_idx, const double* values );

/**
 * Deallocates the JIT'ted code as returned by libxs_create_* functions,
 * unregisters and releases code from the code registry.
 */
LIBXS_API void libxs_release_kernel(const void* kernel);

/** Matrix copy function; "in" can be NULL to zero the destination (BLAS-like equivalent is "omatcopy"). */
LIBXS_API void libxs_matcopy(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo);

/** Matrix copy function (per-thread form); "in" can be NULL when zeroing (BLAS-like equivalent is "omatcopy"). */
LIBXS_API void libxs_matcopy_task(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  /*unsigned*/int tid, /*unsigned*/int ntasks);

/** Matrix copy function (MT via libxsext); "in" can be NULL when zeroing (BLAS-like equivalent is "omatcopy"). */
LIBXS_APIEXT void libxs_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo);

/** Matrix transposition; out-of-place form (BLAS-like equivalent is "omatcopy"). */
LIBXS_API void libxs_otrans(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo);

/** Matrix transposition (per-thread form); out-of-place (BLAS-like equivalent is "omatcopy"). */
LIBXS_API void libxs_otrans_task(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  /*unsigned*/int tid, /*unsigned*/int ntasks);

/** Matrix transposition (MT via libxsext); out-of-place (BLAS-like equivalent is "omatcopy"). */
LIBXS_APIEXT void libxs_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo);

/** Matrix transposition; in-place (BLAS-like equivalent is "imatcopy"). */
LIBXS_API void libxs_itrans(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo);

/** Series/batch of matrix transpositions; in-place. See also libxs_mmbatch. */
LIBXS_API void libxs_itrans_batch(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  libxs_blasint index_base, libxs_blasint index_stride,
  const libxs_blasint stride[], libxs_blasint batchsize,
  /*unsigned*/int tid, /*unsigned*/int ntasks);

/** Series/batch of matrix transpositions ((MT via libxsext)); in-place. */
LIBXS_APIEXT void libxs_itrans_batch_omp(void* inout, unsigned int typesize,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo,
  libxs_blasint index_base, libxs_blasint index_stride,
  const libxs_blasint stride[], libxs_blasint batchsize);

/** Initialize GEMM-handle; allows to better amortize setup overhead. */
LIBXS_API libxs_gemm_handle* libxs_gemm_handle_init(libxs_gemm_blob* blob,
  libxs_datatype iprec, libxs_datatype oprec, const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const libxs_blasint* lda, const libxs_blasint* ldb, const libxs_blasint* ldc,
  const void* alpha, const void* beta, int flags, /*unsigned*/int ntasks);

/** Calculate required scratch buffer size needed to perform libxs_gemm_task. */
LIBXS_API size_t libxs_gemm_handle_get_scratch_size(const libxs_gemm_handle* handle);

/** Low-level type-agnostic GEMM suitable for external threads or tasks. */
LIBXS_API void libxs_gemm_task(const libxs_gemm_handle* handle, void* scratch,
  const void* a, const void* b, void* c, /*unsigned*/int tid, /*unsigned*/int ntasks);

/** General dense matrix multiplication (sequential). */
LIBXS_API void libxs_xgemm(libxs_datatype iprec, libxs_datatype oprec,
  const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda, const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);

/** General dense matrix multiplication (libxsext); available as xgemm (generic), dgemm (DP), and sgemm (SP). */
LIBXS_APIEXT void libxs_xgemm_omp(libxs_datatype iprec, libxs_datatype oprec,
  const char* transa, const char* transb, const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const void* alpha, const void* a, const libxs_blasint* lda, const void* b, const libxs_blasint* ldb,
  const void* beta, void* c, const libxs_blasint* ldc);

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXS_API void libxs_dgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
  const double* b, const libxs_blasint* ldb,
  const double* beta, double* c, const libxs_blasint* ldc);
/** Dispatched general dense matrix multiplication (single-precision). */
LIBXS_API void libxs_sgemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
  const float* b, const libxs_blasint* ldb,
  const float* beta, float* c, const libxs_blasint* ldc);

#if !defined(LIBXS_DEFAULT_CONFIG) && !defined(LIBXS_SOURCE_H)

#endif /*!defined(LIBXS_DEFAULT_CONFIG)*/
#if defined(__cplusplus)

/** Map built-in type to libxs_datatype (libxs_datatype_enum). */
template<typename T> struct LIBXS_RETARGETABLE libxs_datatype_enum          { static const libxs_datatype value = static_cast<libxs_datatype>(LIBXS_DATATYPE_UNSUPPORTED); };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<double>            { static const libxs_datatype value = LIBXS_DATATYPE_F64; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<float>             { static const libxs_datatype value = LIBXS_DATATYPE_F32; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<int>               { static const libxs_datatype value = LIBXS_DATATYPE_I32; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum</*signed*/short>   { static const libxs_datatype value = LIBXS_DATATYPE_I16; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<libxs_bfloat16>  { static const libxs_datatype value = LIBXS_DATATYPE_BF16; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<Eigen::bfloat16>   { static const libxs_datatype value = LIBXS_DATATYPE_BF16; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<signed char>       { static const libxs_datatype value = LIBXS_DATATYPE_I8; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<unsigned char>     { static const libxs_datatype value = LIBXS_DATATYPE_I8; };
template<> struct LIBXS_RETARGETABLE libxs_datatype_enum<char>              { static const libxs_datatype value = LIBXS_DATATYPE_I8; };

/** Determine default output type based on the input-type. */
template<typename INP_TYPE> struct LIBXS_RETARGETABLE libxs_gemm_default_output   { typedef INP_TYPE type; };
template<> struct LIBXS_RETARGETABLE libxs_gemm_default_output</*signed*/short>   { typedef int type; };
template<> struct LIBXS_RETARGETABLE libxs_gemm_default_output<libxs_bfloat16>  { typedef float type; };

/** Construct and execute a specialized function. */
template<typename INP_TYPE, typename OUT_TYPE = typename libxs_gemm_default_output<INP_TYPE>::type>
class LIBXS_RETARGETABLE libxs_mmfunction {
  mutable/*retargetable*/ libxs_xmmfunction m_function;
public:
  typedef INP_TYPE itype;
  typedef OUT_TYPE otype;
public:
  libxs_mmfunction() { m_function.xmm = 0; }
  libxs_mmfunction(libxs_blasint m, libxs_blasint n, libxs_blasint k, int flags = LIBXS_FLAGS) {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, m, k, m,
      NULL/*alpha*/, NULL/*beta*/, flags, libxs_get_gemm_xprefetch(NULL));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
  libxs_mmfunction(int flags, libxs_blasint m, libxs_blasint n, libxs_blasint k, int prefetch) {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, m, k, m,
      NULL/*alpha*/, NULL/*beta*/, flags, libxs_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
  libxs_mmfunction(int flags, libxs_blasint m, libxs_blasint n, libxs_blasint k, otype alpha, otype beta) {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, m, k, m,
      &alpha, &beta, flags, libxs_get_gemm_xprefetch(NULL));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
  libxs_mmfunction(int flags, libxs_blasint m, libxs_blasint n, libxs_blasint k, otype alpha, otype beta, int prefetch) {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, m, k, m,
      &alpha, &beta, flags, libxs_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
  libxs_mmfunction(int flags, libxs_blasint m, libxs_blasint n, libxs_blasint k,
    libxs_blasint lda, libxs_blasint ldb, libxs_blasint ldc, int prefetch)
  {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, lda, ldb, ldc,
      NULL/*alpha*/, NULL/*beta*/, flags, libxs_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
  libxs_mmfunction(int flags, libxs_blasint m, libxs_blasint n, libxs_blasint k,
    libxs_blasint lda, libxs_blasint ldb, libxs_blasint ldc, otype alpha, otype beta)
  {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, lda, ldb, ldc,
      &alpha, &beta, flags, libxs_get_gemm_xprefetch(NULL));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
  libxs_mmfunction(int flags, libxs_blasint m, libxs_blasint n, libxs_blasint k,
    libxs_blasint lda, libxs_blasint ldb, libxs_blasint ldc, otype alpha, otype beta, int prefetch)
  {
    libxs_descriptor_blob blob;
    const libxs_gemm_descriptor *const desc = libxs_gemm_descriptor_init2(&blob,
      libxs_datatype_enum<itype>::value, libxs_datatype_enum<otype>::value, m, n, k, lda, ldb, ldc,
      &alpha, &beta, flags, libxs_get_gemm_prefetch(prefetch));
    m_function.xmm = (0 != desc ? libxs_xmmdispatch(desc).xmm : 0);
  }
public:
  const libxs_xmmfunction& kernel() const {
    return m_function;
  }
  operator const void*() const {
    return 0 != m_function.xmm ? this : 0;
  }
  void operator()(const itype* a, const itype* b, otype* c) const {
    LIBXS_MMCALL_ABC(m_function.xmm, a, b, c);
  }
  void operator()(const itype* a, const itype* b, otype* c, const itype* pa, const itype* pb, const otype* pc) const {
    LIBXS_UNUSED( pa );
    LIBXS_UNUSED( pb );
    LIBXS_UNUSED( pc );
    LIBXS_MMCALL_PRF(m_function.xmm, a, b, c, pa, pb, pc);
  }
};

/** Matrix copy function ("in" can be NULL to zero the destination). */
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_matcopy(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_matcopy(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy(T* out, const T* in,
  libxs_blasint m, libxs_blasint n)
{
  return libxs_matcopy(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy(T* out, const T* in,
  libxs_blasint n)
{
  return libxs_matcopy(out, in, n, n);
}

/** Matrix copy function ("in" can be NULL to zero the destination); MT via libxsext. */
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy_omp(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_matcopy_omp(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy_omp(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_matcopy_omp(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy_omp(T* out, const T* in,
  libxs_blasint m, libxs_blasint n)
{
  return libxs_matcopy_omp(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_matcopy_omp(T* out, const T* in,
  libxs_blasint n)
{
  return libxs_matcopy_omp(out, in, n, n);
}

/** Matrix transposition (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_otrans(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_trans(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* out, const T* in,
  libxs_blasint m, libxs_blasint n)
{
  return libxs_trans(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* out, const T* in,
  libxs_blasint n)
{
  return libxs_trans(out, in, n, n);
}

/** Matrix transposition; MT via libxsext (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans_omp(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_otrans_omp(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans_omp(T* out, const T* in,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_trans_omp(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans_omp(T* out, const T* in,
  libxs_blasint m, libxs_blasint n)
{
  return libxs_trans_omp(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans_omp(T* out, const T* in,
  libxs_blasint n)
{
  return libxs_trans_omp(out, in, n, n);
}

/** Matrix transposition (in-place form). */
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi, libxs_blasint ldo)
{
  return libxs_itrans(inout, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* inout,
  libxs_blasint m, libxs_blasint n, libxs_blasint ldi)
{
  return libxs_itrans(inout, sizeof(T), m, n, ldi, n);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* inout,
  libxs_blasint m, libxs_blasint n)
{
  return libxs_itrans(inout, sizeof(T), m, n, m, n);
}
template<typename T> inline/*superfluous*/ LIBXS_RETARGETABLE int libxs_trans(T* inout,
  libxs_blasint m)
{
  return libxs_itrans(inout, sizeof(T), m, m, m, m);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
                       const double* b, const libxs_blasint* ldb,
   const double* beta,       double* c, const libxs_blasint* ldc)
{
  libxs_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb,
  /* by-value */ libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const double* alpha, const double* a, const libxs_blasint* lda,
                       const double* b, const libxs_blasint* ldb,
   const double* beta,       double* c, const libxs_blasint* ldc)
{
  libxs_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
                      const float* b, const libxs_blasint* ldb,
   const float* beta,       float* c, const libxs_blasint* ldc)
{
  libxs_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXS_RETARGETABLE void libxs_gemm(const char* transa, const char* transb,
  /* by-value */ libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const float* alpha, const float* a, const libxs_blasint* lda,
                      const float* b, const libxs_blasint* ldb,
   const float* beta,       float* c, const libxs_blasint* ldc)
{
  libxs_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const double* alpha, const double* a, const libxs_blasint* lda,
                       const double* b, const libxs_blasint* ldb,
   const double* beta,       double* c, const libxs_blasint* ldc)
{
  libxs_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb,
  /* by-value */ libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const double* alpha, const double* a, const libxs_blasint* lda,
                       const double* b, const libxs_blasint* ldb,
   const double* beta,       double* c, const libxs_blasint* ldc)
{
  libxs_blas_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb,
  const libxs_blasint* m, const libxs_blasint* n, const libxs_blasint* k,
  const float* alpha, const float* a, const libxs_blasint* lda,
                      const float* b, const libxs_blasint* ldb,
   const float* beta,       float* c, const libxs_blasint* ldc)
{
  libxs_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
inline LIBXS_RETARGETABLE void libxs_blas_gemm(const char* transa, const char* transb,
  /* by-value */ libxs_blasint m, libxs_blasint n, libxs_blasint k,
  const float* alpha, const float* a, const libxs_blasint* lda,
                      const float* b, const libxs_blasint* ldb,
   const float* beta,       float* c, const libxs_blasint* ldc)
{
  libxs_blas_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*__cplusplus*/
#endif /*LIBXS_H*/

