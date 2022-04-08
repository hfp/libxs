/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_fullyconnected_forward.h"
#include "libxs_main.h"

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom_f32_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom_bf16_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx_emu(libxs_dnn_fullyconnected* handle, int start_thread, int tid);

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom_f32_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxs_smmfunction gemm_function;
  element_input_type alpha = (element_input_type)1;
  element_input_type beta = (element_input_type)0;
  libxs_blasint lda = (libxs_blasint)handle->ofmblock;
  libxs_blasint ldb = (libxs_blasint)handle->desc.C;
  libxs_blasint ldc = (libxs_blasint)handle->desc.K;

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
    gemm_function gemm_kernel = libxs_smmdispatch(handle->ofmblock, handle->desc.N, handle->desc.C, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# include "template/libxs_dnn_fullyconnected_st_fwd_custom_generic.tpl.c"
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom_bf16_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef float element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  typedef libxs_smmfunction gemm_function;
  libxs_blasint lda = (libxs_blasint)handle->ofmblock;
  libxs_blasint ldb = (libxs_blasint)handle->desc.C;
  libxs_blasint ldc = (libxs_blasint)handle->desc.K;
  float alpha = (element_input_type)1;
  float beta = (element_input_type)0;

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
    gemm_function gemm_kernel = libxs_smmdispatch(handle->ofmblock, handle->desc.N, handle->desc.C, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# define LIBXS_DNN_FULLYCONNECTED_FWD_BF16_F32
# include "template/libxs_dnn_fullyconnected_st_fwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FULLYCONNECTED_FWD_BF16_F32
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  libxs_smmfunction_reducebatch_strd batchreduce_kernel_beta     = handle->gemm_fwd.xgemm.smrs;
  libxs_smmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.smrs;

#define LIBXS_DNN_FC_FWD_USE_AVX512
  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#undef LIBXS_DNN_FC_FWD_USE_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  libxs_bsmmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.bsmrs;
  libxs_bmmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.bmrs;
  libxs_bmmfunction_reducebatch_strd batchreduce_kernel_beta = handle->gemm_fwd3.xgemm.bmrs;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXS_INTRINSICS_AVX512_CPX)
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  libxs_bsmmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.bsmrs;
  libxs_bmmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.bmrs;
  libxs_bmmfunction_reducebatch_strd batchreduce_kernel_beta = handle->gemm_fwd3.xgemm.bmrs;

#define LIBXS_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXS_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  return libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu( handle, start_thread, tid );
}
#endif

#if defined(LIBXS_INTRINSICS_AVX512_CPX)
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  libxs_bsmmfunction_reducebatch_strd batchreduce_kernel              = handle->gemm_fwd.xgemm.bsmrs;
  libxs_bmmfunction_reducebatch_strd bf16_batchreduce_kernel_zerobeta = handle->gemm_fwd3.xgemm.bmrs;
  libxs_bsmmfunction tile_config_kernel = handle->fwd_config_kernel;
#define LIBXS_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if (handle->compressed_A == 1) {
    libxs_xmmfunction batchreduce_kernel_decompress = handle->sparse_gemm9;
    libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_decompress = handle->sparse_gemm11;
    if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd4.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm12;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd5.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm13;
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd6.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm14;
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd7.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm15;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd8.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm16;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else {
      status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
    }
  } else {
    if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd4.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd5.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd6.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd7.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd8.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else {
      status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
    }
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXS_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx(libxs_dnn_fullyconnected* handle, int start_thread, int tid) {
  return libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx_emu( handle, start_thread, tid );
}
#endif

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx_emu(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  libxs_bsmmfunction_reducebatch_strd batchreduce_kernel              = handle->gemm_fwd.xgemm.bsmrs;
  libxs_bmmfunction_reducebatch_strd bf16_batchreduce_kernel_zerobeta = handle->gemm_fwd3.xgemm.bmrs;
  libxs_bsmmfunction tile_config_kernel = handle->fwd_config_kernel;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if (handle->compressed_A == 1) {
    libxs_xmmfunction batchreduce_kernel_decompress = handle->sparse_gemm9;
    libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_decompress = handle->sparse_gemm11;
    if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd4.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm12;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd5.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm13;
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd6.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm14;
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd7.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm15;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd8.xgemm.bmrs_meltwfused;
      libxs_xmmfunction bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress = handle->sparse_gemm16;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_sparse_A_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else {
      status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
    }
  } else {
    if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd4.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd5.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd6.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd7.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
      libxs_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise = handle->gemm_fwd8.xgemm.bmrs_meltwfused;
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
    } else {
      status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
    }
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"

#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if (handle->reg_input == 0 || handle->reg_output == 0 ||
      handle->reg_filter == 0                              ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXS_X86_AVX512) && (handle->target_archid <= LIBXS_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fullyconnected_st_fwd_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 && handle->target_archid >= LIBXS_X86_AVX512_CORE ) {
      status = libxs_dnn_fullyconnected_st_fwd_custom_bf16_f32( handle, start_thread, tid);
    }
#endif
    else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_smmfunction gemm_function;
      libxs_blasint lda = (libxs_blasint)handle->ofmblock;
      libxs_blasint ldb = (libxs_blasint)handle->desc.C;
      libxs_blasint ldc = (libxs_blasint)handle->desc.K;
      element_input_type beta = (element_input_type)0;
      element_input_type alpha = (element_input_type)1;

      if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
        gemm_function gemm_kernel = libxs_smmdispatch(handle->ofmblock, handle->desc.N, handle->desc.C, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# include "template/libxs_dnn_fullyconnected_st_fwd_custom_generic.tpl.c"
      } else {
        status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int l_emu_amx = 0;
  const char *const l_env_emu_amx = getenv("EMULATE_AMX");
  if ( 0 == l_env_emu_amx ) {
  } else {
    l_emu_amx = atoi(l_env_emu_amx);
  }

  /* check if all required tensors are bound */
  if (handle->reg_input == 0 || handle->reg_output == 0 ||
      handle->reg_filter == 0                              ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( ((handle->desc.fuse_ops & LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) != 0) && ( handle->reg_bias == 0 ) )  {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( ((handle->desc.fuse_ops & LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) != 0) && ( handle->relumask == 0 ) )  {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXS_X86_AVX512) && (handle->target_archid <= LIBXS_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_CORE && handle->target_archid < LIBXS_X86_AVX512_CPX) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_CPX && handle->target_archid < LIBXS_X86_AVX512_SPR) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_SPR) {
      if ( l_emu_amx == 0 ) {
        status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx( handle, start_thread, tid);
      } else {
        status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx_emu( handle, start_thread, tid);
      }
    }
#elif defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_CORE && handle->target_archid < LIBXS_X86_AVX512_SPR ) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_SPR ) {
      if ( l_emu_amx == 0 ) {
        status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx( handle, start_thread, tid);
      } else {
        status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_amx_emu( handle, start_thread, tid);
      }
    }
#endif
    else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    LIBXS_UNUSED( l_emu_amx );
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      libxs_smmfunction_reducebatch_strd batchreduce_kernel_beta     = handle->gemm_fwd.xgemm.smrs;
      libxs_smmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.smrs;

      if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXS_DNN_FC_FWD_FUSE_NONE
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_NONE
      } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
      } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
      } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
      } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_RELU
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_RELU
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
      } else if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXS_DNN_FC_FWD_FUSE_BIAS
#define LIBXS_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXS_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXS_DNN_FC_FWD_FUSE_BIAS
      } else {
        status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_nhwc(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );
  return status;
}

