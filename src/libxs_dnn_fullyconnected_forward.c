/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_fullyconnected_forward.h"
#include <libxs_intrinsics_x86.h>
#include "libxs_main.h"
#include <libxs.h>
#define STRIDE_BRGEMM

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom_f32_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_custom_bf16_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32(libxs_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxs_dnn_fullyconnected* handle, int start_thread, int tid);


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

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
    libxs_smmfunction_reducebatch_addr batchreduce_kernel = handle->gemm_fwd.xgemm.smra;
#endif
#ifdef OFFSET_BRGEMM
    libxs_smmfunction_reducebatch_offs batchreduce_kernel = handle->gemm_fwd.xgemm.smro;
#endif
#ifdef STRIDE_BRGEMM
    libxs_smmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.smrs;
#endif
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
    libxs_bsmmfunction_reducebatch_addr batchreduce_kernel = handle->gemm_fwd.xgemm.bsmra;
#endif
#ifdef OFFSET_BRGEMM
    libxs_bsmmfunction_reducebatch_offs batchreduce_kernel = handle->gemm_fwd.xgemm.bsmro;
#endif
#ifdef STRIDE_BRGEMM
    libxs_bsmmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.bsmrs;
#endif
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
  } else {
    status = LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
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
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fullyconnected_st_fwd_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 && libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
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

  /* check if all required tensors are bound */
  if (handle->reg_input == 0 || handle->reg_output == 0 ||
      handle->reg_filter == 0                              ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16( handle, start_thread, tid);
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

      if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
        libxs_smmfunction_reducebatch_addr batchreduce_kernel = handle->gemm_fwd.xgemm.smra;
#endif
#ifdef OFFSET_BRGEMM
        libxs_smmfunction_reducebatch_offs batchreduce_kernel = handle->gemm_fwd.xgemm.smro;
#endif
#ifdef STRIDE_BRGEMM
        libxs_smmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.smrs;
#endif
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
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

