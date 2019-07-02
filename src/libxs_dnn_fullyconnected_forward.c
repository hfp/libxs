/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
    status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
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
    status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
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
  element_input_type alpha = (element_input_type)1;
  element_input_type beta = (element_input_type)0;
  libxs_blasint lda = (libxs_blasint)handle->bk;
  libxs_blasint ldb = (libxs_blasint)handle->bc;
  libxs_blasint ldc = (libxs_blasint)handle->bk;

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
    status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxs_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  float alpha = (element_input_type)1;
  float beta = (element_input_type)0;
  libxs_blasint lda = (libxs_blasint)handle->bk;
  libxs_blasint ldb = (libxs_blasint)handle->bc;
  libxs_blasint ldc = (libxs_blasint)handle->bk;

  if ( handle->desc.fuse_ops == LIBXS_DNN_FULLYCONNECTED_FUSE_NONE ) {
    libxs_bsmmfunction_reducebatch_addr batchreduce_kernel = handle->gemm_fwd.xgemm.bsmra;
# include "template/libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
  } else {
    status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
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
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fullyconnected_st_fwd_custom_f32_f32( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 && libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
      status = libxs_dnn_fullyconnected_st_fwd_custom_bf16_f32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
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
        status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
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
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
      status = libxs_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      libxs_blasint lda = (libxs_blasint)handle->bk;
      libxs_blasint ldb = (libxs_blasint)handle->bc;
      libxs_blasint ldc = (libxs_blasint)handle->bk;
      element_input_type beta = (element_input_type)0;
      element_input_type alpha = (element_input_type)1;

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
        status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
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

