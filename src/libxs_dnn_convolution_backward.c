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
#include "libxs_dnn_convolution_backward.h"
#include <libxs_intrinsics_x86.h>
#include "libxs_main.h"
#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_rsck_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxs_dnn_layer* handle, int start_thread, int tid);


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxs_blasint ldB = (libxs_blasint)handle->ofmblock;
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->ifmblock * handle->desc.v) : (libxs_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
    int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
  } else {
    const libxs_blasint ldC = (libxs_blasint)(handle->desc.v*handle->ifmblock);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxs_smmfunction gemm_function;
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxs_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldC, NULL, NULL, NULL, NULL);
#include "template/libxs_dnn_convolve_st_bwd_custom_custom_fallback_generic.tpl.c"
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxs_blasint ldB = (libxs_blasint)handle->ofmblock;
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->ifmblock * handle->desc.v) : (libxs_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
    typedef libxs_bfloat16 element_input_type;
    typedef libxs_bfloat16 element_output_type;
    typedef libxs_bfloat16 element_filter_type;
    typedef libxs_bsmmfunction_reducebatch_addr gemm_br_function;
    typedef libxs_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
    int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxs_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxs_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_bwd_custom_custom_generic_bf16.tpl.c"
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


#if defined(LIBXS_INTRINSICS_AVX512_CPX)
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxs_blasint ldB = (libxs_blasint)handle->ofmblock;
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->ifmblock * handle->desc.v) : (libxs_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
    typedef libxs_bfloat16 element_input_type;
    typedef libxs_bfloat16 element_output_type;
    typedef libxs_bfloat16 element_filter_type;
    typedef libxs_bsmmfunction_reducebatch_addr gemm_br_function;
    typedef libxs_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
    int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxs_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxs_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_CONVOLUTION_BWD_AVX512_CPX
# include "template/libxs_dnn_convolve_st_bwd_custom_custom_generic_bf16.tpl.c"
#undef LIBXS_DNN_CONVOLUTION_BWD_AVX512_CPX
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxs_dnn_layer* handle, int start_thread, int tid)
{
  return libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid );
}
#endif


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxs_blasint)(handle->blocksifm * handle->ifmblock);
    const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
    int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
  } else {
    const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxs_smmfunction gemm_function;
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxs_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_rsck_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxs_blasint)(handle->blocksifm * handle->ifmblock);
    const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
    int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
  } else {
    const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
    const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
    const libxs_blasint ldC = (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxs_smmfunction gemm_function;
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxs_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_custom_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_convolve_st_bwd_custom_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && libxs_target_archid >= LIBXS_X86_AVX512_CORE && libxs_target_archid < LIBXS_X86_AVX512_CPX ) {
      status = libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && libxs_target_archid >= LIBXS_X86_AVX512_CPX ) {
      status = libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16( handle, start_thread, tid);
    }
#elif defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
      status = libxs_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    }
#endif
    else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      if (handle->use_fallback_bwd_loops == 0) {
        const libxs_blasint ldx = ((libxs_blasint)handle->ofmblock);
        const libxs_blasint ldA = handle->ifmblock;
        const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? handle->ifmblock * handle->desc.v : handle->ifmblock;
        const float beta = (handle->avoid_acc_load_bwd) ? 0.f : 1.f;
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
        int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
        gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
      } else {
        const libxs_blasint ldx = ((libxs_blasint)handle->desc.v*handle->ifmblock);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxs_smmfunction gemm_function;
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxs_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldx, NULL, NULL, NULL, NULL);
#include "template/libxs_dnn_convolve_st_bwd_custom_custom_fallback_generic.tpl.c"
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_rsck(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_convolve_st_bwd_nhwc_rsck_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      if (handle->use_fallback_bwd_loops == 0) {
        const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
        const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
        const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxs_blasint)(handle->blocksifm * handle->ifmblock);
        const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
        int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
        gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
      } else {
        const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
        const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
        const libxs_blasint ldC = (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxs_smmfunction gemm_function;
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxs_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_bwd_nhwc_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_convolve_st_bwd_nhwc_custom_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      if (handle->use_fallback_bwd_loops == 0) {
        const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
        const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
        const libxs_blasint ldC = (handle->spread_input_bwd == 1) ? (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxs_blasint)(handle->blocksifm * handle->ifmblock);
        const float  beta = (handle->avoid_acc_load_bwd) ? 0.0 : 1.0;
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
        int l_flags = LIBXS_GEMM_FLAGS('N', 'N');
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
        gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
      } else {
        const libxs_blasint ldB = (libxs_blasint)(handle->blocksofm * handle->ofmblock);
        const libxs_blasint ldA = (libxs_blasint)handle->ifmblock;
        const libxs_blasint ldC = (libxs_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxs_smmfunction gemm_function;
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxs_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#include "template/libxs_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXS_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

