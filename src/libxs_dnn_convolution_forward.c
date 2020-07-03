/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_convolution_forward.h"
#include "libxs_main.h"

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_nhwc_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_nhwc_rsck_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu_amx(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_amx(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_i8_i32(libxs_dnn_layer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_i8_i8(libxs_dnn_layer* handle, int start_thread, int tid);


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->ifmblock : (libxs_blasint)handle->desc.v*handle->ifmblock;
  const libxs_blasint ldA = handle->ofmblock;
  const libxs_blasint ldC = handle->ofmblock;
  const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
  int l_flags = ( LIBXS_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
  /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
  gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic.tpl.c"
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"
  const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->ifmblock : (libxs_blasint)handle->desc.v*handle->ifmblock;
  const libxs_blasint ldA = handle->ofmblock;
  const libxs_blasint ldC = handle->ofmblock;
  const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
  typedef libxs_bsmmfunction_reducebatch_addr gemm_br_function;
  typedef libxs_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
  int l_flags = ( LIBXS_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') )| handle->fwd_flags;

  /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
  gemm_br_function br_gemm_kernel = libxs_bsmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function br_gemm_kernel2 = libxs_bsmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic_bf16.tpl.c"

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu_amx(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  typedef libxs_bsmmfunction gemm_function;
  typedef libxs_bsmmfunction_reducebatch_offs gemm_br_function_offs;
  typedef libxs_bsmmfunction_reducebatch_strd gemm_br_function_strd;
  gemm_br_function_offs br_gemm_kernel_offs = handle->fwd_compute_kernel_offs;
  gemm_br_function_strd br_gemm_kernel_strd = handle->fwd_compute_kernel_strd;
  gemm_function tile_config_kernel = handle->fwd_config_kernel;
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic_bf16_amx.tpl.c"


# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXS_INTRINSICS_AVX512_CPX)
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

#define LIBXS_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  typedef libxs_bsmmfunction_reducebatch_addr gemm_br_function;
  typedef libxs_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
  const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->ifmblock : (libxs_blasint)handle->desc.v*handle->ifmblock;
  const libxs_blasint ldA = handle->ofmblock;
  const libxs_blasint ldC = handle->ofmblock;
  const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
  int l_flags = ( LIBXS_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | handle->fwd_flags;
  gemm_br_function br_gemm_kernel = libxs_bsmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function br_gemm_kernel2 = libxs_bsmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxs_bmmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic_bf16.tpl.c"

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
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16(libxs_dnn_layer* handle, int start_thread, int tid)
{
  return libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid );
}
#endif

#if defined(LIBXS_INTRINSICS_AVX512_CPX)
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_amx(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

#define LIBXS_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  typedef libxs_bsmmfunction gemm_function;
  typedef libxs_bsmmfunction_reducebatch_offs gemm_br_function_offs;
  typedef libxs_bsmmfunction_reducebatch_strd gemm_br_function_strd;
  gemm_br_function_offs br_gemm_kernel_offs = handle->fwd_compute_kernel_offs;
  gemm_br_function_strd br_gemm_kernel_strd = handle->fwd_compute_kernel_strd;
  gemm_function tile_config_kernel = handle->fwd_config_kernel;
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic_bf16_amx.tpl.c"

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
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_amx(libxs_dnn_layer* handle, int start_thread, int tid)
{
  return libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu_amx( handle, start_thread, tid );
}
#endif

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_i8_i32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef unsigned char element_input_type;
  typedef int element_output_type;
  typedef char element_filter_type;
  /* Basically we need only offset based and strided BRGEMMs */
  libxs_subimmfunction_reducebatch_strd br_gemm_kernel_strided = handle->gemm_fwd.xgemm.subimrs;
  libxs_subimmfunction_reducebatch_strd br_gemm_kernel_strided2 = handle->gemm_fwd2.xgemm.subimrs;
  libxs_subimmfunction_reducebatch_offs br_gemm_kernel_offset = handle->gemm_fwd.xgemm.subimro;
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic_i8i32.tpl.c"
#else
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom_i8_i8(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef unsigned char element_input_type;
  typedef unsigned char element_output_type;
  typedef char element_filter_type;
  /* Basically we need only offset based and strided BRGEMMs */
  libxs_sububmmfunction_reducebatch_strd br_gemm_kernel_strided = handle->gemm_fwd.xgemm.sububmrs;
  libxs_sububmmfunction_reducebatch_offs br_gemm_kernel_offset = handle->gemm_fwd.xgemm.sububmro;
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic_i8i8.tpl.c"
#else
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_nhwc_custom_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->blocksifm*handle->ifmblock : (libxs_blasint)handle->blocksifm*handle->desc.v*handle->ifmblock;
  const libxs_blasint ldA = handle->ofmblock;
  const libxs_blasint ldC = handle->blocksofm*handle->ofmblock;
  const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
  int l_flags = ( LIBXS_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
  /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
  gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxs_dnn_convolve_st_fwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_convolve_st_fwd_nhwc_rsck_f32_f32(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->blocksifm*handle->ifmblock : (libxs_blasint)handle->blocksifm*handle->desc.v*handle->ifmblock;
  const libxs_blasint ldA = handle->blocksofm*handle->ofmblock;
  const libxs_blasint ldC = handle->blocksofm*handle->ofmblock;
  const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
  int l_flags = ( LIBXS_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
  /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
  gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
  gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxs_dnn_convolve_st_fwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( handle->target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_f32_f32( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_I8 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_I32 ) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_i8_i32( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_I8 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_I8 ) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_i8_i8( handle, start_thread, tid);
    }
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_CORE && handle->target_archid < LIBXS_X86_AVX512_CPX) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_CPX && handle->target_archid < LIBXS_X86_AVX512_SPR) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_SPR) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_amx( handle, start_thread, tid);
    }
#elif defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_CORE && handle->target_archid < LIBXS_X86_AVX512_SPR) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXS_X86_AVX512_SPR) {
      status = libxs_dnn_convolve_st_fwd_custom_custom_bf16_bf16_emu_amx( handle, start_thread, tid);
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
      const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->ifmblock : (libxs_blasint)handle->desc.v*handle->ifmblock;
      const libxs_blasint ldA = handle->ofmblock;
      const libxs_blasint ldC = handle->ofmblock;
      const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
      int l_flags = ( LIBXS_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_generic.tpl.c"
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_nhwc_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( handle->target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_convolve_st_fwd_nhwc_custom_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->blocksifm*handle->ifmblock : (libxs_blasint)handle->blocksifm*handle->desc.v*handle->ifmblock;
      const libxs_blasint ldA = handle->ofmblock;
      const libxs_blasint ldC = handle->blocksofm*handle->ofmblock;
      const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
      int l_flags = ( LIBXS_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxs_dnn_convolve_st_fwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_nhwc_rsck(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( handle->target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_convolve_st_fwd_nhwc_rsck_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      const libxs_blasint ldx = (handle->pack_input == 1) ? (libxs_blasint)handle->blocksifm*handle->ifmblock : (libxs_blasint)handle->blocksifm*handle->desc.v*handle->ifmblock;
      const libxs_blasint ldA = handle->blocksofm*handle->ofmblock;
      const libxs_blasint ldC = handle->blocksofm*handle->ofmblock;
      const float beta = (handle->avoid_acc_load) ? 0.f : 1.f;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_smmfunction_reducebatch_addr gemm_br_function;
      int l_flags = ( LIBXS_GEMM_FLAGS('N', 'N') ) | handle->fwd_flags;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_br_function br_gemm_kernel = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*handle->fwd_ofw_rb, handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function br_gemm_kernel2 = libxs_smmdispatch_reducebatch_addr(handle->ofmblock, handle->fwd_ofh_rb*(handle->fwd_ofw_rb-1), handle->ifmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxs_dnn_convolve_st_fwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

