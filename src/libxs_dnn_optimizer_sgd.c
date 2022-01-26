/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_optimizer_sgd.h"
#include "libxs_main.h"


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_optimizer_sgd_st_f32_f32(libxs_dnn_optimizer* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_optimizer_sgd_st_bf16_bf16(libxs_dnn_optimizer* handle, int start_thread, int tid);


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_optimizer_sgd_st_f32_f32(libxs_dnn_optimizer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_filter_type;

# define LIBXS_DNN_OPTIMIZER_SGD_F32_AVX512
# include "template/libxs_dnn_optimizer_sgd_st_generic.tpl.c"
# undef LIBXS_DNN_OPTIMIZER_SGD_F32_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_optimizer_sgd_st_bf16_bf16(libxs_dnn_optimizer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_filter_type;
  typedef float            element_master_type;

# define LIBXS_DNN_OPTIMIZER_SGD_BF16_AVX512
# include "template/libxs_dnn_optimizer_sgd_st_generic.tpl.c"
# undef LIBXS_DNN_OPTIMIZER_SGD_BF16_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_optimizer_sgd_st(libxs_dnn_optimizer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have filter, grad_filter */
  if ( handle->reg_filter == 0 || handle->grad_filter == 0 ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( (handle->master_filter == 0) && (handle->desc.datatype == LIBXS_DNN_DATATYPE_BF16) ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_optimizer_sgd_st_f32_f32( handle, start_thread, tid);
    } else if ( handle->desc.datatype == LIBXS_DNN_DATATYPE_BF16 ) {
      status = libxs_dnn_optimizer_sgd_st_bf16_bf16( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if ( handle->desc.datatype == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_filter_type;

# include "template/libxs_dnn_optimizer_sgd_st_generic.tpl.c"
    } else if ( handle->desc.datatype == LIBXS_DNN_DATATYPE_BF16 ) {
      typedef libxs_bfloat16 element_filter_type;
      typedef float            element_master_type;

# define LIBXS_DNN_OPTIMIZER_SGD_BF16
# include "template/libxs_dnn_optimizer_sgd_st_generic.tpl.c"
# undef LIBXS_DNN_OPTIMIZER_SGD_BF16
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

