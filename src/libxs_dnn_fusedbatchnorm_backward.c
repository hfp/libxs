/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_fusedbatchnorm_backward.h"
#include "libxs_main.h"


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c16(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c32(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c64(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c16(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c32(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c64(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid);


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c16(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
         (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c32(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
         (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c64(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
         (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE  ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c16(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXS_DNN_FUSEDBN_BWD_BF16
  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
         (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXS_DNN_FUSEDBN_BWD_BF16
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c32(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXS_DNN_FUSEDBN_BWD_BF16
  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
         (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXS_DNN_FUSEDBN_BWD_BF16
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c64(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXS_DNN_FUSEDBN_BWD_BF16
  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
         (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXS_DNN_FUSEDBN_BWD_BF16
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_custom(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if ( handle->reg_input == 0  || handle->reg_gamma == 0   ||
       handle->grad_input == 0 || handle->grad_output == 0 ||
       handle->grad_beta == 0  || handle->grad_gamma == 0  ||
       handle->expvalue == 0   || handle->rcpstddev == 0      ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_BN) > 0 ) {
    if ( handle->scratch == 0 ) {
      status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }
  if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) > 0 ) {
    if ( handle->grad_add == 0 ) {
      status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }
  if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) > 0 ) {
    if ( handle->reg_output == 0 ) {
      status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }
  if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) > 0 ) {
    if ( handle->relumask == 0 ) {
      status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( ( libxs_target_archid >= LIBXS_X86_AVX512 ) &&
       (handle->ofmblock == 16) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c16( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      status = libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c16( handle, start_thread, tid );
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxs_target_archid >= LIBXS_X86_AVX512 ) &&
       (handle->ofmblock == 32) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c32( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      status = libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c32( handle, start_thread, tid );
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxs_target_archid >= LIBXS_X86_AVX512 ) &&
       (handle->ofmblock == 64) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fusedbatchnorm_st_bwd_custom_f32_f32_c64( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      status = libxs_dnn_fusedbatchnorm_st_bwd_custom_bf16_bf16_c64( handle, start_thread, tid );
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_stats_type;

      if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
        status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
      } else {
        if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
             (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
        } else {
          status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
        }
      }
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      typedef libxs_bfloat16 element_input_type;
      typedef libxs_bfloat16 element_output_type;
      typedef float element_stats_type;

# define LIBXS_DNN_FUSEDBN_BWD_BF16
      if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
        status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER;
      } else {
        if ( (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BN)            ||
             (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS) || (handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSTATS_NORED)    ) {
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_ELTWISE) == LIBXS_DNN_FUSEDBN_OPS_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU) == LIBXS_DNN_FUSEDBN_OPS_RELU ) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU
        } else if ( (handle->desc.fuse_ops & LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) ==  LIBXS_DNN_FUSEDBN_OPS_RELU_WITH_MASK) {
# define LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxs_dnn_fusedbatchnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXS_DNN_FUSEDBN_BWD_ENABLE_RELU_WITH_MASK
        } else {
          status = LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION;
        }
      }
# undef LIBXS_DNN_FUSEDBN_BWD_BF16
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_st_bwd_nhwc(libxs_dnn_fusedbatchnorm* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbatchnorm_reduce_stats_st_bwd_custom(libxs_dnn_fusedbatchnorm** handles, int num_handles, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int l_count;

  /* check if all required tensors are bound */
  for ( l_count = 0; l_count < num_handles; ++l_count ) {
    if ( handles[l_count]->grad_beta == 0  || handles[l_count]->grad_gamma == 0 || handles[l_count]->scratch == 0 ) {
      status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }

#if 0
  /* check if we are on an AVX512 platform */
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    status = libxs_dnn_fusedbatchnorm_reduce_stats_st_bwd_custom_avx512( handles, num_handles, start_thread, tid );
  } else
#endif
  {
    const int nImg = handles[0]->desc.partN;
    const int nBlocksFm = handles[0]->blocksifm;
    const int nFmBlock = handles[0]->ifmblock;
    /* computing first logical thread */
    const int ltid = tid - start_thread;
    /* number of tasks that could be run in parallel */
    const int work2 = nBlocksFm;
    /* compute chunk size */
    const int chunksize2 = (work2 % handles[0]->desc.threads == 0) ? (work2 / handles[0]->desc.threads) : ((work2 / handles[0]->desc.threads) + 1);
    /* compute thr_begin and thr_end */
    const int thr_begin2 = (ltid * chunksize2 < work2) ? (ltid * chunksize2) : work2;
    const int thr_end2 = ((ltid + 1) * chunksize2 < work2) ? ((ltid + 1) * chunksize2) : work2;
    int v = 0, fm;

    LIBXS_VLA_DECL(2, float, dgamma0,      (float*)handles[0]->grad_gamma->data, nFmBlock);
    LIBXS_VLA_DECL(2, float, dbeta0,       (float*)handles[0]->grad_beta->data,  nFmBlock);
    LIBXS_VLA_DECL(3, float, dgamma_img0, (float*)handles[0]->scratch,                                                        nImg, nFmBlock);
    LIBXS_VLA_DECL(3, float, dbeta_img0,  ((float*)handles[0]->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)nFmBlock), nImg, nFmBlock);

    /* lazy barrier init */
    libxs_barrier_init(handles[0]->barrier, ltid);

    for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
      float* dgamma0_ptr = &LIBXS_VLA_ACCESS(2, dgamma0, fm, 0, nFmBlock);
      float* dbeta0_ptr  = &LIBXS_VLA_ACCESS(2, dbeta0,  fm, 0, nFmBlock);
      float* dgamma_img0_ptr = &LIBXS_VLA_ACCESS(3, dgamma_img0, fm, 0, 0, nImg, nFmBlock);
      float* dbeta_img0_ptr  = &LIBXS_VLA_ACCESS(3, dbeta_img0,  fm, 0, 0, nImg, nFmBlock);

      LIBXS_PRAGMA_SIMD
      for ( v=0; v < nFmBlock; v++ ) {
        dgamma0_ptr[v] = dgamma_img0_ptr[v];
        dbeta0_ptr[v]  = dbeta_img0_ptr[v];
      }
    }

    /* now we need to reduce the dgamma and dbeta  */
    for ( l_count = 1; l_count < num_handles; ++l_count ) {
      LIBXS_VLA_DECL(3, float, dgamma_imgr, (float*)handles[l_count]->scratch,                                                        nImg, nFmBlock);
      LIBXS_VLA_DECL(3, float, dbeta_imgr,  ((float*)handles[l_count]->scratch) + ((size_t)nImg * (size_t)nBlocksFm * (size_t)nFmBlock), nImg, nFmBlock);

      for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
        float* dgamma0_ptr = &LIBXS_VLA_ACCESS(2, dgamma0, fm, 0, nFmBlock);
        float* dbeta0_ptr  = &LIBXS_VLA_ACCESS(2, dbeta0,  fm, 0, nFmBlock);
        float* dgamma_imgr_ptr = &LIBXS_VLA_ACCESS(3, dgamma_imgr, fm, 0, 0, nImg, nFmBlock);
        float* dbeta_imgr_ptr  = &LIBXS_VLA_ACCESS(3, dbeta_imgr,  fm, 0, 0, nImg, nFmBlock);

        LIBXS_PRAGMA_SIMD
        for ( v=0; v < nFmBlock; v++ ) {
          dgamma0_ptr[v] += dgamma_imgr_ptr[v];
          dbeta0_ptr[v]  += dbeta_imgr_ptr[v];
        }
      }
    }

    for ( l_count = 1; l_count < num_handles; ++l_count ) {
      LIBXS_VLA_DECL(2, float, dgammar, (float*)handles[l_count]->grad_gamma->data, nFmBlock);
      LIBXS_VLA_DECL(2, float, dbetar,  (float*)handles[l_count]->grad_beta->data,  nFmBlock);

      for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
        float* dgamma0_ptr = &LIBXS_VLA_ACCESS(2, dgamma0, fm, 0, nFmBlock);
        float* dbeta0_ptr  = &LIBXS_VLA_ACCESS(2, dbeta0,  fm, 0, nFmBlock);
        float* dgammar_ptr = &LIBXS_VLA_ACCESS(2, dgammar, fm, 0, nFmBlock);
        float* dbetar_ptr  = &LIBXS_VLA_ACCESS(2, dbetar,  fm, 0, nFmBlock);

        LIBXS_PRAGMA_SIMD
        for ( v=0; v < nFmBlock; v++ ) {
          dgammar_ptr[v] = dgamma0_ptr[v];
          dbetar_ptr[v]  = dbeta0_ptr[v];
        }
      }
    }

    libxs_barrier_wait(handles[0]->barrier, ltid);
  }

  return status;
}

