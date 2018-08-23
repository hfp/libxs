/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
#include "libxs_dnn_fusedbatchnorm_forward.h"
#include <libxs_intrinsics_x86.h>
#include "libxs_main.h"
#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <math.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbn_st_fwd_custom_f32_f32(libxs_dnn_fusedbn* handle, int start_thread, int tid);
/*LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbn_st_fwd_custom_bf16_bf16(libxs_dnn_fusedbn* handle, int start_thread, int tid);*/

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbn_st_fwd_custom_f32_f32(libxs_dnn_fusedbn* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
    status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_ORDER;
  } else {
    if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE ) {
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE_RELU ) {
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
    } else if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
    } else {
      status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


#if 0
LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_fusedbn_st_fwd_custom_bf16_bf16(libxs_dnn_fusedbn* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  /* @TODO */
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_st_fwd_custom(libxs_dnn_fusedbn* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 ||
      handle->reg_beta == 0  || handle->reg_gamma == 0  ||
      handle->expvalue == 0  || handle->stddev == 0        ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
  if ( libxs_target_archid == LIBXS_X86_AVX512      || libxs_target_archid == LIBXS_X86_AVX512_MIC ||
       libxs_target_archid == LIBXS_X86_AVX512_CORE || libxs_target_archid == LIBXS_X86_AVX512_ICL    ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_fusedbn_st_fwd_custom_f32_f32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      /*status = libxs_dnn_fusedbn_st_fwd_custom_bf16_bf16( handle, start_thread, tid);*/
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_stats_type;

      if ( handle->desc.fuse_order != LIBXS_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU ) {
        status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_ORDER;
      } else {
        if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE ) {
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
        } else if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE ) {
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
        } else if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE_RELU ) {
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
        } else if ( handle->desc.fuse_ops == LIBXS_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU ) {
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
# define LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
# include "template/libxs_dnn_fusedbatchnorm_st_fwd_custom.tpl.c"
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_RELU
# undef LIBXS_DNN_FUSEDBN_FWD_ENABLE_ELTWISE
        } else {
         status = LIBXS_DNN_ERR_FUSEBN_UNSUPPORTED_FUSION;
        }
      }
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      /* TODO */
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_fusedbn_st_fwd_nhwc(libxs_dnn_fusedbn* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );
  return status;
}

