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
/* Kunal Banerjee (Intel Corp.), Rajkishore Barik (Intel Corp.),
 * Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxs.h>
#include "libxs_dnn_convolution_winograd_weight_update.h"
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <assert.h>
# include <stdio.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_INLINE void internal_upd_input_transform_custom_custom(
                                           float *inp,
                                           float *tinp,
                                           float *Iwp,
                                           const libxs_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXS_API_INLINE void internal_upd_input_transform_nhwc_custom(
                                         float *inp,
                                         float *tinp,
                                         float *Iwp,
                                         const libxs_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXS_API_INLINE void internal_upd_deloutput_transform_custom_custom(
                                               float *inp,
                                               float *tinp,
                                               float *Owp,
                                               const libxs_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_deloutput_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_deloutput_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXS_API_INLINE void internal_upd_deloutput_transform_nhwc_custom(
                                             float *inp,
                                             float *tinp,
                                             float *Owp,
                                             const libxs_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_deloutput_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_deloutput_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXS_API_INLINE void internal_upd_delweight_transform(
                                 float *wp,
                                 float *twp,
                                 const libxs_dnn_layer* handle )
{
  if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_delweight_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_delweight_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
    assert(0);
  }
#endif
}

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_winograd_st_upd_custom_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32) {
      const libxs_blasint ldx = (libxs_blasint)handle->desc.W+((libxs_blasint)2*handle->desc.pad_w);
      const libxs_blasint ldx_alt = (libxs_blasint)handle->desc.v*handle->ifmblock;
      const libxs_blasint ldb_alt = (libxs_blasint)handle->ofwp;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel = libxs_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, NULL, &ldx, NULL, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxs_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &ldx_alt, &ldb_alt, NULL, NULL, NULL, NULL, NULL);
# include "template/libxs_dnn_convolve_st_upd_custom_custom_generic.tpl.c"
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32) {
      if (handle->cwino_upd.alpha == 6  && libxs_target_archid == LIBXS_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 4  && libxs_target_archid == LIBXS_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
        assert(0);
      }
#endif
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_convolve_winograd_st_upd_nhwc_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32) {
      const libxs_blasint lda     = (libxs_blasint)handle->blocksofm*handle->ofmblock;
      const libxs_blasint ldb     = (libxs_blasint)handle->desc.W+((libxs_blasint)2*handle->desc.pad_w);
      const libxs_blasint ldc     = (libxs_blasint)handle->ofmblock;
      const libxs_blasint lda_alt = (libxs_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                            ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      const libxs_blasint ldb_alt = (libxs_blasint)handle->ofwp;
      const libxs_blasint ldc_alt = (libxs_blasint)handle->ifmblock;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel     = libxs_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxs_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &lda_alt, &ldb_alt, &ldc_alt, NULL, NULL, NULL, NULL);
#define LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxs_dnn_convolve_st_upd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXS_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32) {
      if (handle->cwino_upd.alpha == 6 && libxs_target_archid == LIBXS_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 4 && libxs_target_archid == LIBXS_X86_AVX512_KNM && (handle->cwino_upd.itiles*handle->cwino_upd.jtiles*handle->cwino_upd.bimg % 4) == 0) {
        if (handle->scratchVk == 0) {
          status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_inlined_knm.tpl.c"
#undef TDVLEN
#undef ALPHA
        }
      } else if (handle->cwino_upd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_upd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_weight_update_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_upd.alpha);
        assert(0);
      }
#endif
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}
