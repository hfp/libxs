/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#include "libxs_dnn_convolution_winograd_forward.h"
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


LIBXS_INLINE LIBXS_RETARGETABLE void internal_fwd_input_transform_custom_custom(
                                           const float *inp,
                                           float *tinp,
                                           float *Iwp,
                                           const libxs_dnn_layer* handle )
{
  if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_input_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_input_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_input_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_input_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
    assert(0);
  }
#endif
}

LIBXS_INLINE LIBXS_RETARGETABLE void internal_fwd_input_transform_nhwc_custom(
                                         const float *inp,
                                         float *tinp,
                                         float *Iwp,
                                         const libxs_dnn_layer* handle )
{
  if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_input_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_input_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_input_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_input_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
    assert(0);
  }
#endif
}

LIBXS_INLINE LIBXS_RETARGETABLE void internal_fwd_weight_transform( float *wp,
                              float *twp,
                              const libxs_dnn_layer* handle )
{
  if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_weight_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_weight_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_weight_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_weight_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
    assert(0);
  }
#endif
}

LIBXS_INLINE LIBXS_RETARGETABLE void internal_fwd_output_transform_custom_custom( float *toutp,
                                            float *outp,
                                            float *Owp,
                                            const int vratio,
                                            float bias[/*vratio*/][16/*tdvlen*/],
                                            const libxs_dnn_layer* handle )
{
  LIBXS_UNUSED(bias); /* TODO: remove */
  if (vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_output_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_output_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_output_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_output_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
    assert(0);
  }
#endif
}

LIBXS_INLINE LIBXS_RETARGETABLE void internal_fwd_output_transform_nhwc_custom( float *toutp,
                                          float *outp,
                                          float *Owp,
                                          const int vratio,
                                          float bias[/*vratio*/][16/*tdvlen*/],
                                          const libxs_dnn_layer* handle )
{
  LIBXS_UNUSED(bias); /* TODO: remove */
  if (vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_output_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_output_trans_alpha6.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_output_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  } else if (vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_output_trans_alpha4.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
    assert(0);
  }
#endif
}

LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_convolve_winograd_st_fwd_custom_custom( libxs_dnn_layer* handle, int start_thread, int tid )
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxs_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
#if 0
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_sconvfunction libxs_convfunction;
# include "template/libxs_dnn_convolve_winograd_st_fwd_custom_custom.tpl.c"
#endif

      if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_custom_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
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

LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_convolve_winograd_st_fwd_nhwc_custom( libxs_dnn_layer* handle, int start_thread, int tid )
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxs_dnn_convolve_st_fwd_nhwc_custom_fallback.tpl.c"
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
#if 0
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxs_sconvfunction libxs_convfunction;
# include "template/libxs_dnn_convolve_winograd_st_fwd_nhwc_custom.tpl.c"
#endif

      if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_fwd.vratio == 1 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 16
#define VRATIO 1
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_fwd.vratio == 2 && handle->cwino_fwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
#define FDVLEN 32
#define VRATIO 2
# include "template/libxs_dnn_convolution_winograd_forward_nhwc_custom_inlined.tpl.c"
#undef VRATIO
#undef FDVLEN
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXS error: Unsupported fdvlen %u or unsupported alpha %u\n", handle->cwino_fwd.vratio*16, handle->cwino_fwd.alpha);
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
