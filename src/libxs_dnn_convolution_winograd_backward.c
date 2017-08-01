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
#include "libxs_dnn_convolution_winograd_backward.h"
#include "libxs_main.h"
#include <libxs_intrinsics_x86.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <assert.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* Enable/disable specific code paths */
#if defined(LIBXS_INTRINSICS_AVX512) && !defined(LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512
#endif


/* function pointer for the CPUID-dispatched implementation */
LIBXS_API_VARIABLE void (*internal_bwd_input_transform_custom_custom_alpha6)(const float*, float*, float*, const libxs_dnn_layer*);
LIBXS_API_VARIABLE void (*internal_bwd_input_transform_nhwc_custom_alpha6)(const float*, float*, float*, const libxs_dnn_layer*);
LIBXS_API_VARIABLE void (*internal_bwd_output_transform_custom_custom_alpha6)(float*, float*, float*, const libxs_dnn_layer*);
LIBXS_API_VARIABLE void (*internal_bwd_output_transform_nhwc_custom_alpha6)(float*, float*, float*, const libxs_dnn_layer*);
LIBXS_API_VARIABLE void (*internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6)(libxs_dnn_layer*, int, int);


LIBXS_API_INLINE void internal_bwd_input_transform_custom_custom_alpha6_default(
  const float* inp, float* tinp, float* Iwp, const libxs_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxs_dnn_convolution_winograd_backward_custom_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
LIBXS_ATTRIBUTE_UNUSED void internal_bwd_input_transform_custom_custom_alpha6_avx512(
  const float* inp, float* tinp, float* Iwp, const libxs_dnn_layer* handle)
{
#if defined(LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_custom_custom_input_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
  LIBXS_UNUSED(Iwp);
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_input_transform_custom_custom_alpha6_default(inp, tinp, Iwp, handle);
#endif
}

LIBXS_API_INLINE /*LIBXS_INTRINSICS(LIBXS_X86_AVX512)*/
void internal_bwd_input_transform_custom_custom(
  const float *inp, float *tinp, float *Iwp,
  const libxs_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
    internal_bwd_input_transform_custom_custom_alpha6_avx512(inp, tinp, Iwp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_input_transform_custom_custom_alpha6);
    internal_bwd_input_transform_custom_custom_alpha6(inp, tinp, Iwp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_custom_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXS_API_INLINE void internal_bwd_input_transform_nhwc_custom_alpha6_default(
  const float* inp, float* tinp, float* Iwp, const libxs_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_input_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
LIBXS_ATTRIBUTE_UNUSED void internal_bwd_input_transform_nhwc_custom_alpha6_avx512(
  const float* inp, float* tinp, float* Iwp, const libxs_dnn_layer* handle)
{
#if defined(LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_input_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
  LIBXS_UNUSED(Iwp);
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_input_transform_nhwc_custom_alpha6_default(inp, tinp, Iwp, handle);
#endif
}


LIBXS_API_INLINE void internal_bwd_input_transform_nhwc_custom(
  const float* inp, float* tinp, float* Iwp, const libxs_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
    internal_bwd_input_transform_nhwc_custom_alpha6_avx512(inp, tinp, Iwp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_input_transform_nhwc_custom_alpha6);
    internal_bwd_input_transform_nhwc_custom_alpha6(inp, tinp, Iwp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_input_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXS_API_INLINE void internal_bwd_weight_transform(float *wp, float *twp, const libxs_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_weight_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_weight_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXS_API_INLINE void internal_bwd_output_transform_custom_custom_alpha6_default(
  float *toutp, float *outp, float *Owp, const libxs_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxs_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
LIBXS_ATTRIBUTE_UNUSED void internal_bwd_output_transform_custom_custom_alpha6_avx512(
  float *toutp, float *outp, float *Owp, const libxs_dnn_layer* handle)
{
#if defined(LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_output_transform_custom_custom_alpha6_default(toutp, outp, Owp, handle);
#endif
}


LIBXS_API_INLINE void internal_bwd_output_transform_custom_custom(
  float *toutp, float *outp, float *Owp, const libxs_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
    internal_bwd_output_transform_custom_custom_alpha6_avx512(toutp, outp, Owp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_output_transform_custom_custom_alpha6);
    internal_bwd_output_transform_custom_custom_alpha6(toutp, outp, Owp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_custom_custom_output_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXS_API_INLINE void internal_bwd_output_transform_nhwc_custom_alpha6_default(
  float *toutp, float *outp, float *Owp, const libxs_dnn_layer* handle)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha6.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
LIBXS_ATTRIBUTE_UNUSED void internal_bwd_output_transform_nhwc_custom_alpha6_avx512(
  float *toutp, float *outp, float *Owp, const libxs_dnn_layer* handle)
{
#if defined(LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha6_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
#else /* next lower/available code path (fall-back chain) */
  internal_bwd_output_transform_nhwc_custom_alpha6_default(toutp, outp, Owp, handle);
#endif
}


LIBXS_API_INLINE void internal_bwd_output_transform_nhwc_custom(
  float *toutp, float *outp, float *Owp, const libxs_dnn_layer* handle)
{
  if (handle->cwino_bwd.alpha == 6) {
    /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
    internal_bwd_output_transform_nhwc_custom_alpha6_avx512(toutp, outp, Owp, handle);
#else /* pointer based function call */
    assert(0 != internal_bwd_output_transform_nhwc_custom_alpha6);
    internal_bwd_output_transform_nhwc_custom_alpha6(toutp, outp, Owp, handle);
#endif
  } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_output_trans_alpha4.tpl.c"
#undef TDVLEN
#undef ALPHA
  }
#if !defined(NDEBUG)
  else {
    fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
    assert(0);
  }
#endif
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_convolve_winograd_st_bwd_custom_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxs_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
      if (handle->flag_reuseInput == 1) {
        if (handle->scratchInput == 0) {
          status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
        handle->scratch3 = handle->scratchInput;
        }
      }
      if (handle->cwino_bwd.alpha == 6) {
#define ALPHA 6
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_custom_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
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


LIBXS_API_INLINE void internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_default(
  libxs_dnn_layer* handle, int start_thread, int tid)
{
#define ALPHA 6
#define TDVLEN 16
#include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
}


LIBXS_API_INLINE LIBXS_INTRINSICS(LIBXS_X86_AVX512)
LIBXS_ATTRIBUTE_UNUSED void internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_avx512(
  libxs_dnn_layer* handle, int start_thread, int tid)
{
#if defined(LIBXS_DNN_CONVOLUTION_WINOGRAD_BACKWARD_AVX512)
# define ALPHA 6
# define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_inlined_avx512.tpl.c"
# undef TDVLEN
# undef ALPHA
  LIBXS_UNUSED(Iwp);
#else /* next lower/available code path (fall-back chain) */
  internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_default(handle, start_thread, tid);
#endif
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_convolve_winograd_st_bwd_nhwc_custom(libxs_dnn_layer* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 || handle->scratch3 == 0 || handle->scratch4 == 0 || handle->scratchIw == 0 || handle->scratchOw == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxs_dnn_convolve_st_bwd_nhwc_custom_fallback.tpl.c"
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXS_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXS_DNN_DATATYPE_F32) {
      if (handle->flag_reuseInput == 1) {
        if (handle->scratchInput == 0) {
          status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
          return status;
        } else {
        handle->scratch3 = handle->scratchInput;
        }
      }
      if (handle->cwino_bwd.alpha == 6) {
        /* if highest implemented code path is statically present, no need for an indirect call (function pointer) */
#if (LIBXS_X86_AVX512 <= LIBXS_STATIC_TARGET_ARCH)
        internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_avx512(toutp, outp, Owp, handle);
#else /* pointer based function call */
        assert(0 != internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6);
        internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6(handle, start_thread, tid);
#endif
      } else if (handle->cwino_bwd.alpha == 4) {
#define ALPHA 4
#define TDVLEN 16
# include "template/libxs_dnn_convolution_winograd_backward_nhwc_custom_inlined.tpl.c"
#undef TDVLEN
#undef ALPHA
      }
#if !defined(NDEBUG)
      else {
        fprintf(stderr, "LIBXS error: Unsupported alpha %u\n", handle->cwino_bwd.alpha);
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


LIBXS_API_DEFINITION void libxs_dnn_convolve_winograd_bwd_init(int target_arch)
{
  if (LIBXS_X86_AVX512 <= target_arch) {
    internal_bwd_input_transform_custom_custom_alpha6 = internal_bwd_input_transform_custom_custom_alpha6_avx512;
    internal_bwd_input_transform_nhwc_custom_alpha6 = internal_bwd_input_transform_nhwc_custom_alpha6_avx512;
    internal_bwd_output_transform_custom_custom_alpha6 = internal_bwd_output_transform_custom_custom_alpha6_avx512;
    internal_bwd_output_transform_nhwc_custom_alpha6 = internal_bwd_output_transform_nhwc_custom_alpha6_avx512;
    internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6 = internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_avx512;
  }
  else {
    internal_bwd_input_transform_custom_custom_alpha6 = internal_bwd_input_transform_custom_custom_alpha6_default;
    internal_bwd_input_transform_nhwc_custom_alpha6 = internal_bwd_input_transform_nhwc_custom_alpha6_default;
    internal_bwd_output_transform_custom_custom_alpha6 = internal_bwd_output_transform_custom_custom_alpha6_default;
    internal_bwd_output_transform_nhwc_custom_alpha6 = internal_bwd_output_transform_nhwc_custom_alpha6_default;
    internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6 = internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6_default;
  }
  assert(0 != internal_bwd_input_transform_custom_custom_alpha6);
  assert(0 != internal_bwd_input_transform_nhwc_custom_alpha6);
  assert(0 != internal_bwd_output_transform_custom_custom_alpha6);
  assert(0 != internal_bwd_output_transform_nhwc_custom_alpha6);
  assert(0 != internal_dnn_convolve_winograd_st_bwd_nhwc_custom_alpha6);
}


LIBXS_API_DEFINITION void libxs_dnn_convolve_winograd_bwd_finalize(void)
{
}

