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
#include "libxs_dnn_rnncell_forward.h"
#include "libxs_dnn_elementwise.h"
#include "libxs_main.h"
#include <libxs_intrinsics_x86.h>
#include <libxs.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_ck_f32_f32(libxs_dnn_rnncell* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_ck_bf16_bf16(libxs_dnn_rnncell* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32(libxs_dnn_rnncell* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_kcck_f32_f32(libxs_dnn_rnncell* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16(libxs_dnn_rnncell* handle, int start_thread, int tid);

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_ck_f32_f32(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
# define LIBXS_DNN_RNN_RELU_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
# undef LIBXS_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXS_DNN_RNN_SIGMOID_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
# undef LIBXS_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
# define LIBXS_DNN_RNN_TANH_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
# undef LIBXS_DNN_RNN_TANH_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
#define LIBXS_RNN_CELL_AVX512
# include "template/libxs_dnn_rnncell_st_lstm_fwd_nc_ck_generic.tpl.c"
#undef LIBXS_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_fwd_nc_ck_generic.tpl.c"
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_ck_bf16_bf16(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__, __AVX512BW__, __AVX512DQ__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
#define LIBXS_RNN_CELL_AVX512
# include "template/libxs_dnn_rnncell_st_lstm_fwd_nc_ck_generic_bf16.tpl.c"
#undef LIBXS_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
# define LIBXS_DNN_RNN_RELU_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXS_DNN_RNN_SIGMOID_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
# define LIBXS_DNN_RNN_TANH_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_TANH_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_kcck_f32_f32(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
# define LIBXS_DNN_RNN_RELU_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXS_DNN_RNN_SIGMOID_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
# define LIBXS_DNN_RNN_TANH_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_TANH_FWD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
#define LIBXS_RNN_CELL_AVX512
# include "template/libxs_dnn_rnncell_st_lstm_fwd_nc_kcck.tpl.c"
#undef LIBXS_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_fwd_nc_kcck.tpl.c"
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
#define LIBXS_RNN_CELL_AVX512
# include "template/libxs_dnn_rnncell_st_lstm_fwd_nc_kcck_bf16.tpl.c"
#undef LIBXS_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_ck(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_rnncell_st_fwd_nc_ck_f32_f32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      status = libxs_dnn_rnncell_st_fwd_nc_ck_bf16_bf16( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
#define LIBXS_DNN_RNN_RELU_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXS_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXS_DNN_RNN_SIGMOID_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXS_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
#define LIBXS_DNN_RNN_TANH_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXS_DNN_RNN_TANH_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_fwd_nc_ck_generic.tpl.c"
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_fwd_nc_ck_generic.tpl.c"
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_ncnc_kcck(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
#define LIBXS_DNN_RNN_RELU_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXS_DNN_RNN_SIGMOID_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
#define LIBXS_DNN_RNN_TANH_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_TANH_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
        status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
        status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_fwd_nc_kcck(libxs_dnn_rnncell* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_rnncell_st_fwd_nc_kcck_f32_f32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      status = libxs_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
#define LIBXS_DNN_RNN_RELU_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXS_DNN_RNN_SIGMOID_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
#define LIBXS_DNN_RNN_TANH_FWD
# include "template/libxs_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_TANH_FWD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_fwd_nc_kcck.tpl.c"
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_fwd_nc_kcck.tpl.c"
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}
