/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_rnncell_backward_weight_update.h"
#include "libxs_dnn_elementwise.h"
#include "libxs_main.h"


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck_f32_f32(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck_f32_f32(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid);


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck_f32_f32(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
#define LIBXS_RNN_CELL_AVX512
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
# define LIBXS_DNN_RNN_RELU_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
# undef LIBXS_DNN_RNN_RELU_BWDUPD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXS_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
# undef LIBXS_DNN_RNN_SIGMOID_BWDUPD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
# define LIBXS_DNN_RNN_TANH_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
# undef LIBXS_DNN_RNN_TANH_BWDUPD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic.tpl.c"
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_bwdupd_nc_ck_generic.tpl.c"
  } else {
    /* should not happen */
  }
#undef LIBXS_RNN_CELL_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
#define LIBXS_RNN_CELL_AVX512
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXS_RNN_CELL_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXS_RNN_CELL_AVX512
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

#define LIBXS_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXS_DNN_BF16_USE_CPX_AVX512_NI
#undef LIBXS_RNN_CELL_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CORE)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
#define LIBXS_RNN_CELL_AVX512
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_kcck_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"

#undef LIBXS_RNN_CELL_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512_CPX)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXS_RNN_CELL_AVX512
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;
  typedef libxs_bfloat16 element_filter_type;

#define LIBXS_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxs_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_kcck_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
    status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxs_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXS_DNN_BF16_USE_CPX_AVX512_NI

#undef LIBXS_RNN_CELL_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck_f32_f32(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
#define LIBXS_RNN_CELL_AVX512
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
# define LIBXS_DNN_RNN_RELU_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_RELU_BWDUPD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXS_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_SIGMOID_BWDUPD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
# define LIBXS_DNN_RNN_TANH_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
# undef LIBXS_DNN_RNN_TANH_BWDUPD
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_kcck.tpl.c"
  } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_bwdupd_nc_kcck.tpl.c"
  } else {
    /* should not happen */
  }
#undef LIBXS_RNN_CELL_AVX512
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
#if 0
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_ncnc_kcck_generic.tpl.c"
#endif
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid); LIBXS_UNUSED(kind);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_ck(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
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
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_rnncell_st_bwdupd_nc_ck_f32_f32( handle, kind, start_thread, tid );
    }
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16) {
      if ( handle->desc.N % 2 != 0 ) {
        status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
      } else {
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
        if ( libxs_target_archid >= LIBXS_X86_AVX512_CORE && libxs_target_archid < LIBXS_X86_AVX512_CPX ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu( handle, kind, start_thread, tid );
        } else if ( libxs_target_archid >= LIBXS_X86_AVX512_CPX ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16( handle, kind, start_thread, tid );
        }
#else
        if ( libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu( handle, kind, start_thread, tid );
        }
#endif
        else {
          status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          return status;
        }
      }
    }
#endif
    else  {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
#define LIBXS_DNN_RNN_RELU_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
#undef LIBXS_DNN_RNN_RELU_BWDUPD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXS_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
#undef LIBXS_DNN_RNN_SIGMOID_BWDUPD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
#define LIBXS_DNN_RNN_TANH_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
#undef LIBXS_DNN_RNN_TANH_BWDUPD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic.tpl.c"
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_bwdupd_nc_ck_generic.tpl.c"
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

LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_nc_kcck(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
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
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_rnncell_st_bwdupd_nc_kcck_f32_f32( handle, kind, start_thread, tid );
    }
#if defined(LIBXS_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      if ( handle->desc.N % 2 != 0 ) {
        status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
      } else {
#if defined(LIBXS_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
        if ( libxs_target_archid >= LIBXS_X86_AVX512_CORE && libxs_target_archid < LIBXS_X86_AVX512_CPX ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid );
        } else if ( libxs_target_archid >= LIBXS_X86_AVX512_CPX ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16( handle, kind, start_thread, tid );
        }
#else
        if ( libxs_target_archid >= LIBXS_X86_AVX512_CORE ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid );
        }
#endif
        else {
          status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          return status;
        }
      }
    }
#endif
    else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_RELU ) {
#define LIBXS_DNN_RNN_RELU_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_RELU_BWDUPD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXS_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_SIGMOID_BWDUPD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_RNN_TANH ) {
#define LIBXS_DNN_RNN_TANH_BWDUPD
# include "template/libxs_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
#undef LIBXS_DNN_RNN_TANH_BWDUPD
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_LSTM ) {
# include "template/libxs_dnn_rnncell_st_lstm_bwdupd_nc_kcck.tpl.c"
      } else if ( handle->desc.cell_type == LIBXS_DNN_RNNCELL_GRU ) {
# include "template/libxs_dnn_rnncell_st_gru_bwdupd_nc_kcck.tpl.c"
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


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_rnncell_st_bwdupd_ncnc_kcck(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind, int start_thread, int tid)
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
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxs_target_archid >= LIBXS_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      status = libxs_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32( handle, kind, start_thread, tid );
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      LIBXS_UNUSED(kind); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
      status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

