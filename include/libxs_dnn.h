/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_H
#define LIBXS_DNN_H

#include "libxs_typedefs.h"

typedef unsigned int libxs_dnn_err_t;

/** Define error and warning codes */
#define LIBXS_DNN_SUCCESS                             0

#define LIBXS_DNN_WARN_FALLBACK                   90000
#define LIBXS_DNN_WARN_RNN_SUBOPTIMAL_N_BLOCKING  90001
#define LIBXS_DNN_WARN_RNN_SUBOPTIMAL_C_BLOCKING  90002
#define LIBXS_DNN_WARN_RNN_SUBOPTIMAL_K_BLOCKING  90003
#define LIBXS_DNN_WARN_FC_SUBOPTIMAL_N_BLOCKING   90004
#define LIBXS_DNN_WARN_FC_SUBOPTIMAL_C_BLOCKING   90005
#define LIBXS_DNN_WARN_FC_SUBOPTIMAL_K_BLOCKING   90006

#define LIBXS_DNN_ERR_GENERAL                    100000
#define LIBXS_DNN_ERR_CREATE_HANDLE              100001
#define LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE       100002
#define LIBXS_DNN_ERR_INVALID_BLOCKING           100003
#define LIBXS_DNN_ERR_INVALID_HANDLE             100004
#define LIBXS_DNN_ERR_DATA_NOT_BOUND             100005
#define LIBXS_DNN_ERR_CREATE_TENSOR              100006
#define LIBXS_DNN_ERR_INVALID_TENSOR             100007
#define LIBXS_DNN_ERR_MISMATCH_TENSOR            100008
#define LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR      100009
#define LIBXS_DNN_ERR_INVALID_KIND               100010
#define LIBXS_DNN_ERR_INVALID_FORMAT_NCHW        100011
#define LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT     100012
#define LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT     100013
#define LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE    100014
#define LIBXS_DNN_ERR_INVALID_FORMAT_KCRS        100015
#define LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL     100016
#define LIBXS_DNN_ERR_CREATE_LAYOUT              100017
#define LIBXS_DNN_ERR_INVALID_LAYOUT             100018
#define LIBXS_DNN_ERR_UNSUPPORTED_ARCH           100019
#define LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED        100020
#define LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE        100021
#define LIBXS_DNN_ERR_INVALID_ALGO               100022
#define LIBXS_DNN_ERR_INVALID_PADDING            100023
#define LIBXS_DNN_ERR_UNKNOWN_BIAS_TYPE          100024
#define LIBXS_DNN_ERR_MISMATCH_BIAS              100025
#define LIBXS_DNN_ERR_INVALID_HANDLE_BIAS        100026
#define LIBXS_DNN_ERR_TIME_STEPS_TOO_SMALL       100027
#define LIBXS_DNN_ERR_CREATE_LAYOUT_ARRAYS       100028
#define LIBXS_DNN_ERR_NOT_IMPLEMENTED            100029
#define LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER  100030
#define LIBXS_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION 100031
#define LIBXS_DNN_ERR_INVALID_FORMAT_FUSEDBN     100032
#define LIBXS_DNN_ERR_UNSUPPORTED_POOLING        100033
#define LIBXS_DNN_ERR_INVALID_FORMAT_FC          100034
#define LIBXS_DNN_ERR_INVALID_RNN_TYPE           100035
#define LIBXS_DNN_ERR_RNN_INVALID_SEQ_LEN        100036
#define LIBXS_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER  100037
#define LIBXS_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION 100038
#define LIBXS_DNN_ERR_FC_UNSUPPORTED_FUSION      100039

/** Kinds of supported compute flavor operations. */
typedef enum libxs_dnn_compute_kind {
  /** Forward path */
  LIBXS_DNN_COMPUTE_KIND_FWD,
  /** Backward path */
  LIBXS_DNN_COMPUTE_KIND_BWD,
  /** Updated weights. */
  LIBXS_DNN_COMPUTE_KIND_UPD,
  /** Backward and weightupdate combined, useful for RNNs */
  LIBXS_DNN_COMPUTE_KIND_BWDUPD,
  /** All routines, need for some init routines. */
  LIBXS_DNN_COMPUTE_KIND_ALL
} libxs_dnn_compute_kind;

/** these are some quantization definitions, not sure if we want to
    move them into some main part of LIBXS */
/* @TODO check position of these declarations and defines */
typedef union LIBXS_RETARGETABLE libxs_intfloat {
  unsigned int ui;
  float f;
} libxs_intfloat;

/* F32 masking defines */
#define LIBXSNN_DNN_MASK_SIGN_F32      0x80000000
#define LIBXS_DNN_MASK_EXP_F32       0x7f800000
#define LIBXS_DNN_MASK_MANT_F32      0x007fffff
#define LIBXS_DNN_MASK_ABS_F32       0x7fffffff
#define LIBXS_DNN_MASK_FULL_F32      0xffffffff
#define LIBXS_DNN_MANT_SZ_F32        23
#define LIBXS_DNN_SZ_F32             32

/* DFP16 masking defines */
#define LIBXS_DNN_MANT_DFP16         15
#define LIXSMMM_DNN_RES_DFP16          libxs_sexp2_i8i(-(LIBXS_DNN_MANT_DFP16))

/* Quantization Rounding Defines */
#define LIBXS_DNN_QUANT_NO_ROUND       80000
#define LIBXS_DNN_QUANT_BIAS_ROUND     80001
#define LIBXS_DNN_QUANT_STOCH_ROUND    80002
#define LIBXS_DNN_QUANT_NEAREST_ROUND  80003
#define LIBXS_DNN_QUANT_FPHW_ROUND     80004

/** get string of error code */
LIBXS_API const char* libxs_dnn_get_error(libxs_dnn_err_t code);
LIBXS_API size_t libxs_dnn_typesize(libxs_dnn_datatype datatype);
LIBXS_API size_t libxs_dnn_get_simd_width(libxs_dnn_datatype datatype);

/** some quantization helper functions,
    @TODO need to be integrated better for all different ways of quantizations */
LIBXS_API void libxs_dnn_quantize( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXS_API void libxs_dnn_quantize_act( float* in_buffer, short* out_buffer, unsigned int N, unsigned int C, unsigned int H, unsigned int W, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXS_API void libxs_dnn_quantize_fil( float* in_buffer, short* out_buffer, unsigned int K, unsigned int C, unsigned int R, unsigned int S, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int kblk_f32, unsigned int kblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXS_API void libxs_dnn_dequantize( short* in_buffer, float* out_buffer, int length, unsigned char scf );

/** some BF16<->FP32 conversion functions
    @TODO we need to find a final place for those */
LIBXS_API void libxs_truncate_convert_f32_bf16(const float* in, libxs_bfloat16* out, unsigned int length);
LIBXS_API void libxs_rnaz_convert_fp32_bf16(const float* in, libxs_bfloat16* out, unsigned int len);
LIBXS_API void libxs_rne_convert_fp32_bf16(const float* in, libxs_bfloat16* out, unsigned int len);
LIBXS_API void libxs_convert_bf16_f32(const libxs_bfloat16* in, float* out, unsigned int length);

#endif /*LIBXS_DNN_H*/
