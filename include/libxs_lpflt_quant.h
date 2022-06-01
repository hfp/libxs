/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_LPFLT_QUANT_H
#define LIBXS_LPFLT_QUANT_H

#include "libxs_typedefs.h"

/** these are some quantization definitions, not sure if we want to
    move them into some main part of LIBXS */
/* F32 masking defines */
#define LIBXS_MASK_SIGN_F32      0x80000000
#define LIBXS_MASK_EXP_F32       0x7f800000
#define LIBXS_MASK_MANT_F32      0x007fffff
#define LIBXS_MASK_ABS_F32       0x7fffffff
#define LIBXS_MASK_FULL_F32      0xffffffff
#define LIBXS_MANT_SZ_F32        23
#define LIBXS_SZ_F32             32

/* DFP16 masking defines */
#define LIBXS_MANT_DFP16         15
#define LIBXS_RES_DFP16          libxs_sexp2_i8i(-(LIBXS_MANT_DFP16))

/* Quantization Rounding Defines */
#define LIBXS_QUANT_NO_ROUND       80000
#define LIBXS_QUANT_BIAS_ROUND     80001
#define LIBXS_QUANT_STOCH_ROUND    80002
#define LIBXS_QUANT_NEAREST_ROUND  80003
#define LIBXS_QUANT_FPHW_ROUND     80004

/** some quantization helper functions,
    @TODO need to be integrated better for all different ways of quantizations */
LIBXS_API void libxs_quantize_i16( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXS_API void libxs_dequantize_i16( short* in_buffer, float* out_buffer, int length, unsigned char scf );

/** BF16<->FP32 conversion functions */
LIBXS_API void libxs_truncate_convert_f32_bf16(const float* in, libxs_bfloat16* out, unsigned int length);
LIBXS_API void libxs_rnaz_convert_fp32_bf16(const float* in, libxs_bfloat16* out, unsigned int len);
LIBXS_API void libxs_rne_convert_fp32_bf16(const float* in, libxs_bfloat16* out, unsigned int len);
LIBXS_API void libxs_convert_bf16_f32(const libxs_bfloat16* in, float* out, unsigned int length);
/** FP16<->FP32 conversion functions */
LIBXS_API float libxs_convert_f16_to_f32( libxs_float16 in );
LIBXS_API libxs_float16 libxs_convert_f32_to_f16( float in );
/** BF8<->FP32 conversion functions */
LIBXS_API void libxs_rne_convert_fp32_bf8(const float* in, libxs_bfloat8* out, unsigned int len);
LIBXS_API void libxs_convert_bf8_f32(const libxs_bfloat8* in, float* out, unsigned int length);
LIBXS_API void libxs_stochastic_convert_fp32_bf8(const float* in, libxs_bfloat8* out, unsigned int length);

#endif /*LIBXS_LPFLT_QUANT_H*/
