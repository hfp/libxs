/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_pooling_forward.h"
#include "libxs_main.h"


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_f32_f32_c16(libxs_dnn_pooling* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_f32_f32_c32(libxs_dnn_pooling* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_f32_f32_c64(libxs_dnn_pooling* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c16(libxs_dnn_pooling* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c32(libxs_dnn_pooling* handle, int start_thread, int tid);
LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c64(libxs_dnn_pooling* handle, int start_thread, int tid);


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_f32_f32_c16(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
    typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_f32_f32_c32(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
    typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_f32_f32_c64(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
    typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
  }
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c16(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;

# define LIBXS_DNN_POOLING_FWD_BF16
  if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
    typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXS_DNN_POOLING_FWD_BF16
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c32(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;

# define LIBXS_DNN_POOLING_FWD_BF16
  if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
    typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXS_DNN_POOLING_FWD_BF16
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN LIBXS_INTRINSICS(LIBXS_X86_AVX512)
libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c64(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxs_bfloat16 element_input_type;
  typedef libxs_bfloat16 element_output_type;

# define LIBXS_DNN_POOLING_FWD_BF16
  if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
    typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXS_DNN_POOLING_FWD_BF16
#else /* should not happen */
  LIBXS_UNUSED(handle); LIBXS_UNUSED(start_thread); LIBXS_UNUSED(tid);
  status = LIBXS_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_custom(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check if we have input, output and mask */
  if ( handle->reg_input == 0 || handle->reg_output == 0 ||
       ( (handle->mask == 0) && (handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX) ) ) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXS_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( ( libxs_target_archid >= LIBXS_X86_AVX512 ) &&
       (handle->ofmblock == 16) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      LIBXS_ASSERT(NULL != handle->mask);
      status = libxs_dnn_pooling_st_fwd_custom_f32_f32_c16( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      LIBXS_ASSERT(NULL != handle->mask);
      status = libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c16( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxs_target_archid >= LIBXS_X86_AVX512 ) &&
       (handle->ofmblock == 32) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      LIBXS_ASSERT(NULL != handle->mask);
      status = libxs_dnn_pooling_st_fwd_custom_f32_f32_c32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      LIBXS_ASSERT(NULL != handle->mask);
      status = libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c32( handle, start_thread, tid);
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxs_target_archid >= LIBXS_X86_AVX512 ) &&
       (handle->ofmblock == 64) ) {
    if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
      LIBXS_ASSERT(NULL != handle->mask);
      status = libxs_dnn_pooling_st_fwd_custom_f32_f32_c64( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      LIBXS_ASSERT(NULL != handle->mask);
      status = libxs_dnn_pooling_st_fwd_custom_bf16_bf16_c64( handle, start_thread, tid);
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

      if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
        typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
      } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
      } else {
        status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
      }
    } else if (handle->desc.datatype_in == LIBXS_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXS_DNN_DATATYPE_BF16 ) {
      typedef libxs_bfloat16 element_input_type;
      typedef libxs_bfloat16 element_output_type;

# define LIBXS_DNN_POOLING_FWD_BF16
      if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_MAX ) {
# define LIBXS_DNN_POOLING_FWD_MAX
        typedef int element_mask_type;
# include "template/libxs_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_MAX
      } else if ( handle->desc.pooling_type == LIBXS_DNN_POOLING_AVG ) {
# define LIBXS_DNN_POOLING_FWD_AVG
# include "template/libxs_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXS_DNN_POOLING_FWD_AVG
      } else {
        status = LIBXS_DNN_ERR_UNSUPPORTED_POOLING;
      }
# undef LIBXS_DNN_POOLING_FWD_BF16
    } else {
      status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXS_API_INTERN libxs_dnn_err_t libxs_dnn_pooling_st_fwd_nhwc(libxs_dnn_pooling* handle, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_ERR_NOT_IMPLEMENTED;
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );
  return status;
}

