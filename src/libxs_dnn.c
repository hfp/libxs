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
/* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.),
   Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxs.h>
#include <libxs_sync.h>
#include "libxs_main.h"
#include "libxs_dnn_handle.h"
#include "libxs_dnn_convolution_forward.h"
#include "libxs_dnn_convolution_backward.h"
#include "libxs_dnn_convolution_weight_update.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_DEFINITION const char* libxs_dnn_get_error(libxs_dnn_err_t code)
{
  switch (code) {
    case LIBXS_DNN_SUCCESS:
      return "LIBXS DNN Success!";
    case LIBXS_DNN_WARN_FALLBACK:
      return "LIBXS DNN Warning: Falling back to naive code as target is currently not supported by LIBXS!";
    case LIBXS_DNN_ERR_GENERAL:
      return "LIBXS DNN Error: General error occured!";
    case LIBXS_DNN_ERR_CREATE_HANDLE:
      return "LIBXS DNN Error: Handle creation failed!";
    case LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE:
      return "LIBXS DNN Error: Requested datatype is not available!";
    case LIBXS_DNN_ERR_INVALID_BLOCKING:
      return "LIBXS DNN Error: Requested Input/Output buffer size cannot be blocked!";
    case LIBXS_DNN_ERR_INVALID_HANDLE:
      return "LIBXS DNN Error: An invalid handle was proivded!";
    case LIBXS_DNN_ERR_DATA_NOT_BOUND:
      return "LIBXS DNN Error: Not all required sources and destinations have been bound to convolution!";
    case LIBXS_DNN_ERR_CREATE_BUFFER:
      return "LIBXS DNN Error: Layer creation failed!";
    case LIBXS_DNN_ERR_INVALID_BUFFER:
      return "LIBXS DNN Error: Invalid buffer was specified!";
    case LIBXS_DNN_ERR_CREATE_FILTER:
      return "LIBXS DNN Error: Filter creation failed!";
    case LIBXS_DNN_ERR_INVALID_FILTER:
      return "LIBXS DNN Error: Invalid filter was specified!";
    case LIBXS_DNN_ERR_CREATE_BIAS:
      return "LIBXS DNN Error: Bias creation failed!";
    case LIBXS_DNN_ERR_INVALID_BIAS:
      return "LIBXS DNN Error: Invalid Bias was specified";
    case LIBXS_DNN_ERR_MISMATCH_BUFFER:
      return "LIBXS DNN Error: Layer doesn't match handle it should be bind to!";
    case LIBXS_DNN_ERR_INVALID_HANDLE_BUFFER:
      return "LIBXS DNN Error: Invalid hanlde or buffer!";
    case LIBXS_DNN_ERR_MISMATCH_FILTER:
      return "LIBXS DNN Error: Filter doens't match handle it should be bind to!";
    case LIBXS_DNN_ERR_INVALID_HANDLE_FILTER:
      return "LIBXS DNN Error: Invalid handle or filter!";
    case LIBXS_DNN_ERR_INVALID_KIND:
      return "LIBXS DNN Error: Invalid convolution kind!";
    case LIBXS_DNN_ERR_INVALID_FORMAT_NCHW:
      return "LIBXS DNN Error: NCHW format is currently not natively supported by LIBXS!";
    case LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT:
      return "LIBXS DNN Error: Unsupported destination format when copying data!";
    case LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT:
      return "LIBXS DNN Error: Unsupported source format when copying data!";
    case LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE:
      return "LIBXS DNN Error: Unsupported format when requesting a convolution!";
    case LIBXS_DNN_ERR_INVALID_FORMAT_KCRS:
      return "LIBXS DNN Error: KCRS format is currently not natively supported by LIBXS!";
    case LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL:
      return "LIBXS DNN Error: Invalid format was specified!";
    case LIBXS_DNN_ERR_CREATE_LAYOUT:
      return "LIBXS DNN Error: Layout creation failed!";
    case LIBXS_DNN_ERR_INVALID_LAYOUT:
      return "LIBXS DNN Error: Invalid layout was specified!";
    case LIBXS_DNN_ERR_UNSUPPORTED_ARCH:
      return "LIBXS DNN Error: Unsupported architecture!";
    default:
      return "LIBXS DNN Error: Unknown error or warning occured!";
  }
}


LIBXS_API_DEFINITION size_t libxs_dnn_typesize(libxs_dnn_datatype datatype)
{
  switch (datatype) {
    case LIBXS_DNN_DATATYPE_F32: return 4;
    case LIBXS_DNN_DATATYPE_I32: return 4;
    case LIBXS_DNN_DATATYPE_I16: return 2;
    case LIBXS_DNN_DATATYPE_I8:  return 1;
    /* no error expected as enumeration really arrives at an enum; compiler-checked */
    default: return 1;
  }
}


LIBXS_API_DEFINITION size_t libxs_dnn_get_simd_width(libxs_dnn_datatype datatype)
{
  size_t l_cl_width_bytes;
  if ( libxs_get_target_archid() == LIBXS_X86_GENERIC ) {
    l_cl_width_bytes = libxs_dnn_typesize(datatype);
  } else if ( libxs_get_target_archid() == LIBXS_X86_SSE3   ||
              libxs_get_target_archid() == LIBXS_X86_SSE4_1 || 
              libxs_get_target_archid() == LIBXS_X86_SSE4_2   ) {
    l_cl_width_bytes = 16;
  } else if ( libxs_get_target_archid() == LIBXS_X86_AVX2 ||
              libxs_get_target_archid() == LIBXS_X86_AVX ) {
    l_cl_width_bytes = 32;
  } else {
    l_cl_width_bytes = 64;
  }

  return l_cl_width_bytes/libxs_dnn_typesize(datatype);
}


LIBXS_API_DEFINITION libxs_dnn_conv_handle* libxs_dnn_create_conv_handle(
  libxs_dnn_conv_desc     conv_desc)
{
  libxs_dnn_err_t status;
  return libxs_dnn_create_conv_handle_check( conv_desc, &status);
}


LIBXS_API_DEFINITION libxs_dnn_conv_handle* libxs_dnn_create_conv_handle_check(
  libxs_dnn_conv_desc     conv_desc,
  libxs_dnn_err_t*        status)
{
  libxs_dnn_conv_handle* handle = 0;
  *status = LIBXS_DNN_SUCCESS;

  /* currently we don't support NCHW */
  if ( (conv_desc.buffer_format & LIBXS_DNN_CONV_FORMAT_NCHW) > 0 ) {
    *status = LIBXS_DNN_ERR_INVALID_FORMAT_NCHW;
    return 0;
  }
  /* currently we don't support KCRS */
  if ( (conv_desc.buffer_format & LIBXS_DNN_CONV_FORMAT_KCRS) > 0 ) {
    *status = LIBXS_DNN_ERR_INVALID_FORMAT_KCRS;
    return 0;
  }

  handle = (libxs_dnn_conv_handle*)malloc(sizeof(libxs_dnn_conv_handle));

  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = conv_desc;
    handle->datatype_in = conv_desc.datatype_in;
    handle->datatype_out = conv_desc.datatype_out;
    handle->buffer_format = conv_desc.buffer_format;
    handle->filter_format = conv_desc.filter_format;
    handle->fuse_ops = conv_desc.fuse_ops;
    handle->options = conv_desc.options;
    /* derive additional values */
    handle->ifhp = conv_desc.H + 2*conv_desc.pad_h_in;
    handle->ifwp = conv_desc.W + 2*conv_desc.pad_w_in;
    handle->ofh = (conv_desc.H + 2*conv_desc.pad_h - conv_desc.R) / conv_desc.u + 1;
    handle->ofw = (conv_desc.W + 2*conv_desc.pad_w - conv_desc.S) / conv_desc.v + 1;
    handle->ofhp = handle->ofh + 2*conv_desc.pad_h_out;
    handle->ofwp = handle->ofw + 2*conv_desc.pad_w_out;
    handle->avx512avx2fallback = 0;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->blocksifm = conv_desc.C;
    handle->blocksofm = conv_desc.K;
    handle->fwd_ofw_rb = 1;
    handle->fwd_ofw_rb_2 = 0;
    handle->fwd_ofh_rb = 1;
    handle->bwd_ofw_rb = 1;
    handle->bwd_ofh_rb = 1;
    handle->upd_ofw_rb = 1;
    handle->upd_ofh_rb = 1;
    handle->fm_lp_block = 1;
    handle->upd_use_thread_fil = 0;
    handle->upd_use_external_reduce = 0;
    handle->filter_transposed = 0;

    /* Set algorithm to use */
    if (conv_desc.algo == LIBXS_DNN_CONV_ALGO_AUTO) {
      handle->algo = LIBXS_DNN_CONV_ALGO_DIRECT;
    } else {
      handle->algo = conv_desc.algo;
    }

    if ( handle->algo == LIBXS_DNN_CONV_ALGO_DIRECT ) {
       *status = libxs_dnn_internal_create_conv_handle_direct_check( handle );
    } else {
      /* shouldn't happen */
    }
  }
  else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_conv_handle(const libxs_dnn_conv_handle* handle)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    /* deallocate data components; not an error to deallocate a NULL-pointer
       deallocate code known to be not registered; no index attached
       do not use libxs_release_kernel here! */
    if ( (libxs_get_target_archid() == LIBXS_X86_AVX512_MIC  ||
          libxs_get_target_archid() == LIBXS_X86_AVX512_CORE    ) && (handle->avx512avx2fallback == 0) ) {
      libxs_free(handle->code_fwd[0].pmm);
      libxs_free(handle->code_fwd[1].pmm);
      libxs_free(handle->code_fwd[2].pmm);
      libxs_free(handle->code_fwd[3].pmm);
      libxs_free(handle->code_bwd[0].pmm);
      if ((handle->filter_format == LIBXS_DNN_CONV_FORMAT_LIBXS) && (handle->buffer_format == LIBXS_DNN_CONV_FORMAT_LIBXS)) {
        libxs_free(handle->code_bwd[1].pmm);
        libxs_free(handle->code_bwd[2].pmm);
        libxs_free(handle->code_bwd[3].pmm);
      }
      libxs_free(handle->code_upd[0].pmm);
      if ((handle->filter_format == LIBXS_DNN_CONV_FORMAT_LIBXS) && (handle->buffer_format == LIBXS_DNN_CONV_FORMAT_LIBXS)) {
        libxs_free(handle->code_upd[1].pmm);
        libxs_free(handle->code_upd[2].pmm);
        libxs_free(handle->code_upd[3].pmm);
        libxs_free(handle->code_upd[4].pmm);
        libxs_free(handle->code_upd[5].pmm);
      }
    } else if ( (libxs_get_target_archid() == LIBXS_X86_AVX2) || (handle->avx512avx2fallback != 0) ) {
      libxs_free(handle->code_fwd[0].pmm);
      if (handle->fwd_ofw_rb_2 != 0) {
        libxs_free(handle->code_fwd[1].pmm);
      }
      libxs_free(handle->code_bwd[0].pmm);
      libxs_free(handle->code_upd[0].pmm);
    } else {
      /* no kernel was JITed */
    }

    /*Deallocate scratch in handle*/
    libxs_free(handle->scratch1);
    if (handle->scratch2 != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->scratch2); }
    libxs_free(handle->scratch3);
    libxs_free(handle->scratch4);

    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_conv_handle*)handle);
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_buffer* libxs_dnn_link_input_buffer(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format)
{
  libxs_dnn_err_t status;
  return libxs_dnn_link_input_buffer_check( handle, data, in_format, &status );
}


LIBXS_API_DEFINITION libxs_dnn_buffer* libxs_dnn_link_input_buffer_check(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format, libxs_dnn_err_t* status)
{
  libxs_dnn_buffer* buffer = (libxs_dnn_buffer*)malloc(sizeof(libxs_dnn_buffer));
  *status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && buffer != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    buffer->N = handle->desc.N;
    buffer->fmb = handle->blocksifm;
    buffer->bfm = handle->ifmblock;
    buffer->H = handle->ifhp;
    buffer->W = handle->ifwp;
    buffer->format = in_format;
    buffer->datatype = handle->datatype_in;
    buffer->lpb = handle->fm_lp_block;
    /* NHWC */
    if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXS_DNN_CONV_FORMAT_NHWC ) > 0)  && ((in_format & LIBXS_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    /* custom LIBXS format */
    } else if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXS_DNN_CONV_FORMAT_LIBXS ) > 0)  && ((in_format & LIBXS_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    } else {
      *status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_CREATE_BUFFER;
    buffer = 0;
  }

  if (*status != LIBXS_DNN_SUCCESS) {
    free((libxs_dnn_buffer*)buffer);
    buffer = 0;
  }

  return buffer;
}


LIBXS_API_DEFINITION libxs_dnn_conv_datalayout* libxs_dnn_get_input_buffer_datalayout(const libxs_dnn_conv_handle* handle) {
  libxs_dnn_err_t status;
  return libxs_dnn_get_input_buffer_datalayout_check( handle, &status );
}


LIBXS_API_DEFINITION libxs_dnn_conv_datalayout* libxs_dnn_get_input_buffer_datalayout_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status) {
  libxs_dnn_conv_datalayout* layout;

  *status = LIBXS_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxs_dnn_conv_datalayout*) malloc(sizeof(libxs_dnn_conv_datalayout));
    memset( layout, 0, sizeof(libxs_dnn_conv_datalayout) );

    if (layout != 0) {
      if ((handle->buffer_format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0) {
        if ( handle->datatype_in == LIBXS_DNN_DATATYPE_F32 ) {
          layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(5*sizeof(libxs_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

          layout->num_dims = 5;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->ifwp;
          layout->dim_size[2] = handle->ifhp;
          layout->dim_size[3] = handle->blocksifm;
          layout->dim_size[4] = handle->desc.N;
          layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_W;
          layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_H;
          layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[4] = LIBXS_DNN_CONV_DIMTYPE_N;
        } else if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_I16) || (handle->datatype_in == LIBXS_DNN_DATATYPE_I8) ) {
          layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(6*sizeof(libxs_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));

          layout->num_dims = 6;
          layout->dim_size[0] = handle->fm_lp_block;
          layout->dim_size[1] = handle->ifmblock;
          layout->dim_size[2] = handle->ifwp;
          layout->dim_size[3] = handle->ifhp;
          layout->dim_size[4] = handle->blocksifm;
          layout->dim_size[5] = handle->desc.N;
          layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_W;
          layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_H;
          layout->dim_type[4] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[5] = LIBXS_DNN_CONV_DIMTYPE_N;
        } else {
          free(layout);
          *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->buffer_format & LIBXS_DNN_CONV_FORMAT_NHWC) > 0) {
        layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(4*sizeof(libxs_dnn_conv_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

        layout->num_dims = 4;
        layout->dim_size[0] = handle->ifmblock * handle->blocksifm;
        layout->dim_size[1] = handle->ifwp;
        layout->dim_size[2] = handle->ifhp;
        layout->dim_size[3] = handle->desc.N;
        layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_C;
        layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_W;
        layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_H;
        layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_N;
      } else {
        free(layout);
        *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}


LIBXS_API_DEFINITION libxs_dnn_buffer* libxs_dnn_link_output_buffer(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format)
{
  libxs_dnn_err_t status;
  return libxs_dnn_link_output_buffer_check( handle, data, in_format, &status );
}


LIBXS_API_DEFINITION libxs_dnn_buffer* libxs_dnn_link_output_buffer_check(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format, libxs_dnn_err_t* status)
{
  libxs_dnn_buffer* buffer = (libxs_dnn_buffer*)malloc(sizeof(libxs_dnn_buffer));
  *status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && buffer != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    buffer->N = handle->desc.N;
    buffer->fmb = handle->blocksofm;
    buffer->bfm = handle->ofmblock;
    buffer->H = handle->ofhp;
    buffer->W = handle->ofwp;
    buffer->format = in_format;
    buffer->datatype = handle->datatype_out;
    buffer->lpb = 1;
    /* NHWC */
    if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXS_DNN_CONV_FORMAT_NHWC ) > 0)  && ((in_format & LIBXS_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    /* custom LIBXS format */
    } else if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXS_DNN_CONV_FORMAT_LIBXS ) > 0)  && ((in_format & LIBXS_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    } else {
      *status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_CREATE_BUFFER;
    buffer = 0;
  }

  if (*status != LIBXS_DNN_SUCCESS) {
    free((libxs_dnn_buffer*)buffer);
    buffer = 0;
  }

  return buffer;
}


LIBXS_API_DEFINITION libxs_dnn_conv_datalayout* libxs_dnn_get_output_buffer_datalayout(const libxs_dnn_conv_handle* handle) {
  libxs_dnn_err_t status;
  return libxs_dnn_get_output_buffer_datalayout_check( handle, &status );
}


LIBXS_API_DEFINITION libxs_dnn_conv_datalayout* libxs_dnn_get_output_buffer_datalayout_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status) {
  libxs_dnn_conv_datalayout* layout;

  *status = LIBXS_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxs_dnn_conv_datalayout*) malloc(sizeof(libxs_dnn_conv_datalayout));
    memset( layout, 0, sizeof(libxs_dnn_conv_datalayout) );

    if (layout != 0) {
      if ((handle->buffer_format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0) {
        if ( (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) || (handle->datatype_out == LIBXS_DNN_DATATYPE_I32) ) {
          layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(5*sizeof(libxs_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

          layout->num_dims = 5;
          layout->dim_size[0] = handle->ofmblock;
          layout->dim_size[1] = handle->ifwp;
          layout->dim_size[2] = handle->ifhp;
          layout->dim_size[3] = handle->blocksofm;
          layout->dim_size[4] = handle->desc.N;
          layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_W;
          layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_H;
          layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[4] = LIBXS_DNN_CONV_DIMTYPE_N;
        } else {
          free(layout);
          *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->buffer_format & LIBXS_DNN_CONV_FORMAT_NHWC) > 0) {
        layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(4*sizeof(libxs_dnn_conv_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

        layout->num_dims = 4;
        layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
        layout->dim_size[1] = handle->ifwp;
        layout->dim_size[2] = handle->ifhp;
        layout->dim_size[3] = handle->desc.N;
        layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_C;
        layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_W;
        layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_H;
        layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_N;
      } else {
        free(layout);
        *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_buffer(const libxs_dnn_buffer* buffer)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != buffer) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer, just deallocate if it's LIBXS private data */
    if ( (buffer->format & LIBXS_DNN_CONV_FORMAT_PTR) == 0 ) {
      libxs_free(buffer->data);
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_buffer*)buffer);
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_filter* libxs_dnn_link_filter(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format)
{
  libxs_dnn_err_t status;
  return libxs_dnn_link_filter_check(handle, data, in_format, &status);
}


LIBXS_API_DEFINITION libxs_dnn_filter* libxs_dnn_link_filter_check(const libxs_dnn_conv_handle* handle, const void* data, libxs_dnn_conv_format in_format, libxs_dnn_err_t* status)
{
  libxs_dnn_filter* filter = (libxs_dnn_filter*)malloc(sizeof(libxs_dnn_filter));
  *status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && filter != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    filter->ifmb = handle->blocksifm;
    filter->bifm = handle->ifmblock;
    filter->ofmb = handle->blocksofm;
    filter->bofm = handle->ofmblock;
    filter->R = handle->desc.R;
    filter->S = handle->desc.S;
    filter->format = in_format;
    filter->datatype = handle->datatype_in;
    filter->lpb = handle->fm_lp_block;
    /* RSCK */
    if ( ((handle->filter_format & in_format) > 0) && ((in_format & LIBXS_DNN_CONV_FORMAT_RSCK ) > 0)  && ((in_format & LIBXS_DNN_CONV_FORMAT_PTR ) > 0) ) {
      filter->data = (void*)data;
    /* custom LIBXS format */
    } else if ( ((handle->filter_format & in_format) > 0) && ((in_format & LIBXS_DNN_CONV_FORMAT_LIBXS ) > 0)  && ((in_format & LIBXS_DNN_CONV_FORMAT_PTR ) > 0) ) {
      filter->data = (void*)data;
    } else {
      *status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_CREATE_FILTER;
    filter = 0;
  }

  if (*status != LIBXS_DNN_SUCCESS) {
    *status = LIBXS_DNN_ERR_CREATE_FILTER;
    free((libxs_dnn_filter*)filter);
    filter = 0;
  }

  return filter;
}


LIBXS_API_DEFINITION libxs_dnn_conv_datalayout* libxs_dnn_get_filter_datalayout(const libxs_dnn_conv_handle* handle) {
  libxs_dnn_err_t status;
  return libxs_dnn_get_filter_datalayout_check( handle, &status );
}


LIBXS_API_DEFINITION libxs_dnn_conv_datalayout* libxs_dnn_get_filter_datalayout_check(const libxs_dnn_conv_handle* handle, libxs_dnn_err_t* status) {
  libxs_dnn_conv_datalayout* layout;

  *status = LIBXS_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxs_dnn_conv_datalayout*) malloc(sizeof(libxs_dnn_conv_datalayout));
    memset( layout, 0, sizeof(libxs_dnn_conv_datalayout) );

    if (layout != 0) {
      if ((handle->filter_format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0) {
        if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
          layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(6*sizeof(libxs_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));

          layout->num_dims = 6;
          layout->dim_size[0] = handle->ofmblock;
          layout->dim_size[1] = handle->ifmblock;
          layout->dim_size[2] = handle->desc.S;
          layout->dim_size[3] = handle->desc.R;
          layout->dim_size[4] = handle->blocksofm;
          layout->dim_size[5] = handle->blocksofm;
          layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_K;
          layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_S;
          layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_R;
          layout->dim_type[4] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[5] = LIBXS_DNN_CONV_DIMTYPE_K;
        } else if ( ((handle->datatype_in == LIBXS_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXS_DNN_DATATYPE_I32)) ||
                    ((handle->datatype_in == LIBXS_DNN_DATATYPE_I8)  && (handle->datatype_out == LIBXS_DNN_DATATYPE_I16)) ||
                    ((handle->datatype_in == LIBXS_DNN_DATATYPE_I8)  && (handle->datatype_out == LIBXS_DNN_DATATYPE_I32))    ) {
          layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(7*sizeof(libxs_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));

          layout->num_dims = 7;
          layout->dim_size[0] = handle->fm_lp_block;
          layout->dim_size[1] = handle->ofmblock;
          layout->dim_size[2] = handle->ifmblock;
          layout->dim_size[3] = handle->desc.S;
          layout->dim_size[4] = handle->desc.R;
          layout->dim_size[5] = handle->blocksofm;
          layout->dim_size[6] = handle->blocksofm;
          layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_K;
          layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_S;
          layout->dim_type[4] = LIBXS_DNN_CONV_DIMTYPE_R;
          layout->dim_type[5] = LIBXS_DNN_CONV_DIMTYPE_C;
          layout->dim_type[6] = LIBXS_DNN_CONV_DIMTYPE_K;
        } else {
          free(layout);
          *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->filter_format & LIBXS_DNN_CONV_FORMAT_RSCK) > 0) {
        layout->dim_type = (libxs_dnn_conv_dimtype*) malloc(4*sizeof(libxs_dnn_conv_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

        layout->num_dims = 4;
        layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
        layout->dim_size[1] = handle->ofmblock * handle->blocksofm;
        layout->dim_size[2] = handle->desc.S;
        layout->dim_size[3] = handle->desc.K;
        layout->dim_type[0] = LIBXS_DNN_CONV_DIMTYPE_K;
        layout->dim_type[1] = LIBXS_DNN_CONV_DIMTYPE_C;
        layout->dim_type[2] = LIBXS_DNN_CONV_DIMTYPE_S;
        layout->dim_type[3] = LIBXS_DNN_CONV_DIMTYPE_R;
      } else {
        free(layout);
        *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_filter(const libxs_dnn_filter* filter)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != filter) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    if ( (filter->format & LIBXS_DNN_CONV_FORMAT_PTR) == 0 ) {
      libxs_free(filter->data);
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_filter*)filter);
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_bias(const libxs_dnn_bias* bias)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != bias) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    libxs_free(bias->data);
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_bias*)bias);
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_BIAS;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_datalayout(libxs_dnn_conv_datalayout* layout) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != layout) {
    free(layout->dim_type);
    free(layout->dim_size);
    free(layout);
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_LAYOUT;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyin_buffer(const libxs_dnn_buffer* buffer, const void* data, libxs_dnn_conv_format in_format)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != buffer) {
    switch (in_format) {
      case LIBXS_DNN_CONV_FORMAT_NCHW: {
        if ( (buffer->format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0 ) {
          switch (buffer->datatype) {
            case LIBXS_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxs_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I32: {
              typedef int element_type;
#             include "template/libxs_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxs_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxs_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            default: {
              status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT;
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
      }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_zero_buffer(const libxs_dnn_buffer* buffer)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  const size_t size = (size_t)buffer->N * (size_t)buffer->fmb
                    * (size_t)buffer->bfm * (size_t)buffer->H * (size_t)buffer->W;
  size_t i;

  if (0 != buffer) {
    /* use for-loops to potentially leverage NUMA in the future */
    switch (buffer->datatype) {
      case LIBXS_DNN_DATATYPE_F32: {
        float* fp32_data = (float*)buffer->data;
        for (i = 0; i < size; ++i) fp32_data[i] = 0.0f;
      } break;
      case LIBXS_DNN_DATATYPE_I32: {
        int* int32_data = (int*)buffer->data;
        for (i = 0; i < size; ++i) int32_data[i] = 0;
      } break;
      case LIBXS_DNN_DATATYPE_I16: {
        short* int16_data = (short*)buffer->data;
        for (i = 0; i < size; ++i) int16_data[i] = 0;
      } break;
      case LIBXS_DNN_DATATYPE_I8: {
        char* int8_data = (char*)buffer->data;
        for (i = 0; i < size; ++i) int8_data[i] = 0;
      } break;
      default: {
        status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyout_buffer(const libxs_dnn_buffer* buffer, void* data, libxs_dnn_conv_format out_format)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != buffer) {
    switch (out_format) {
      case LIBXS_DNN_CONV_FORMAT_NCHW: {
        if ( (buffer->format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0 ) {
          switch (buffer->datatype) {
            case LIBXS_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxs_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I32: {
              typedef int element_type;
#             include "template/libxs_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxs_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxs_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            default: {
              status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT;
      }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyin_filter(const libxs_dnn_filter* filter, const void* data, libxs_dnn_conv_format in_format)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != filter) {
    switch (in_format) {
      case LIBXS_DNN_CONV_FORMAT_KCRS: {
        if ( (filter->format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0 ) {
          switch (filter->datatype) {
            case LIBXS_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxs_dnn_filter_copy_in_kcrs.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxs_dnn_filter_copy_in_kcrs.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxs_dnn_filter_copy_in_kcrs.tpl.c"
            } break;
            default: {
              status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT;
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
      }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyout_filter(const libxs_dnn_filter* filter, void* data, libxs_dnn_conv_format out_format)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != filter) {
    switch (out_format) {
      case LIBXS_DNN_CONV_FORMAT_KCRS: {
        if ( (filter->format & LIBXS_DNN_CONV_FORMAT_LIBXS) > 0 ) {
          switch (filter->datatype) {
            case LIBXS_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxs_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I32: {
              typedef int element_type;
#             include "template/libxs_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxs_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            case LIBXS_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxs_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            default: {
              status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXS_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_UNSUPPORTED_DST_FORMAT;
      }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


#if 0
LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyin_bias(const libxs_dnn_bias* bias, const void* data)
{
  LIBXS_UNUSED(bias); LIBXS_UNUSED(data); /* TODO: libxs_dnn_copyin_input */
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyout_bias(const libxs_dnn_bias* bias, void* data)
{
  LIBXS_UNUSED(bias); LIBXS_UNUSED(data); /* TODO: libxs_dnn_copyin_input */
}
#endif


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_bind_input_buffer(libxs_dnn_conv_handle* handle, const libxs_dnn_buffer* buffer)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && buffer != 0) {
    /* check if format matches */
    if ( handle->desc.N == buffer->N
      && handle->ifwp == buffer->W
      && handle->ifhp == buffer->H
      && handle->ifmblock == buffer->bfm
      && handle->blocksifm == buffer->fmb
      && handle->datatype_in == buffer->datatype
      && handle->fm_lp_block == buffer->lpb
      && ((handle->buffer_format & buffer->format) > 0) )
    {
      handle->input = (libxs_dnn_buffer*)buffer;
    }
    else {
      status = LIBXS_DNN_ERR_MISMATCH_BUFFER;
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_bind_output_buffer(libxs_dnn_conv_handle* handle, const libxs_dnn_buffer* buffer)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && buffer != 0) {
    /* check if format matches */
    if ( handle->desc.N == buffer->N
      && handle->ofwp == buffer->W
      && handle->ofhp == buffer->H
      && handle->ofmblock == buffer->bfm
      && handle->blocksofm == buffer->fmb
      && buffer->lpb == 1
      && ((handle->buffer_format & buffer->format) > 0)
      && handle->datatype_out == buffer->datatype )
    {
      handle->output = (libxs_dnn_buffer*)buffer;
    }
    else {
      status = LIBXS_DNN_ERR_MISMATCH_BUFFER;
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_bind_filter(libxs_dnn_conv_handle* handle, const libxs_dnn_filter* filter)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && filter != 0) {
    /* check if format matches */
    if ( handle->desc.R == filter->R
      && handle->desc.S == filter->S
      && handle->ifmblock == filter->bifm
      && handle->blocksifm == filter->ifmb
      && handle->ofmblock == filter->bofm
      && handle->blocksofm == filter->ofmb
      && handle->fm_lp_block == filter->lpb
      && ((handle->filter_format & filter->format) > 0)
      && handle->datatype_in == filter->datatype)
    {
      handle->filter = (libxs_dnn_filter*)filter;
    }
    else {
      status = LIBXS_DNN_ERR_MISMATCH_FILTER;
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_FILTER;
  }

  return status;
}


LIBXS_INLINE LIBXS_RETARGETABLE libxs_dnn_err_t internal_convolve_st(libxs_dnn_conv_handle* handle,
  libxs_dnn_conv_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_CONV_KIND_FWD: {
        switch (handle->buffer_format) {
          case LIBXS_DNN_CONV_FORMAT_LIBXS: {
            switch (handle->filter_format) {
              case LIBXS_DNN_CONV_FORMAT_LIBXS: {
                status = libxs_dnn_convolve_st_fwd_custom_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXS_DNN_CONV_FORMAT_NHWC: {
            switch (handle->filter_format) {
              case LIBXS_DNN_CONV_FORMAT_RSCK: {
                status = libxs_dnn_convolve_st_fwd_nhwc_rsck(handle, start_thread, tid);
              } break;
              case LIBXS_DNN_CONV_FORMAT_LIBXS: {
                status = libxs_dnn_convolve_st_fwd_nhwc_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
          }
        }
      } break;
      case LIBXS_DNN_CONV_KIND_BWD: {
        switch (handle->buffer_format) {
          case LIBXS_DNN_CONV_FORMAT_LIBXS: {
            switch (handle->filter_format) {
              case LIBXS_DNN_CONV_FORMAT_LIBXS: {
                status = libxs_dnn_convolve_st_bwd_custom_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXS_DNN_CONV_FORMAT_NHWC: {
            switch (handle->filter_format) {
              case LIBXS_DNN_CONV_FORMAT_RSCK: {
                status = libxs_dnn_convolve_st_bwd_nhwc_rsck(handle, start_thread, tid);
              } break;
              case LIBXS_DNN_CONV_FORMAT_LIBXS: {
                status = libxs_dnn_convolve_st_bwd_nhwc_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
          }
        }
      } break;
      case LIBXS_DNN_CONV_KIND_UPD: {
        switch (handle->buffer_format) {
          case LIBXS_DNN_CONV_FORMAT_LIBXS: {
            switch (handle->filter_format) {
              case LIBXS_DNN_CONV_FORMAT_LIBXS: {
                status = libxs_dnn_convolve_st_upd_custom_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXS_DNN_CONV_FORMAT_NHWC: {
            switch (handle->filter_format) {
              case LIBXS_DNN_CONV_FORMAT_RSCK: {
                status = libxs_dnn_convolve_st_upd_nhwc_rsck(handle, start_thread, tid);
              } break;
              case LIBXS_DNN_CONV_FORMAT_LIBXS: {
                status = libxs_dnn_convolve_st_upd_nhwc_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          default: {
            status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
          }
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_INVALID_KIND;
      }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION void libxs_dnn_convolve(libxs_dnn_conv_handle* handle, libxs_dnn_conv_kind kind)
{
#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
  {
    const int tid = omp_get_thread_num();
    internal_convolve_st(handle, kind, 0, tid);
  }
#else
  internal_convolve_st(handle, kind, 0/*start_thread*/, 0/*tid*/);
#endif
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_transpose_filter(libxs_dnn_conv_handle* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int ofm1, ifm1, kj, ki, ifm2, ofm2;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check that filter is in RSCK storage */
  if ( (handle->filter_format & LIBXS_DNN_CONV_FORMAT_RSCK) == 0 ) {
    status = LIBXS_DNN_ERR_MISMATCH_FILTER;
    return status;
  }

  /* check that we are in FP32 */
  if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
    LIBXS_VLA_DECL(6, float, wt, (float*)handle->filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
    LIBXS_VLA_DECL(6, float, tr_wt, (float*)handle->scratch1, handle->desc.S, handle->blocksofm, handle->ofmblock, handle->blocksifm, handle->ifmblock);

    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
        for(kj=0; kj < handle->desc.R; ++kj) {
          for(ki=0; ki < handle->desc.S; ++ki) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXS_VLA_ACCESS(6, tr_wt, kj, ki, ofm1, ofm2, ifm1, ifm2, handle->desc.S, handle->blocksofm, handle->ofmblock, handle->blocksifm, handle->ifmblock) =
                  LIBXS_VLA_ACCESS(6, wt,  kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
              }
            }
          }
        }  
      }
    }
    handle->filter_transposed = 1;
    return status;
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_reduce_wu_filters(libxs_dnn_conv_handle* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int i, j, filter_size;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* calculate filter size */
  filter_size = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * handle->ofmblock * handle->ifmblock;

  /* check that we are in FP32 */
  if (handle->datatype_in == LIBXS_DNN_DATATYPE_F32 && handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
    if (handle->upd_use_external_reduce != 0) {
      float* filter_ptr = (float*)handle->filter->data;
      for ( i = 0; i < handle->desc.threads; i++ ) {
        float* tmp_filter_ptr = ((float*)handle->scratch4) + (i*filter_size);
        for ( j = 0; j < filter_size; j++) {
          filter_ptr[j] += tmp_filter_ptr[j];
        }
      }
    }
  } else {
    status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_get_codegen_success(libxs_dnn_conv_handle* handle, libxs_dnn_conv_kind kind) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_CONV_KIND_FWD: {
        if (handle->code_fwd[0].xconv.sconv == 0) {
          status = LIBXS_DNN_WARN_FALLBACK;
        }
      } break;
      case LIBXS_DNN_CONV_KIND_BWD: {
        if (handle->code_bwd[0].xconv.sconv == 0) {
          status = LIBXS_DNN_WARN_FALLBACK;
        }
      } break;
      case LIBXS_DNN_CONV_KIND_UPD: {
        if (handle->code_upd[0].xconv.sconv == 0) {
          status = LIBXS_DNN_WARN_FALLBACK;
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_get_parallel_tasks(libxs_dnn_conv_handle* handle, libxs_dnn_conv_kind kind, unsigned int* num_tasks) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_CONV_KIND_FWD: {
        *num_tasks = handle->desc.N * handle->blocksofm;
      } break;
      case LIBXS_DNN_CONV_KIND_BWD: {
        *num_tasks = handle->desc.N * handle->blocksifm;
      } break;
      case LIBXS_DNN_CONV_KIND_UPD: {
        if (handle->upd_use_thread_fil > 0) {
          *num_tasks = handle->desc.N * handle->blocksifm * handle->blocksofm;
        } else {
          *num_tasks = handle->blocksifm * handle->blocksofm;
        }
      } break;
      default: {
        status = LIBXS_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_convolve_st(libxs_dnn_conv_handle* handle,
  libxs_dnn_conv_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  return internal_convolve_st(handle, kind, start_thread, tid);
}


#if defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API)

LIBXS_API_DEFINITION libxs_sconvfunction libxs_create_sconv_forward(
  const libxs_convolution_forward_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cfwd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CFWD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: invalid descriptor (forward convolution)!\n");
    }
  }
#endif
  return code.xconv.sconv;
}


LIBXS_API_DEFINITION libxs_sconvfunction libxs_create_sconv_backward(
  const libxs_convolution_backward_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cbwd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CBWD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: invalid descriptor (backward convolution)!\n");
    }
  }
#endif
  return code.xconv.sconv;
}


LIBXS_API_DEFINITION libxs_sconvfunction libxs_create_sconv_update_weights(
  const libxs_convolution_weight_update_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cupd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CUPD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: invalid convolution descriptor (weight update)!\n");
    }
  }
#endif
  return code.xconv.sconv;
}

LIBXS_API_DEFINITION void* libxs_create_xconv_forward(
  const libxs_convolution_forward_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cfwd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CFWD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: invalid descriptor (forward convolution)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXS_API_DEFINITION void* libxs_create_xconv_backward(
  const libxs_convolution_backward_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cbwd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CBWD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: invalid descriptor (backward convolution)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXS_API_DEFINITION void* libxs_create_xconv_update_weights(
  const libxs_convolution_weight_update_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
  if (0 != descriptor) {
    libxs_build_request request;
    request.descriptor.cupd = descriptor;
    request.kind = LIBXS_BUILD_KIND_CUPD;
    libxs_build(&request, LIBXS_REGSIZE/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXS: invalid convolution descriptor (weight update)!\n");
    }
  }
#endif
  return code.pmm;
}

#endif /*defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API)*/
