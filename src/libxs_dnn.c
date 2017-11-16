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
/* Hans Pabst, Alexander Heinecke, Rajkishore Barik (Intel Corp.)
 ******************************************************************************/
#include <libxs.h>
#include <libxs_sync.h>
#include "libxs_main.h"
#include "libxs_dnn_handle.h"
#include "libxs_dnn_convolution_forward.h"
#include "libxs_dnn_convolution_backward.h"
#include "libxs_dnn_convolution_weight_update.h"
#include "libxs_dnn_convolution_winograd_forward.h"
#include "libxs_dnn_convolution_winograd_backward.h"
#include "libxs_dnn_convolution_winograd_weight_update.h"


#define FP64_BN_STATS

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


LIBXS_API_DEFINITION void libxs_dnn_init(int target_arch)
{
  libxs_dnn_convolve_winograd_fwd_init(target_arch);
  libxs_dnn_convolve_winograd_bwd_init(target_arch);
}


LIBXS_API_DEFINITION void libxs_dnn_finalize(void)
{
  libxs_dnn_convolve_winograd_fwd_finalize();
  libxs_dnn_convolve_winograd_bwd_finalize();
}


LIBXS_API_DEFINITION const char* libxs_dnn_get_error(libxs_dnn_err_t code)
{
  switch (code) {
    case LIBXS_DNN_SUCCESS:
      return "LIBXS DNN Success!";
    case LIBXS_DNN_WARN_FALLBACK:
      return "LIBXS DNN Warning: Falling back to naive code as target is currently not supported by LIBXS!";
    case LIBXS_DNN_ERR_GENERAL:
      return "LIBXS DNN Error: General error occurred!";
    case LIBXS_DNN_ERR_CREATE_HANDLE:
      return "LIBXS DNN Error: Handle creation failed!";
    case LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE:
      return "LIBXS DNN Error: Requested datatype is not available!";
    case LIBXS_DNN_ERR_INVALID_BLOCKING:
      return "LIBXS DNN Error: Requested Input/Output buffer size cannot be blocked!";
    case LIBXS_DNN_ERR_INVALID_HANDLE:
      return "LIBXS DNN Error: An invalid handle was provided!";
    case LIBXS_DNN_ERR_DATA_NOT_BOUND:
      return "LIBXS DNN Error: Not all required sources and destinations have been bound to convolution!";
    case LIBXS_DNN_ERR_CREATE_TENSOR:
      return "LIBXS DNN Error: Tensor creation failed!";
    case LIBXS_DNN_ERR_INVALID_TENSOR:
      return "LIBXS DNN Error: Invalid tensor was specified!";
    case LIBXS_DNN_ERR_MISMATCH_TENSOR:
      return "LIBXS DNN Error: Tensor doesn't match handle it should be bind to!";
    case LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR:
      return "LIBXS DNN Error: Invalid handle or tensor!";
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
    case LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED:
      return "LIBXS DNN Error: scratch binding failed as scratch was not allocated!";
    case LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE:
      return "LIBXS DNN Error: an unknown tensor type was provided!";
    case LIBXS_DNN_ERR_INVALID_ALGO:
      return "LIBXS DNN Error: Invalid algorithm was specified!";
    case LIBXS_DNN_ERR_INVALID_PADDING:
      return "LIBXS DNN Error: Invalid padding was specified!";
    default:
      return "LIBXS DNN Error: Unknown error or warning occurred!";
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
  if ( libxs_target_archid == LIBXS_X86_GENERIC ) {
    l_cl_width_bytes = libxs_dnn_typesize(datatype);
  } else if ( libxs_target_archid == LIBXS_X86_SSE3 ||
      libxs_target_archid == LIBXS_X86_SSE4 ) {
    l_cl_width_bytes = 16;
  } else if ( libxs_target_archid == LIBXS_X86_AVX2 ||
      libxs_target_archid == LIBXS_X86_AVX ) {
    l_cl_width_bytes = 32;
  } else {
    l_cl_width_bytes = 64;
  }

  return l_cl_width_bytes/libxs_dnn_typesize(datatype);
}


LIBXS_API_DEFINITION libxs_dnn_layer* libxs_dnn_create_conv_layer(
    libxs_dnn_conv_desc     conv_desc,
    libxs_dnn_err_t*        status)
{
  libxs_dnn_layer* handle = 0;
  *status = LIBXS_DNN_SUCCESS;

  /* currently we don't support NCHW */
  if ( (conv_desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCHW) > 0 ) {
    *status = LIBXS_DNN_ERR_INVALID_FORMAT_NCHW;
    return 0;
  }
  /* currently we don't support KCRS */
  if ( (conv_desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_KCRS) > 0 ) {
    *status = LIBXS_DNN_ERR_INVALID_FORMAT_KCRS;
    return 0;
  }

  /* TODO remove this check later */
  if (conv_desc.N != conv_desc.threads) {
    printf("For this version of LIBXS minibatch size needs to match with number of threads!\n");
    exit(-1);
  }

  handle = (libxs_dnn_layer*)malloc(sizeof(libxs_dnn_layer));

  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = conv_desc;
    handle->datatype_in = conv_desc.datatype_in;
    handle->datatype_out = conv_desc.datatype_out;
    /* select the intermediate format, only applicable for integer types */
    if ( (conv_desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (conv_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXS_DNN_DATATYPE_I16) && (conv_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
    }  else if ( (conv_desc.datatype_in == LIBXS_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXS_DNN_DATATYPE_I32) ) {
      /* error */
    }  else if ( (conv_desc.datatype_in == LIBXS_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
    } else {
      /* fine, no error */
    }
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
    handle->blocksifm_blocking = 1;
    handle->blocksofm_blocking = 1;
    handle->upd_use_thread_fil = 0;
    handle->upd_use_external_reduce = 0;
    handle->filter_transposed = 0;
    /* Set algorithm to use */
    if (conv_desc.algo == LIBXS_DNN_CONV_ALGO_AUTO) {
      if ( (((conv_desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) || ((conv_desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NHWC) > 0)) &&
          ((conv_desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) &&
          (3 == conv_desc.R) && (3 == conv_desc.S) &&
          (1 == conv_desc.u) && (1 == conv_desc.v) &&
          (0 == (conv_desc.C % 16)) && (0 == (conv_desc.K % 16)) &&
          (conv_desc.datatype_in  == LIBXS_DNN_DATATYPE_F32) ) {
        handle->algo = LIBXS_DNN_CONV_ALGO_WINOGRAD;
      } else {
        handle->algo = LIBXS_DNN_CONV_ALGO_DIRECT;
      }
    } else {
      handle->algo = conv_desc.algo;
    }
    if (handle->algo != LIBXS_DNN_CONV_ALGO_WINOGRAD && handle->algo != LIBXS_DNN_CONV_ALGO_DIRECT ) {
      *status = LIBXS_DNN_ERR_INVALID_ALGO;
      free(handle);
      handle = 0;
      return 0;
    }
    /* @TODO we might want to fall back to direct convolution if winograd fails */
    if ( handle->algo == LIBXS_DNN_CONV_ALGO_WINOGRAD ) {
      *status = libxs_dnn_internal_create_conv_handle_winograd_check( handle );
      if ( *status == LIBXS_DNN_WARN_FALLBACK ) {
        handle->algo = LIBXS_DNN_CONV_ALGO_DIRECT;
        *status = libxs_dnn_internal_create_conv_handle_direct( handle );
      }
    }
    else if ( handle->algo == LIBXS_DNN_CONV_ALGO_DIRECT ) {
      *status = libxs_dnn_internal_create_conv_handle_direct( handle );
    } else {
      assert(0/*should not happen*/);
    }
  }
  else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_conv_layer(const libxs_dnn_layer* handle)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int loop;

  if (0 != handle) {
    /* deallocate data components; not an error to deallocate a NULL-pointer
       deallocate code known to be not registered; no index attached
       do not use libxs_release_kernel here! */
    if ( (libxs_target_archid == LIBXS_X86_AVX512_MIC  ||
          libxs_target_archid == LIBXS_X86_AVX512_KNM  ||
          libxs_target_archid == LIBXS_X86_AVX512_CORE ) && (handle->avx512avx2fallback == 0) ) {
      if (handle->custom_format_type != LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
        libxs_free(handle->code_fwd[0].pmm);
      }
      libxs_free(handle->code_fwd[1].pmm);
      libxs_free(handle->code_fwd[2].pmm);
      libxs_free(handle->code_fwd[3].pmm);
      if (handle->custom_format_type != LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
        libxs_free(handle->code_bwd[0].pmm);
      }
      if ((handle->filter_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS) && (handle->buffer_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS)) {
        libxs_free(handle->code_bwd[1].pmm);
        libxs_free(handle->code_bwd[2].pmm);
        libxs_free(handle->code_bwd[3].pmm);
      }
      if (handle->custom_format_type != LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
        libxs_free(handle->code_upd[0].pmm);
      }
      if ((handle->filter_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS) && (handle->buffer_format == LIBXS_DNN_TENSOR_FORMAT_LIBXS)) {
        libxs_free(handle->code_upd[1].pmm);
        libxs_free(handle->code_upd[2].pmm);
        libxs_free(handle->code_upd[3].pmm);
        libxs_free(handle->code_upd[4].pmm);
        libxs_free(handle->code_upd[5].pmm);
      }
    } else if ( (libxs_target_archid == LIBXS_X86_AVX2) || (handle->avx512avx2fallback != 0) ) {
      if (handle->custom_format_type != LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
        libxs_free(handle->code_fwd[0].pmm);
      }
      if (handle->fwd_ofw_rb_2 != 0) {
        libxs_free(handle->code_fwd[1].pmm);
      }
      if (handle->custom_format_type != LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
        libxs_free(handle->code_bwd[0].pmm);
      }
      if (handle->custom_format_type != LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
        libxs_free(handle->code_upd[0].pmm);
      }
    } else {
      /* no kernel was JITed */
    }

    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->barrier); }

    /*Deallocate scratch in handle*/
    libxs_free(handle->scratch1);
    libxs_free(handle->scratch3);
    libxs_free(handle->scratch4);

    /* Deallocate per-thread jitted data structures */
    if ( handle->use_thread_private_jit ) {

      /* Free per thread allocated arrays  */
      for (loop = 0; loop < handle->desc.threads; loop++) {
        /* Fwd related arrays */
        if ( handle->compute_fwd_indices_ptrs[loop] != NULL ) {
          libxs_free( handle->compute_fwd_indices_ptrs[loop] );
        }
        if ( handle->kernel_fwd_variant_ptrs[loop] != NULL ) {
          libxs_free( handle->kernel_fwd_variant_ptrs[loop] );
        }
        if ( handle->fwd_code_segments[loop] != NULL ) {
          libxs_free( handle->fwd_code_segments[loop] );
        }
        /* Bwd related arrays  */
        if ( handle->compute_bwd_indices_ptrs[loop] != NULL ) {
          libxs_free( handle->compute_bwd_indices_ptrs[loop] );
        }
        if ( handle->kernel_bwd_variant_ptrs[loop] != NULL ) {
          libxs_free( handle->kernel_bwd_variant_ptrs[loop] );
        }
        if ( handle->bwd_code_segments[loop] != NULL ) {
          libxs_free( handle->bwd_code_segments[loop] );
        }
        if ( handle->transpose_bwd_indices_ptrs[loop] != NULL ) {
          libxs_free( handle->transpose_bwd_indices_ptrs[loop] );
        }
      }

      /* Free shared arrays  */
      free( handle->compute_fwd_indices_ptrs );
      free( handle->kernel_fwd_variant_ptrs );
      free( handle->n_entries_fwd );
      free( handle->n_fwd_code_segments );
      free( handle->ofh_fwd_start );
      free( handle->ofh_fwd_end );

      free( handle->compute_bwd_indices_ptrs );
      free( handle->kernel_bwd_variant_ptrs );
      free( handle->n_entries_bwd );
      free( handle->n_bwd_code_segments );
      free( handle->ofh_bwd_start );
      free( handle->ofh_bwd_end );
      free( handle->transpose_bwd_indices_ptrs );
    }

    if (handle->padding_flag) libxs_free(handle->scratch5);
    if (handle->use_lp_kernel == 1) libxs_free(handle->scratch6);

    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_layer*)handle);
  }
#if 0 /* releasing a NULL-handle should be not an error (similar to freeing a NULL pointer) */
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }
#endif
  return status;
}


LIBXS_API_DEFINITION libxs_dnn_tensor* libxs_dnn_link_tensor(const libxs_dnn_tensor_datalayout* layout, const void* data, libxs_dnn_err_t* status)
{
  return libxs_dnn_link_qtensor(layout, data, 0, status);
}


LIBXS_API_DEFINITION libxs_dnn_tensor* libxs_dnn_link_qtensor(const libxs_dnn_tensor_datalayout* layout, const void* data, const char exp, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor* tensor = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
  *status = LIBXS_DNN_SUCCESS;

  if (layout != 0 && tensor != 0 && data != 0) {
    memset(tensor, 0, sizeof(libxs_dnn_tensor));
    tensor->layout = libxs_dnn_duplicate_tensor_datalayout(layout, status);
    tensor->data = (void*)data;
    tensor->exp = exp;
    /* when layout copy failed, free layout */
    if (*status != LIBXS_DNN_SUCCESS) {
      libxs_dnn_destroy_tensor_datalayout(tensor->layout);
    }
  } else {
    *status = LIBXS_DNN_ERR_CREATE_TENSOR;
  }

  if (*status != LIBXS_DNN_SUCCESS) {
    free((libxs_dnn_tensor*)tensor);
    tensor = 0;
  }

  return tensor;
}


LIBXS_API_DEFINITION libxs_dnn_tensor_datalayout* libxs_dnn_create_tensor_datalayout(const libxs_dnn_layer* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* layout;

  *status = LIBXS_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      layout->custom_format = handle->custom_format_type;
      if ( (type == LIBXS_DNN_REGULAR_INPUT)  || (type == LIBXS_DNN_GRADIENT_INPUT)  || (type == LIBXS_DNN_INPUT)  ||
           (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT)    ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_ACTIVATION;

        if ((handle->buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( ((handle->datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            if (handle->custom_format_type == LIBXS_DNN_TENSOR_FORMAT_LIBXS_1) {
              layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(5*sizeof(libxs_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) || (type == LIBXS_DNN_INPUT) ) {
                  layout->dim_size[0] = handle->ifmblock;
                  layout->dim_size[1] = handle->ifwp;
                  layout->dim_size[2] = handle->ifhp;
                  layout->dim_size[3] = handle->blocksifm;
                  layout->dim_size[4] = handle->desc.N;
                } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                  layout->dim_size[0] = handle->ofmblock;
                  layout->dim_size[1] = handle->ofwp;
                  layout->dim_size[2] = handle->ofhp;
                  layout->dim_size[3] = handle->blocksofm;
                  layout->dim_size[4] = handle->desc.N;
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              }
            } else if (handle->custom_format_type == LIBXS_DNN_TENSOR_FORMAT_LIBXS_2) {
              layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(6*sizeof(libxs_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 6;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) || (type == LIBXS_DNN_INPUT) ) {
                  layout->dim_size[0] = handle->ifmblock;
                  layout->dim_size[1] = handle->nbImg;
                  layout->dim_size[2] = handle->ifwp;
                  layout->dim_size[3] = handle->ifhp;
                  layout->dim_size[4] = handle->desc.N/handle->nbImg;
                  layout->dim_size[5] = handle->blocksifm;
                } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                  layout->dim_size[0] = handle->ofmblock;
                  layout->dim_size[1] = handle->nbImg;
                  layout->dim_size[2] = handle->ofwp;
                  layout->dim_size[3] = handle->ofhp;
                  layout->dim_size[4] = handle->desc.N/handle->nbImg;
                  layout->dim_size[5] = handle->blocksofm;
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          /* @TODO this need to change */
          } else if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXS_DNN_DATATYPE_I32) ) {
            if ( ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_INPUT) )  ) {
              layout->datatype = handle->datatype_in;
            } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
              layout->datatype = handle->datatype_out;     
            }
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(6*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) || (type == LIBXS_DNN_INPUT) )   {
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ifmblock;
                layout->dim_size[2] = handle->ifwp;
                layout->dim_size[3] = handle->ifhp;
                layout->dim_size[4] = handle->blocksifm_lp;
                layout->dim_size[5] = handle->desc.N;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                layout->dim_size[0] = 1;
                layout->dim_size[1] = handle->ofmblock;
                layout->dim_size[2] = handle->ofwp;
                layout->dim_size[3] = handle->ofhp;
                layout->dim_size[4] = handle->blocksofm;
                layout->dim_size[5] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            if ( ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_INPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT)  )  ) {
              layout->datatype = handle->datatype_in;
            } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_OUTPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) ) {
              layout->datatype = handle->datatype_out;     
            }
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(6*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_INPUT) )   {
                layout->num_dims = 6;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ifmblock;
                layout->dim_size[2] = handle->ifwp;
                layout->dim_size[3] = handle->ifhp;
                layout->dim_size[4] = handle->blocksifm_lp;
                layout->dim_size[5] = handle->desc.N;
              } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT )   {
                layout->num_dims = 6;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ofmblock;
                layout->dim_size[2] = handle->ofwp;
                layout->dim_size[3] = handle->ofhp;
                layout->dim_size[4] = handle->blocksofm_lp;
                layout->dim_size[5] = handle->desc.N;
              } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else{
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }

        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_REGULAR_FILTER) || (type == LIBXS_DNN_GRADIENT_FILTER) || (type == LIBXS_DNN_FILTER) ) {
        layout->format = handle->filter_format;
        layout->tensor_type = LIBXS_DNN_FILTER;

        if ((handle->filter_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(6*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->blocksofm;
            }
          } else if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            if ( (type == LIBXS_DNN_REGULAR_FILTER) || (type == LIBXS_DNN_FILTER) ) {
              layout->datatype = handle->datatype_in;
            } else if (type == LIBXS_DNN_GRADIENT_FILTER) {
              layout->datatype = handle->datatype_out;       
            }
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(7*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              if ((type == LIBXS_DNN_REGULAR_FILTER) || (type == LIBXS_DNN_FILTER)) {
                layout->num_dims = 7;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_S;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_R;
                layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[6] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ofmblock;
                layout->dim_size[2] = handle->ifmblock;
                layout->dim_size[3] = handle->desc.S;
                layout->dim_size[4] = handle->desc.R;
                layout->dim_size[5] = handle->blocksifm_lp;
                layout->dim_size[6] = handle->blocksofm;
              } else if (type == LIBXS_DNN_GRADIENT_FILTER) {
                layout->num_dims = 6;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
                layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ifmblock;
                layout->dim_size[2] = handle->desc.S;
                layout->dim_size[3] = handle->desc.R;
                layout->dim_size[4] = handle->blocksifm;
                layout->dim_size[5] = handle->blocksofm;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#if 0
        } else if ((handle->filter_format & LIBXS_DNN_TENSOR_FORMAT_RSCK) > 0) {
          layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
          if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
            layout->num_dims = 4;
            layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
            layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
            layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
            layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
            layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
            layout->dim_size[2] = handle->desc.S;
            layout->dim_size[3] = handle->desc.K;
          }
#endif
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( type == LIBXS_DNN_REGULAR_FILTER_TRANS ) {
        layout->format = handle->filter_format;
        layout->tensor_type = LIBXS_DNN_REGULAR_FILTER_TRANS;

        if ((handle->filter_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(6*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ifmblock;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
              layout->dim_size[4] = handle->blocksofm;
              layout->dim_size[5] = handle->blocksifm;
            }
          } else if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_I16) ||
              (handle->datatype_in == LIBXS_DNN_DATATYPE_I8) ) {
            layout->datatype = handle->datatype_in;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(7*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 7;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[4] = LIBXS_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[5] = LIBXS_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[6] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->ifmblock;
              layout->dim_size[3] = handle->desc.S;
              layout->dim_size[4] = handle->desc.R;
              layout->dim_size[5] = handle->blocksofm;
              layout->dim_size[6] = handle->blocksifm*handle->fm_lp_block;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#if 0
        } else if ((handle->filter_format & LIBXS_DNN_TENSOR_FORMAT_RSCK) > 0) {
          layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
          if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
            layout->num_dims = 4;
            layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
            layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_S;
            layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_R;
            layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
            layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
            layout->dim_size[2] = handle->desc.S;
            layout->dim_size[3] = handle->desc.K;
          }
#endif
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_REGULAR_BIAS) || (type == LIBXS_DNN_GRADIENT_BIAS) || (type == LIBXS_DNN_BIAS) ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_BIAS;

        if ((handle->buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->datatype_out;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->blocksofm;
            }
#if 0
          } else if ( (handle->datatype_in == LIBXS_DNN_DATATYPE_I16) || (handle->datatype_in == LIBXS_DNN_DATATYPE_I8) ) {
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(3*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(3*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 3;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->blocksofm;
            }
#endif
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#if 0
        } else if ((handle->buffer_format & LIBXS_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( handle->datatype_in == LIBXS_DNN_DATATYPE_F32 ) {
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(1*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 1;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock*handle->blocksofm;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#endif
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXS_DNN_BATCH_STATS) ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_BATCH_STATS;
#ifdef FP64_BN_STATS     
        layout->datatype = LIBXS_DNN_DATATYPE_F64; 
#endif

        if ((handle->buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->datatype_out;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
              layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_X;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->desc.N;
              layout->dim_size[2] = handle->blocksofm;
              layout->dim_size[3] = 2;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if (type == LIBXS_DNN_MAX_STATS_FWD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_MAX_STATS_FWD;
        layout->datatype = LIBXS_DNN_DATATYPE_F32;
        layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else if (type == LIBXS_DNN_MAX_STATS_BWD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_MAX_STATS_BWD;
        layout->datatype = LIBXS_DNN_DATATYPE_F32;
        layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else if (type == LIBXS_DNN_MAX_STATS_UPD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_MAX_STATS_UPD;
        layout->datatype = LIBXS_DNN_DATATYPE_F32;
        layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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


LIBXS_API_DEFINITION libxs_dnn_tensor_datalayout* libxs_dnn_duplicate_tensor_datalayout(const libxs_dnn_tensor_datalayout* layout, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* dst_layout;

  *status = LIBXS_DNN_SUCCESS;
  dst_layout = 0;

  if (layout != 0 && layout->num_dims != 0) {
    unsigned int dim = 0;

    dst_layout = (libxs_dnn_tensor_datalayout*)malloc(sizeof(libxs_dnn_tensor_datalayout));
    if (0 != dst_layout) {
      memset(dst_layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      dst_layout->dim_type = (libxs_dnn_tensor_dimtype*)malloc(layout->num_dims * sizeof(libxs_dnn_tensor_dimtype));
      dst_layout->dim_size = (unsigned int*)malloc(layout->num_dims * sizeof(unsigned int));
      dst_layout->num_dims = layout->num_dims;
      dst_layout->format = layout->format;
      dst_layout->custom_format = layout->custom_format;
      dst_layout->datatype = layout->datatype;
      dst_layout->tensor_type = layout->tensor_type;
      if (0 != dst_layout->dim_type && 0 != dst_layout->dim_size) {
        for (dim = 0; dim < layout->num_dims; ++dim) {
          dst_layout->dim_type[dim] = layout->dim_type[dim];
          dst_layout->dim_size[dim] = layout->dim_size[dim];
        }
      } else {
        *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
    }
  } else {
    *status = LIBXS_DNN_ERR_INVALID_LAYOUT;
  }

  return dst_layout;
}


LIBXS_API_DEFINITION unsigned int libxs_dnn_compare_tensor_datalayout(const libxs_dnn_tensor_datalayout* layout_a, const libxs_dnn_tensor_datalayout* layout_b, libxs_dnn_err_t* status) {
  unsigned int result = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (layout_a != 0 && layout_b != 0) {
    unsigned int dim = 0;

    if (layout_a->num_dims      != layout_b->num_dims)      { result = 1; }
    if (layout_a->format        != layout_b->format)        { result = 1; }
    if (layout_a->custom_format != layout_b->custom_format) { result = 1; }
    if (layout_a->datatype      != layout_b->datatype)      { result = 1; }

    if (result == 0) {
      for ( dim = 0; dim < layout_a->num_dims; ++dim ) {
        if ( layout_a->dim_type[dim] != layout_b->dim_type[dim] ) { result = 1; }
        if ( layout_a->dim_size[dim] != layout_b->dim_size[dim] ) { result = 1; }
      }
    }
  } else {
    *status = LIBXS_DNN_ERR_INVALID_LAYOUT;
    result = 100;
  }

  return result;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_tensor_datalayout(libxs_dnn_tensor_datalayout* layout) {
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


LIBXS_API_DEFINITION unsigned int libxs_dnn_get_tensor_size(const libxs_dnn_tensor_datalayout* layout, libxs_dnn_err_t* status) {
  unsigned int size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != layout) {
    unsigned int dim = 0;
    size = (unsigned int)libxs_dnn_typesize(layout->datatype);
    for (dim = 0; dim < layout->num_dims; ++dim) {
      size *= layout->dim_size[dim];
    }
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_LAYOUT;
  }

  return size;
}


LIBXS_API_DEFINITION unsigned int libxs_dnn_get_tensor_elements(const libxs_dnn_tensor_datalayout* layout, libxs_dnn_err_t* status) {
  unsigned int elements = 1;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != layout) {
    unsigned int dim = 0;
    for ( dim = 0; dim < layout->num_dims; ++dim ) {
      elements *= layout->dim_size[dim];
    }
  } else {
    *status = LIBXS_DNN_ERR_INVALID_LAYOUT;
    elements = 0;
  }

  return elements;
}


LIBXS_API_DEFINITION void* libxs_dnn_get_tensor_data_ptr(const libxs_dnn_tensor* tensor, libxs_dnn_err_t* status)
{
  *status = LIBXS_DNN_SUCCESS;

  if (0 != tensor) {
    return tensor->data;
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }

  return 0;
}


LIBXS_API_DEFINITION char libxs_dnn_get_qtensor_exp(const libxs_dnn_tensor* tensor, libxs_dnn_err_t* status)
{
  *status = LIBXS_DNN_SUCCESS;

  if (0 != tensor) {
    return tensor->exp;
  }
  else {
    *status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }

  return 0;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_set_qtensor_exp(libxs_dnn_tensor* tensor, const char exp)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != tensor) {
    tensor->exp = exp;
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_destroy_tensor(const libxs_dnn_tensor* tensor)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != tensor) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_tensor*)tensor);
  }
#if 0 /* releasing a NULL-buffer should be not an error (similar to freeing a NULL pointer) */
  else {
    status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }
#endif
  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyin_tensor(const libxs_dnn_tensor* tensor, const void* data, const libxs_dnn_tensor_format in_format)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* @TODO check for valid combination */
  if (0 != tensor) {
    switch (tensor->layout->tensor_type) {
      case LIBXS_DNN_REGULAR_INPUT:
      case LIBXS_DNN_GRADIENT_INPUT:
      case LIBXS_DNN_REGULAR_OUTPUT:
      case LIBXS_DNN_GRADIENT_OUTPUT:
      case LIBXS_DNN_INPUT:
      case LIBXS_DNN_OUTPUT:
      case LIBXS_DNN_ACTIVATION: {
                                     switch (in_format) {
                                       case LIBXS_DNN_TENSOR_FORMAT_NCHW: {
                                                                              if ( (tensor->layout->format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0 ) {
                                                                                switch (tensor->layout->datatype) {
                                                                                  case LIBXS_DNN_DATATYPE_F32: {
                                                                                                                   typedef float element_type;
#include "template/libxs_dnn_tensor_buffer_copy_in_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXS_DNN_DATATYPE_I32: {
                                                                                                                   typedef int element_type;
#include "template/libxs_dnn_tensor_buffer_copy_in_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXS_DNN_DATATYPE_I16: {
                                                                                                                   typedef short  element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_buffer_copy_in_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
                                                                                                                 } break;
                                                                                  case LIBXS_DNN_DATATYPE_I8: {
                                                                                                                  typedef char element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_buffer_copy_in_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
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
                                   } break;
      case LIBXS_DNN_REGULAR_FILTER:
      case LIBXS_DNN_GRADIENT_FILTER:
      case LIBXS_DNN_FILTER: {
                                 switch (in_format) {
                                   case LIBXS_DNN_TENSOR_FORMAT_KCRS: {
                                                                          if ( (tensor->layout->format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0 ) {
                                                                            switch (tensor->layout->datatype) {
                                                                              case LIBXS_DNN_DATATYPE_F32: {
                                                                                                               typedef float element_type;
#include "template/libxs_dnn_tensor_filter_copy_in_kcrs.tpl.c"
                                                                                                             } break;
                                                                              case LIBXS_DNN_DATATYPE_I16: {
                                                                                                               typedef short element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_filter_copy_in_kcrs.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
                                                                                                             } break;
                                                                              case LIBXS_DNN_DATATYPE_I8: {
                                                                                                              typedef char element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_filter_copy_in_kcrs.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
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
                               } break;
      case LIBXS_DNN_REGULAR_BIAS:
      case LIBXS_DNN_GRADIENT_BIAS:
      case LIBXS_DNN_BIAS: {
                               switch (in_format) {
                                 case LIBXS_DNN_TENSOR_FORMAT_NCHW: {
                                                                        if ( (tensor->layout->format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0 ) {
                                                                          switch (tensor->layout->datatype) {
                                                                            case LIBXS_DNN_DATATYPE_F32: {
                                                                                                             typedef float element_type;
#include "template/libxs_dnn_tensor_bias_copy_in_nchw.tpl.c"
                                                                                                           } break;
                                                                            case LIBXS_DNN_DATATYPE_I16: {
                                                                                                             typedef short element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_bias_copy_in_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
                                                                                                           } break;
                                                                            case LIBXS_DNN_DATATYPE_I8: {
                                                                                                            typedef char element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_bias_copy_in_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
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
                             } break;
      default: {
                 status = LIBXS_DNN_ERR_INVALID_TENSOR;
               }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_zero_tensor(const libxs_dnn_tensor* tensor)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != tensor) {
    const size_t size = libxs_dnn_get_tensor_elements( tensor->layout, &status );
    size_t i;
    /* use for-loops to potentially leverage NUMA in the future */
    switch (tensor->layout->datatype) {
      case LIBXS_DNN_DATATYPE_F32: {
                                       float* fp32_data = (float*)tensor->data;
                                       for (i = 0; i < size; ++i) fp32_data[i] = 0.0f;
                                     } break;
      case LIBXS_DNN_DATATYPE_I32: {
                                       int* int32_data = (int*)tensor->data;
                                       for (i = 0; i < size; ++i) int32_data[i] = 0;
                                     } break;
      case LIBXS_DNN_DATATYPE_I16: {
                                       short* int16_data = (short*)tensor->data;
                                       for (i = 0; i < size; ++i) int16_data[i] = 0;
                                     } break;
      case LIBXS_DNN_DATATYPE_I8: {
                                      char* int8_data = (char*)tensor->data;
                                      for (i = 0; i < size; ++i) int8_data[i] = 0;
                                    } break;
      default: {
                 status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
               }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_copyout_tensor(const libxs_dnn_tensor* tensor, void* data, const libxs_dnn_tensor_format out_format)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* @TODO check for valid combination */
  if (0 != tensor) {
    switch (tensor->layout->tensor_type) {
      case LIBXS_DNN_REGULAR_INPUT:
      case LIBXS_DNN_GRADIENT_INPUT:
      case LIBXS_DNN_REGULAR_OUTPUT:
      case LIBXS_DNN_GRADIENT_OUTPUT:
      case LIBXS_DNN_INPUT:
      case LIBXS_DNN_OUTPUT:
      case LIBXS_DNN_ACTIVATION: {
                                     switch (out_format) {
                                       case LIBXS_DNN_TENSOR_FORMAT_NCHW: {
                                                                              if ( (tensor->layout->format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0 ) {
                                                                                switch (tensor->layout->datatype) {
                                                                                  case LIBXS_DNN_DATATYPE_F32: {
                                                                                                                   typedef float element_type;
#include "template/libxs_dnn_tensor_buffer_copy_out_nchw.tpl.c"
                                                                                                                 } break;
                                                                                  case LIBXS_DNN_DATATYPE_I32: {
                                                                                                                   typedef int element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION                
#include "template/libxs_dnn_tensor_buffer_copy_out_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION                 
                                                                                                                 } break;
                                                                                  case LIBXS_DNN_DATATYPE_I16: {
                                                                                                                   typedef short element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_buffer_copy_out_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
                                                                                                                 } break;
                                                                                  case LIBXS_DNN_DATATYPE_I8: {
                                                                                                                  typedef char element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_buffer_copy_out_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
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
                                   } break;
      case LIBXS_DNN_REGULAR_FILTER:
      case LIBXS_DNN_GRADIENT_FILTER:
      case LIBXS_DNN_FILTER: {
                                 switch (out_format) {
                                   case LIBXS_DNN_TENSOR_FORMAT_KCRS: {
                                                                          if ( (tensor->layout->format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0 ) {
                                                                            switch (tensor->layout->datatype) {
                                                                              case LIBXS_DNN_DATATYPE_F32: {
                                                                                                               typedef float element_type;
#include "template/libxs_dnn_tensor_filter_copy_out_kcrs.tpl.c"
                                                                                                             } break;
                                                                              case LIBXS_DNN_DATATYPE_I16: {
                                                                                                               typedef short  element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_filter_copy_out_kcrs.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
                                                                                                             } break;
                                                                              case LIBXS_DNN_DATATYPE_I8: {
                                                                                                              typedef char element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_filter_copy_out_kcrs.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
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
                               } break;
      case LIBXS_DNN_REGULAR_BIAS:
      case LIBXS_DNN_GRADIENT_BIAS:
      case LIBXS_DNN_BIAS: {
                               switch (out_format) {
                                 case LIBXS_DNN_TENSOR_FORMAT_NCHW: {
                                                                        if ( (tensor->layout->format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0 ) {
                                                                          switch (tensor->layout->datatype) {
                                                                            case LIBXS_DNN_DATATYPE_F32: {
                                                                                                             typedef float element_type;
#include "template/libxs_dnn_tensor_bias_copy_out_nchw.tpl.c"
                                                                                                           } break;
                                                                            case LIBXS_DNN_DATATYPE_I16: {
                                                                                                             typedef short element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_bias_copy_out_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
                                                                                                           } break;
                                                                            case LIBXS_DNN_DATATYPE_I8: {
                                                                                                            typedef char element_type;
#define LIBXS_DNN_COPY_LOW_PRECISION
#include "template/libxs_dnn_tensor_bias_copy_out_nchw.tpl.c"
#undef LIBXS_DNN_COPY_LOW_PRECISION
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
                             } break;
      default: {
                 status = LIBXS_DNN_ERR_INVALID_TENSOR;
               }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_TENSOR;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_trans_reg_filter(const libxs_dnn_layer* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0) {
    if ( (handle->reg_filter != 0) && (handle->reg_filter_tr != 0) ) {
      /* TODO handle more datatypes */
      int ifm1, ifm2, kj, ki, ofm1, ofm2;
      LIBXS_VLA_DECL(6, float, wt, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      LIBXS_VLA_DECL(6, float, tr_wt, (float*)handle->reg_filter_tr->data, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

      /* TODO we might want to do this in parallel.... */
      for ( ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1 ) {
        for ( ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1 ) {
          for (kj=0; kj < handle->desc.R; ++kj) {
            for (ki=0; ki < handle->desc.S; ++ki) {
              for ( ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2 ) {
                for ( ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2 ) {
                  LIBXS_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2, ifm2, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                    LIBXS_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                }
              }
            }
          }
        }
      }
    } else {
      status = LIBXS_DNN_ERR_INVALID_TENSOR;
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_bind_tensor(libxs_dnn_layer* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)        && (type != LIBXS_DNN_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_REGULAR_OUTPUT)       && (type != LIBXS_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXS_DNN_REGULAR_FILTER)       && (type != LIBXS_DNN_GRADIENT_FILTER) &&
      (type != LIBXS_DNN_REGULAR_BIAS)         && (type != LIBXS_DNN_GRADIENT_BIAS)   &&
      (type != LIBXS_DNN_REGULAR_FILTER_TRANS) && (type != LIBXS_DNN_BATCH_STATS) && (type != LIBXS_DNN_MAX_STATS_FWD) && (type != LIBXS_DNN_MAX_STATS_BWD)  && (type != LIBXS_DNN_MAX_STATS_UPD)  ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_FILTER ) {
        handle->grad_filter = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_BIAS ) {
        handle->reg_bias = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRADIENT_BIAS ) {
        handle->grad_bias = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_REGULAR_FILTER_TRANS ) {
        handle->reg_filter_tr = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_BATCH_STATS ) {
        handle->batch_stats = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_MAX_STATS_FWD ) {
        handle->maxstats_fwd = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_MAX_STATS_BWD ) {
        handle->maxstats_bwd = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_MAX_STATS_UPD ) {
        handle->maxstats_upd = (libxs_dnn_tensor*)tensor;
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXS_DNN_ERR_MISMATCH_TENSOR;
    }

    libxs_dnn_destroy_tensor_datalayout( handle_layout );
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_release_tensor(libxs_dnn_layer* handle, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_REGULAR_INPUT)        && (type != LIBXS_DNN_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_REGULAR_OUTPUT)       && (type != LIBXS_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXS_DNN_REGULAR_FILTER)       && (type != LIBXS_DNN_GRADIENT_FILTER) &&
      (type != LIBXS_DNN_REGULAR_BIAS)         && (type != LIBXS_DNN_GRADIENT_BIAS)   &&
      (type != LIBXS_DNN_REGULAR_FILTER_TRANS) && (type != LIBXS_DNN_BATCH_STATS)        ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_REGULAR_INPUT ) {
      handle->reg_input = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_INPUT ) {
      handle->grad_input = 0;
    } else if ( type == LIBXS_DNN_REGULAR_OUTPUT ) {
      handle->reg_output = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_OUTPUT ) {
      handle->grad_output = 0;
    } else if ( type == LIBXS_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else if ( type == LIBXS_DNN_REGULAR_BIAS ) {
      handle->reg_bias = 0;
    } else if ( type == LIBXS_DNN_GRADIENT_BIAS ) {
      handle->grad_bias = 0;
    } else if ( type == LIBXS_DNN_REGULAR_FILTER_TRANS ) {
      handle->reg_filter_tr = 0;
    } else if ( type == LIBXS_DNN_BATCH_STATS ) {
      handle->batch_stats = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API_DEFINITION size_t libxs_dnn_get_scratch_size(const libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  size_t l_scratch_size = 0;
  size_t scratch5_size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    if (handle->algo == LIBXS_DNN_CONV_ALGO_WINOGRAD) {
      l_scratch_size = 0;
      l_scratch_size += handle->scratch1_size + 64;
      l_scratch_size += handle->scratch3_size + 64;
      l_scratch_size += handle->scratch4_size + 64;
      l_scratch_size += handle->scratchIw_size + 64;
      l_scratch_size += handle->scratchOw_size + 64;
      if (libxs_target_archid == LIBXS_X86_AVX512_KNM) {
        l_scratch_size += handle->scratchVk_size + 64;
      }
    } else {
      switch (kind) {
        case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->fwdbwd_scratch_size;
                                               l_scratch_size = scratch5_size + 64;
                                             }
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                             /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             l_scratch_size = handle->scratch1_size + 64;
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->fwdbwd_scratch_size;
                                               l_scratch_size += scratch5_size + 64;
                                             }
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             l_scratch_size += handle->scratch3_size + 64;
                                             /* potentially we need thread-local filter copies, scratch4 */
                                             if (handle->upd_use_thread_fil == 1) {
                                               l_scratch_size += handle->scratch4_size + 64;
                                             }
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->minibatch_scratch_size;
                                               l_scratch_size += scratch5_size + 64;
                                             }
                                             if (handle->use_lp_kernel == 1) {
                                              l_scratch_size += handle->scratch6_size +64;
                                             }
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                             /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             l_scratch_size += handle->scratch1_size + 64;
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             l_scratch_size += handle->scratch3_size + 64;
                                             /* potentially we need thread-local filter copies, scratch4 */
                                             if (handle->upd_use_thread_fil == 1) {
                                               l_scratch_size += handle->scratch4_size + 64;
                                             }
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->max_scratch5_size;
                                               l_scratch_size += scratch5_size + 64;
                                             }
                                             if (handle->use_lp_kernel == 1) {
                                              l_scratch_size += handle->scratch6_size +64;
                                             }
                                           } break;
        default: {
                   *status = LIBXS_DNN_ERR_INVALID_KIND;
                 }
      }
    }
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_bind_scratch(libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind, const void* scratch)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  size_t address = (size_t)scratch;
  size_t offset = 0;
  size_t scratch5_size = 0;

  if (scratch == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    /* check this if, this is bogus, not sure why there */
#if 0
    if ( (kind == LIBXS_DNN_COMPUTE_KIND_FWD) && (handle->datatype_in == handle->datatype_out) ) {
      status = LIBXS_DNN_SUCCESS;
    }
#endif
    return status;
  }

  if (0 != handle) {
    if (handle->algo == LIBXS_DNN_CONV_ALGO_WINOGRAD) {
      /* + 64 to do alignment while performing bind, scratch1 */
      if (address % 64 == 0) {
        handle->scratch1 = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratch1 = (void*)(address+offset);
      }
      address += handle->scratch1_size + 64;
      if (address % 64 == 0) {
        handle->scratch3 = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratch3 = (void*)(address+offset);
      }
      address += handle->scratch3_size + 64;
      if (address % 64 == 0) {
        handle->scratch4 = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratch4 = (void*)(address+offset);
      }
      address += handle->scratch4_size + 64;
      if (address % 64 == 0) {
        handle->scratchIw = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratchIw = (void*)(address+offset);
      }
      address += handle->scratchIw_size + 64;
      if (address % 64 == 0) {
        handle->scratchOw = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratchOw = (void*)(address+offset);
      }
      address += handle->scratchOw_size + 64;
      if ( libxs_target_archid == LIBXS_X86_AVX512_KNM ) {
        if (address % 64 == 0) {
          handle->scratchVk = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratchVk = (void*)(address+offset);
        }
        address += handle->scratchVk_size + 64;
      }
    } else {
      switch (kind) {
        case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->fwdbwd_scratch_size;;
                                               if (address % 64 == 0) {
                                                 handle->scratch5 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch5 = (void*)(address+offset);
                                               }
                                               address += scratch5_size + 64;
                                             }
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                             /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             if (address % 64 == 0) {
                                               handle->scratch1 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch1 = (void*)(address+offset);
                                             }
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->fwdbwd_scratch_size;;
                                               address += handle->scratch1_size + 64;
                                               if (address % 64 == 0) {
                                                 handle->scratch5 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch5 = (void*)(address+offset);
                                               }
                                             }
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->minibatch_scratch_size;
                                               if (address % 64 == 0) {
                                                 handle->scratch5 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch5 = (void*)(address+offset);
                                               }
                                               address += scratch5_size + 64;
                                             }
                                             if (address % 64 == 0) {
                                               handle->scratch3 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch3 = (void*)(address+offset);
                                             }
                                             /* potentially we need thread-local filter copies, scratch4 */
                                             if (handle->upd_use_thread_fil == 1) {
                                               address += handle->scratch3_size + 64;
                                               if (address % 64 == 0) {
                                                 handle->scratch4 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch4 = (void*)(address+offset);
                                               }
                                               address += handle->scratch4_size + 64;
                                             }
                                             if (handle->use_lp_kernel == 1) {
                                               if (address % 64 == 0) {
                                                 handle->scratch6 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch6 = (void*)(address+offset);
                                               }
                                               address += handle->scratch6_size + 64;
                                             }
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                             /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             if (handle->padding_flag == 1) {
                                               scratch5_size = handle->max_scratch5_size;
                                               if (address % 64 == 0) {
                                                 handle->scratch5 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch5 = (void*)(address+offset);
                                               }
                                               address += scratch5_size + 64;
                                             }
                                             if (address % 64 == 0) {
                                               handle->scratch1 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch1 = (void*)(address+offset);
                                             }
                                             address += handle->scratch1_size + 64;
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             if (address % 64 == 0) {
                                               handle->scratch3 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch3 = (void*)(address+offset);
                                             }
                                             address += handle->scratch3_size + 64;
                                             /* potentially we need thread-local filter copies, scratch4 */
                                             if (handle->upd_use_thread_fil == 1) {
                                               if (address % 64 == 0) {
                                                 handle->scratch4 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch4 = (void*)(address+offset);
                                               }
                                               address += handle->scratch4_size + 64;
                                             }
                                             if (handle->use_lp_kernel == 1) {
                                               if (address % 64 == 0) {
                                                 handle->scratch6 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch6 = (void*)(address+offset);
                                               }
                                               address += handle->scratch6_size + 64;
                                             }
                                           } break;
        default: {
                   status = LIBXS_DNN_ERR_INVALID_KIND;
                 }
      }
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_release_scratch(libxs_dnn_layer* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    if (handle->algo == LIBXS_DNN_CONV_ALGO_WINOGRAD) {
      handle->scratch1 = 0;
      handle->scratch3 = 0;
      handle->scratch4 = 0;
      handle->scratchIw = 0;
      handle->scratchOw = 0;
      handle->scratchVk = 0;
    } else {
      switch (kind) {
        case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                             handle->scratch5 = 0;
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                             handle->scratch1 = 0;
                                             handle->scratch5 = 0;
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                             handle->scratch3 = 0;
                                             handle->scratch4 = 0;
                                             handle->scratch5 = 0;
                                             handle->scratch6 = 0;
                                           } break;
        case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                             handle->scratch1 = 0;
                                             handle->scratch3 = 0;
                                             handle->scratch4 = 0;
                                             handle->scratch5 = 0;
                                             handle->scratch6 = 0;
                                           } break;
        default: {
                   status = LIBXS_DNN_ERR_INVALID_KIND;
                 }
      }
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_INLINE libxs_dnn_err_t internal_execute_st(libxs_dnn_layer* handle,
    libxs_dnn_compute_kind kind, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (handle->algo) {
      case LIBXS_DNN_CONV_ALGO_DIRECT: {
                                           switch (kind) {
                                             case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                                                                  switch (handle->buffer_format) {
                                                                                    case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                              switch (handle->filter_format) {
                                                                                                                                case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                          status = libxs_dnn_convolve_st_fwd_custom_custom(handle, start_thread, tid);
                                                                                                                                                                        } break;
                                                                                                                                default: {
                                                                                                                                           status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                                                                                                                                         }
                                                                                                                              }
                                                                                                                            } break;
                                                                                    case LIBXS_DNN_TENSOR_FORMAT_NHWC: {
                                                                                                                           switch (handle->filter_format) {
                                                                                                                             case LIBXS_DNN_TENSOR_FORMAT_RSCK: {
                                                                                                                                                                    status = libxs_dnn_convolve_st_fwd_nhwc_rsck(handle, start_thread, tid);
                                                                                                                                                                  } break;
                                                                                                                             case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
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
                                             case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                                                                  switch (handle->buffer_format) {
                                                                                    case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                              switch (handle->filter_format) {
                                                                                                                                case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                          status = libxs_dnn_convolve_st_bwd_custom_custom(handle, start_thread, tid);
                                                                                                                                                                        } break;
                                                                                                                                default: {
                                                                                                                                           status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                                                                                                                                         }
                                                                                                                              }
                                                                                                                            } break;
                                                                                    case LIBXS_DNN_TENSOR_FORMAT_NHWC: {
                                                                                                                           switch (handle->filter_format) {
                                                                                                                             case LIBXS_DNN_TENSOR_FORMAT_RSCK: {
                                                                                                                                                                    status = libxs_dnn_convolve_st_bwd_nhwc_rsck(handle, start_thread, tid);
                                                                                                                                                                  } break;
                                                                                                                             case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
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
                                             case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                                                                  switch (handle->buffer_format) {
                                                                                    case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                              switch (handle->filter_format) {
                                                                                                                                case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                          status = libxs_dnn_convolve_st_upd_custom_custom(handle, start_thread, tid);
                                                                                                                                                                        } break;
                                                                                                                                default: {
                                                                                                                                           status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                                                                                                                                         }
                                                                                                                              }
                                                                                                                            } break;
                                                                                    case LIBXS_DNN_TENSOR_FORMAT_NHWC: {
                                                                                                                           switch (handle->filter_format) {
                                                                                                                             case LIBXS_DNN_TENSOR_FORMAT_RSCK: {
                                                                                                                                                                    status = libxs_dnn_convolve_st_upd_nhwc_rsck(handle, start_thread, tid);
                                                                                                                                                                  } break;
                                                                                                                             case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
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
                                         } break;
      case LIBXS_DNN_CONV_ALGO_WINOGRAD: {
                                             switch (kind) {
                                               case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                                                                    switch (handle->buffer_format) {
                                                                                      case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                switch (handle->filter_format) {
                                                                                                                                  case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                            status = libxs_dnn_convolve_winograd_st_fwd_custom_custom(handle, start_thread, tid);
                                                                                                                                                                          } break;
                                                                                                                                  default: {
                                                                                                                                             status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                                                                                                                                           }
                                                                                                                                }
                                                                                                                              } break;
                                                                                      case LIBXS_DNN_TENSOR_FORMAT_NHWC: {
                                                                                                                             switch (handle->filter_format) {
                                                                                                                               case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                         status = libxs_dnn_convolve_winograd_st_fwd_nhwc_custom(handle, start_thread, tid);
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
                                               case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                                                                    switch (handle->buffer_format) {
                                                                                      case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                switch (handle->filter_format) {
                                                                                                                                  case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                            status = libxs_dnn_convolve_winograd_st_bwd_custom_custom(handle, start_thread, tid);
                                                                                                                                                                          } break;
                                                                                                                                  default: {
                                                                                                                                             status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                                                                                                                                           }
                                                                                                                                }
                                                                                                                              } break;
                                                                                      case LIBXS_DNN_TENSOR_FORMAT_NHWC: {
                                                                                                                             switch (handle->filter_format) {
                                                                                                                               case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                         status = libxs_dnn_convolve_winograd_st_bwd_nhwc_custom(handle, start_thread, tid);
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
                                               case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                                                                    switch (handle->buffer_format) {
                                                                                      case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                switch (handle->filter_format) {
                                                                                                                                  case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                            status = libxs_dnn_convolve_winograd_st_upd_custom_custom(handle, start_thread, tid);
                                                                                                                                                                          } break;
                                                                                                                                  default: {
                                                                                                                                             status = LIBXS_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                                                                                                                                           }
                                                                                                                                }
                                                                                                                              } break;
                                                                                      case LIBXS_DNN_TENSOR_FORMAT_NHWC: {
                                                                                                                             switch (handle->filter_format) {
                                                                                                                               case LIBXS_DNN_TENSOR_FORMAT_LIBXS: {
                                                                                                                                                                         status = libxs_dnn_convolve_winograd_st_upd_nhwc_custom(handle, start_thread, tid);
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
                                           } break;
      default: {
                 status = LIBXS_DNN_ERR_INVALID_ALGO;
               }
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_execute_st(libxs_dnn_layer* handle,
    libxs_dnn_compute_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  return internal_execute_st(handle, kind, start_thread, tid);
}


LIBXS_API_DEFINITION void libxs_dnn_execute(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind)
{
#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
  {
    const int tid = omp_get_thread_num();
    internal_execute_st(handle, kind, 0, tid);
  }
#else
  internal_execute_st(handle, kind, 0/*start_thread*/, 0/*tid*/);
#endif
}


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_transpose_filter(libxs_dnn_layer* handle, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int ofm1, ifm1, kj, ki, ifm2, ofm2;

  /* check for filter type */
  if ( (type != LIBXS_DNN_REGULAR_FILTER) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->reg_filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have scratch */
  if (handle->scratch1 == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  /* check that filter is in RSCK storage */
  if ( (handle->filter_format & LIBXS_DNN_TENSOR_FORMAT_RSCK) == 0 ) {
    status = LIBXS_DNN_ERR_MISMATCH_TENSOR;
    return status;
  }

  /* check that we are in FP32 */
  if ( handle->datatype_in == LIBXS_DNN_DATATYPE_F32 ) {
    LIBXS_VLA_DECL(6, float, wt, (float*)handle->reg_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
    LIBXS_VLA_DECL(6, float, tr_wt, (float*)handle->scratch1, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
        for (kj=0; kj < handle->desc.R; ++kj) {
          for (ki=0; ki < handle->desc.S; ++ki) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXS_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kj, ki, ofm2, ifm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
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


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_reduce_wu_filters(libxs_dnn_layer* handle, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  int i, j, filter_size;

  /* check for filter type */
  if ( (type != LIBXS_DNN_GRADIENT_FILTER) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->grad_filter == 0) {
    status = LIBXS_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check that we are in FP32 */
  if (handle->datatype_out == LIBXS_DNN_DATATYPE_F32 ) {
    if (handle->upd_use_external_reduce != 0) {
      float* filter_ptr = (float*)handle->grad_filter->data;
      /* calculate filter size */
      filter_size = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * handle->ofmblock * handle->ifmblock;

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


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_get_codegen_success(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           if (handle->code_fwd[0].xconv.sconv == 0) {
                                             status = LIBXS_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                           if (handle->code_bwd[0].xconv.sconv == 0) {
                                             status = LIBXS_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_UPD: {
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


LIBXS_API_DEFINITION libxs_dnn_err_t libxs_dnn_get_parallel_tasks(libxs_dnn_layer* handle, libxs_dnn_compute_kind kind, unsigned int* num_tasks) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           *num_tasks = handle->desc.N * handle->blocksofm;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                           *num_tasks = handle->desc.N * handle->blocksifm;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_UPD: {
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


/* @TODO make this routine aware of any int type */
LIBXS_API_DEFINITION void libxs_dnn_quantize( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode ) {
  int i = 0;
  libxs_intfloat value;
  libxs_intfloat exp;
  unsigned int qvalue = 0;
  unsigned int mant = 0;
  unsigned int sign = 0;
  unsigned char rhs = 0;
  unsigned char exp_off = 0;
  unsigned char max_exp = 0;
  float absmax_value = 0;

  /* finding absmax float for largest exp */
  absmax_value = (float)fabs((double)(in_buffer[0]));
  for( i = 1; i < length; ++i ) {
    if ((float)fabs((double)(in_buffer[i])) > absmax_value) {
      absmax_value = (float)fabs((double)(in_buffer[i]));
    }
  }
  /* bit-wise conversion to int */
  exp.f = absmax_value;
  /* shift by mantissa to the right and convert to char */
  max_exp = (unsigned char)((exp.ui & LIBXS_DNN_MASK_ABS_F32) >> LIBXS_DNN_MANT_SZ_F32);

  /* if we go for stochstic rounding, let's intialize random seed */
  if ( round_mode == LIBXS_DNN_QUANT_STOCH_ROUND ) {
    srand( time(NULL) );
  }

  for( i = 0; i < length; ++i ) {
    value.f = in_buffer[i];
    /* in case of zero we don't need to do anything */
    if (in_buffer[i] == 0.0f) {
      qvalue = 0;
    } else {
      /* let's get a float copy to work on */
      /* vinp = _mm512_load_ps( in_buffer[i] ); */
      value.f = in_buffer[i];
      /* let's compute the offset of the current exp at pos i from max offset, we need to mask the sign bit though */
      /*__m512i vexp     = _mm512_cvtps_epi32(_mm512_getexp_ps (vinp));
        __m512i vexp_off = _mm512_sub_epi32(maxexpf, vexp);*/
      exp_off = max_exp - (unsigned char)((value.ui & LIBXS_DNN_MASK_ABS_F32) >> LIBXS_DNN_MANT_SZ_F32);
      /* cut out mantissa and set leading bit */
      /*__m512i mmask = _mm512_set1_epi32(LIBXS_DNN_MASK_MANT_F32);
        __m512i vmant = _mm512_or_epi32(_mm512_set1_epi32(0x1 << LIBXS_DNN_MANT_SZ_F32), _mm512_and_epi32( _mm512_castps_si512( vinp ), mmask));*/
      mant = ((0x1 << LIBXS_DNN_MANT_SZ_F32) | (value.ui & LIBXS_DNN_MASK_MANT_F32));
      /* extract sign */
      /* __mmask16 smask =  _mm512_cmplt_ps_mask (inp, _mm512_set1_ps(0)); */
      sign = (value.ui & LIBXSNN_DNN_MASK_SIGN_F32) >> (LIBXS_DNN_SZ_F32-1);
      /* caclulate rhs, be aware of the now explicit leading bit, @TODO add DFP8/4 */
      rhs = (unsigned char)(LIBXS_DNN_MANT_SZ_F32+1) - LIBXS_DNN_MANT_DFP16 + exp_off + add_shift;
      /* some safety, to generate 0 when we fall off quant region, @TODO issue a LIBXS Warning that we shifted out the entire mantissa */
      if (rhs > (LIBXS_DNN_MANT_SZ_F32+1)) { 
        rhs = (LIBXS_DNN_MANT_SZ_F32+1);
      }
      /* finally shfit the value into the region we need, this is now a 15-add_rhs bit number for the max value in in_buffer */
      qvalue = (mant >> rhs);
      /* handle sign, 2 complement */
      if ( sign > 0 && qvalue > 0 ) {
        qvalue = (~qvalue + 1);
      }
      
      if (round_mode == LIBXS_DNN_QUANT_BIAS_ROUND) {
        /* biased rounding towards next bigger number */
        /* first let's determine in the original number if we need a bias rounding, @TODO need fix for F64 */
        int bias_needed = (mant & (0x3 << rhs));
        /* apply bias */
        if (bias_needed > 0) {
          qvalue++;
        }
      } else if (round_mode == LIBXS_DNN_QUANT_STOCH_ROUND) {
        /* stochastic rounding, as implemented in the IBM paper from 2015, @TODO, fix F64 and DFP8 */ 
        float p, q;
        libxs_intfloat fvalue;
        float eps = (float)LIXSMMM_DNN_RES_DFP16;
        /* masking all bits which will be shifted out */
        fvalue.ui = value.ui & ((LIBXS_DNN_MASK_FULL_F32) << rhs);
        /* drawing a random number */
        float r = (float)fabs((double)rand());
        p = r/((float)RAND_MAX);
        q = (in_buffer[i] - fvalue.f)/eps;
        /* apply rounding if needed */
        if (p+q > 0.5f) {
          qvalue++;
        }
      } else {
        /* do nothing about rounding, just chop */
      }
    }
    out_buffer[i] = (short)qvalue;
  }

  *scf = 14-add_shift-(max_exp-127);
}


LIBXS_API_DEFINITION void libxs_dnn_dequantize( short* in_buffer, float* out_buffer, int length, unsigned char scf ) {
  int i = 0;
  float exp = pow(2.0, -scf);
  
  for ( i = 0 ; i < length; ++i ) {
    out_buffer[i] = ((float)in_buffer[i])*exp;
  }
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
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid descriptor (forward convolution)!\n");
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
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid descriptor (backward convolution)!\n");
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
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid convolution descriptor (weight update)!\n");
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
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid descriptor (forward convolution)!\n");
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
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid descriptor (backward convolution)!\n");
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
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid convolution descriptor (weight update)!\n");
      }
    }
#endif
  return code.pmm;
}


LIBXS_API_DEFINITION void* libxs_create_xconv_wino_forward(
    const libxs_convolution_winograd_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
    if (0 != descriptor) {
      libxs_build_request request;
      request.descriptor.cwino = descriptor;
      request.kind = LIBXS_BUILD_KIND_CWFWD;
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid descriptor (forward convolution)!\n");
      }
    }
#endif
  return code.pmm;
}


LIBXS_API_DEFINITION void* libxs_create_xconv_wino_backward(
    const libxs_convolution_winograd_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
    if (0 != descriptor) {
      libxs_build_request request;
      request.descriptor.cwino = descriptor;
      request.kind = LIBXS_BUILD_KIND_CWBWD;
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid descriptor (backward convolution)!\n");
      }
    }
#endif
  return code.pmm;
}


LIBXS_API_DEFINITION void* libxs_create_xconv_wino_update_weights(
    const libxs_convolution_winograd_descriptor* descriptor)
{
  libxs_code_pointer code = { 0 };
  LIBXS_INIT
    if (0 != descriptor) {
      libxs_build_request request;
      request.descriptor.cwino = descriptor;
      request.kind = LIBXS_BUILD_KIND_CWUPD;
      libxs_build(&request, LIBXS_CAPACITY_REGISTRY/*not managed*/, &code);
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else {
      static int error_once = 0;
      if (1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXS ERROR: invalid convolution descriptor (weight update)!\n");
      }
    }
#endif
  return code.pmm;
}

#endif /*defined(LIBXS_BUILD) || defined(LIBXS_DNN_INTERNAL_API)*/
