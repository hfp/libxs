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

#include <libxs.h>
#include "libxs_main.h"

LIBXS_API libxs_dnn_rnncell* libxs_dnn_create_rnncell(libxs_dnn_rnncell_desc rnncell_desc, libxs_dnn_err_t* status) {
  libxs_dnn_rnncell* handle = 0;
  *status = LIBXS_DNN_SUCCESS;

  handle = (libxs_dnn_rnncell*)malloc(sizeof(libxs_dnn_rnncell));
  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = rnncell_desc;
    handle->datatype_in = rnncell_desc.datatype_in;
    handle->datatype_out = rnncell_desc.datatype_out;
    if ( (rnncell_desc.datatype_in != LIBXS_DNN_DATATYPE_F32) || (rnncell_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = rnncell_desc.buffer_format;
    handle->m = rnncell_desc.m;
    handle->n = rnncell_desc.n;
    handle->k = rnncell_desc.k;
    handle->t = rnncell_desc.t;
    if (rnncell_desc.t < 2) {
      *status = LIBXS_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bm = rnncell_desc.bm;
    handle->bn = rnncell_desc.bn;
    handle->bk = rnncell_desc.bk;
    handle->b_m1 = rnncell_desc.b_m1;
    handle->b_n1 = rnncell_desc.b_n1;
    handle->b_k1 = rnncell_desc.b_k1;
    handle->b_m2 = rnncell_desc.b_m2;
    handle->b_n2 = rnncell_desc.b_n2;
    handle->b_k2 = rnncell_desc.b_k2;
  } else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_rnncell(const libxs_dnn_rnncell* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  if (0 != handle) {
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_rnncell*)handle);
  }
  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_rnncell_create_tensor_datalayout(const libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* layout = 0;
  *status = LIBXS_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXS_DNN_REGULAR_INPUT)  || (type == LIBXS_DNN_GRADIENT_INPUT)  || (type == LIBXS_DNN_INPUT)  ||
           (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT)    ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXS_DNN_ACTIVATION;

        if ((handle->buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( ((handle->datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            if (1 /*handle->custom_format_type == LIBXS_DNN_TENSOR_FORMAT_LIBXS_1*/) {
              layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 4;
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_H;
                if ( (type == LIBXS_DNN_REGULAR_INPUT) || (type == LIBXS_DNN_GRADIENT_INPUT) || (type == LIBXS_DNN_INPUT) ) {
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXS_DNN_REGULAR_OUTPUT) || (type == LIBXS_DNN_GRADIENT_OUTPUT) || (type == LIBXS_DNN_OUTPUT) ) {
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
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
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
      }
    } else {
      *status = LIBXS_DNN_ERR_CREATE_LAYOUT;
    } 
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }
  return layout;
}


LIBXS_API size_t libxs_dnn_rnncell_get_scratch_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status) {
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;
  
  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                         } break;
      default: {
                 *status = LIBXS_DNN_ERR_INVALID_KIND;
               }
    }
  } else {
    *status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return size;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* scratch) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  LIBXS_UNUSED( scratch );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API size_t libxs_dnn_rnncell_get_internalstate_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status) {
  size_t size = 0;

  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  *status = LIBXS_DNN_SUCCESS;

  return size;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* internalstate) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  LIBXS_UNUSED( internalstate );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( tensor );
  LIBXS_UNUSED( type );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_tensor* libxs_dnn_rnncell_get_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor* tensor = 0;

  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( type );
  *status = LIBXS_DNN_SUCCESS;

  return tensor;
}

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( type );

  return LIBXS_DNN_SUCCESS;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_execute_st(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  LIBXS_UNUSED( handle );
  LIBXS_UNUSED( kind );
  LIBXS_UNUSED( start_thread );
  LIBXS_UNUSED( tid );

  return LIBXS_DNN_SUCCESS;
}

