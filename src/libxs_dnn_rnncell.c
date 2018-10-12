/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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

#include "libxs_dnn_rnncell_forward.h"
#include "libxs_dnn_rnncell_backward_weight_update.h"
#include "libxs_dnn_elementwise.h"
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <math.h>
#include <string.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXS_API libxs_dnn_rnncell* libxs_dnn_create_rnncell(libxs_dnn_rnncell_desc rnncell_desc, libxs_dnn_err_t* status)
{
  libxs_dnn_rnncell* handle = 0;
  handle = (libxs_dnn_rnncell*)malloc(sizeof(libxs_dnn_rnncell));
  if (0 != handle) {
    *status = LIBXS_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = rnncell_desc;
    if ( (rnncell_desc.datatype_in != LIBXS_DNN_DATATYPE_F32) || (rnncell_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    if (rnncell_desc.t < 1) {
      *status = LIBXS_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bk = 64;
    handle->bn = 64;
    handle->bc = 64;
    if ( LIBXS_X86_AVX512 <= libxs_target_archid ) {
      handle->fwd_generic = 0;
      handle->bwdupd_generic = 0;
    } else {
      handle->fwd_generic = 1;
      handle->bwdupd_generic = 1;
    }
    /* Need to allocate space for scratch libxs_dnn_tensor's */
    handle->internal_z = 0;
    handle->scratch_deltat = 0;
    handle->barrier = libxs_barrier_create(handle->desc.threads, 1);
    if (NULL == handle->barrier)
    {
      *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_rnncell(const libxs_dnn_rnncell* handle)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_rnncell*)handle);
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }
  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_rnncell_create_tensor_datalayout(const libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor_datalayout* layout;
  *status = LIBXS_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      if ( (type == LIBXS_DNN_RNN_REGULAR_INPUT) || (type == LIBXS_DNN_RNN_GRADIENT_INPUT) ||
           (type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXS_DNN_ACTIVATION;
        if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCNC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXS_DNN_RNN_REGULAR_INPUT) || (type == LIBXS_DNN_RNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else if ( (type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
        } else if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXS_DNN_RNN_REGULAR_INPUT) || (type == LIBXS_DNN_RNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[2] = (unsigned int)handle->bn;
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else if ( (type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[2] = (unsigned int)handle->bn;
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
      } else if ( (type == LIBXS_DNN_RNN_REGULAR_WEIGHT)       || (type == LIBXS_DNN_RNN_GRADIENT_WEIGHT) ||
                  (type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXS_DNN_FILTER;
        if ((handle->desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_KCCK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXS_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_WEIGHT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bc;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
              } else if ( (type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bk;
                layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
        } else if ((handle->desc.filter_format & LIBXS_DNN_TENSOR_FORMAT_CK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXS_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_WEIGHT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_C;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[2] = (unsigned int)handle->bc;
                layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
              } else if ( (type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[2] = (unsigned int)handle->bk;
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
      } else if ( (type == LIBXS_DNN_RNN_REGULAR_BIAS) || (type == LIBXS_DNN_RNN_GRADIENT_BIAS) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXS_DNN_CHANNEL_SCALAR;

        if ( ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NC) > 0) || ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_NCNC) > 0) ) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(2*sizeof(libxs_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;

              if ( (type == LIBXS_DNN_RNN_REGULAR_BIAS) || (type == LIBXS_DNN_RNN_GRADIENT_BIAS) ) {
                layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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


LIBXS_API size_t libxs_dnn_rnncell_get_scratch_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        size += 0;
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxs_dnn_typesize(handle->desc.datatype_out) * (size_t)handle->desc.t; /* deltat */
        size += 64;
        size += (size_t)handle->desc.C * (size_t)handle->desc.K * libxs_dnn_typesize(handle->desc.datatype_in); /* wT */
        size += 64;
        size += (size_t)handle->desc.K * (size_t)handle->desc.K * libxs_dnn_typesize(handle->desc.datatype_in); /* uT */
        size += 64;
        size += (size_t)handle->desc.C * (size_t)handle->desc.N * libxs_dnn_typesize(handle->desc.datatype_in); /* xT */
        size += 64;
        size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxs_dnn_typesize(handle->desc.datatype_in); /* hT */
        size += 64;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* scratch)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        /* forward only has no scratch need */
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        if (scratch == 0) {
          status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
          return status;
        }

        /* deltat */
        if (address % 64 == 0) {
          handle->scratch_deltat = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratch_deltat = (void*)(address+offset);
        }
        address += ((size_t)handle->desc.K * (size_t)handle->desc.N * libxs_dnn_typesize(handle->desc.datatype_out) * (size_t)handle->desc.t) + 64;
        /* wT */
        if (address % 64 == 0) {
          handle->scratch_wT = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratch_wT = (void*)(address+offset);
        }
        address += ((size_t)handle->desc.C * (size_t)handle->desc.K * libxs_dnn_typesize(handle->desc.datatype_in)) + 64;
        /* uT */
        if (address % 64 == 0) {
          handle->scratch_uT = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratch_uT = (void*)(address+offset);
        }
        address += ((size_t)handle->desc.K * (size_t)handle->desc.K * libxs_dnn_typesize(handle->desc.datatype_in)) + 64;
        /* xT */
        if (address % 64 == 0) {
          handle->scratch_xT = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratch_xT = (void*)(address+offset);
        }
        address += ((size_t)handle->desc.C * (size_t)handle->desc.N * libxs_dnn_typesize(handle->desc.datatype_in)) + 64;
        /* hT */
        if (address % 64 == 0) {
          handle->scratch_hT = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratch_hT = (void*)(address+offset);
        }
        address += ((size_t)handle->desc.K * (size_t)handle->desc.N * libxs_dnn_typesize(handle->desc.datatype_in)) + 64;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        /* forward only has no scratch need */
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        handle->scratch_deltat = 0;
        handle->scratch_wT = 0;
        handle->scratch_uT = 0;
        handle->scratch_xT = 0;
        handle->scratch_hT = 0;
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

LIBXS_API size_t libxs_dnn_rnncell_get_internalstate_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* zt */
        size += 64;
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* zt */
        size += 64;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* internalstate)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)internalstate;
  size_t offset = 0;

  if (internalstate == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        if (address % 64 == 0) {
          handle->internal_z = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->internal_z = (void*)(address+offset);
        }
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        if (address % 64 == 0) {
          handle->internal_z = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->internal_z = (void*)(address+offset);
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        handle->internal_z = 0;
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        handle->internal_z = 0;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_assign_internalstate(libxs_dnn_rnncell* handle, const void* zgoldtb)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && zgoldtb != 0) {
    const libxs_blasint K = handle->desc.K, N = handle->desc.N, t = handle->desc.t;
    LIBXS_VLA_DECL(2, /*const*/ LIBXS_DNN_ELTWISE_FTYPE, zgold, (/*const*/ LIBXS_DNN_ELTWISE_FTYPE*)zgoldtb, K * N);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z, (LIBXS_DNN_ELTWISE_FTYPE*)handle->internal_z, K * N);
    libxs_blasint it;
    for (it = 0; it < t; ++it) {
      libxs_internal_matrix_copy(K*N, &LIBXS_VLA_ACCESS(2, zgold, it, 0, K * N), &LIBXS_VLA_ACCESS(2, z, it, 0, K * N), 0, 0, 1);
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_RNN_REGULAR_INPUT)        && (type != LIBXS_DNN_RNN_GRADIENT_INPUT)  &&
       (type != LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
       (type != LIBXS_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXS_DNN_RNN_GRADIENT_WEIGHT) &&
       (type != LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) &&
       (type != LIBXS_DNN_RNN_REGULAR_BIAS)         && (type != LIBXS_DNN_RNN_GRADIENT_BIAS) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_rnncell_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_RNN_REGULAR_INPUT ) {
        handle->xt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_GRADIENT_INPUT ) {
        handle->djdxt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE ) {
        handle->ht = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
        handle->djdht = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_REGULAR_WEIGHT ) {
        handle->w = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_GRADIENT_WEIGHT ) {
        handle->djdw = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
        handle->u = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
        handle->djdu = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_REGULAR_BIAS ) {
        handle->b = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_RNN_GRADIENT_BIAS ) {
        handle->djdb = (libxs_dnn_tensor*)tensor;
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


LIBXS_API libxs_dnn_tensor* libxs_dnn_rnncell_get_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor* tensor = 0;
  LIBXS_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXS_DNN_RNN_REGULAR_INPUT)        && (type != LIBXS_DNN_RNN_GRADIENT_INPUT)  &&
       (type != LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
       (type != LIBXS_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXS_DNN_RNN_GRADIENT_WEIGHT) &&
       (type != LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) &&
       (type != LIBXS_DNN_RNN_REGULAR_BIAS)         && (type != LIBXS_DNN_RNN_GRADIENT_BIAS) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_RNN_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_INPUT ) {
      tensor = handle->djdxt;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE ) {
      tensor = handle->ht;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->djdht;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_WEIGHT ) {
      tensor = handle->w;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_WEIGHT ) {
      tensor = handle->djdw;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
      tensor = handle->u;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
      tensor = handle->djdu;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_BIAS ) {
      tensor = handle->b;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_BIAS ) {
      tensor = handle->djdb;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_RNN_REGULAR_INPUT)       && (type != LIBXS_DNN_RNN_GRADIENT_INPUT)  &&
       (type != LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
       (type != LIBXS_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXS_DNN_RNN_GRADIENT_WEIGHT) &&
       (type != LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) &&
       (type != LIBXS_DNN_RNN_REGULAR_BIAS)         && (type != LIBXS_DNN_RNN_GRADIENT_BIAS) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_RNN_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_INPUT ) {
      handle->djdxt = 0;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE ) {
      handle->ht = 0;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
      handle->djdht = 0;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_WEIGHT ) {
      handle->w = 0;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_WEIGHT ) {
      handle->djdw = 0;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
      handle->u = 0;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
      handle->djdu = 0;
    } else if ( type == LIBXS_DNN_RNN_REGULAR_BIAS ) {
      handle->b = 0;
    } else if ( type == LIBXS_DNN_RNN_GRADIENT_BIAS ) {
      handle->djdb = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_execute_st(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NC) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_CK) ) {
          status = libxs_dnn_rnncell_st_fwd_nc_ck( handle, start_thread, tid );
        } else if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NCNC) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_KCCK)  ) {
          status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
          /*status = libxs_dnn_rnncell_st_fwd_ncnc_kcck( handle, start_thread, tid );*/
        } else {
          status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_BWDUPD: {
        if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NC) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_CK) ) {
          status = libxs_dnn_rnncell_st_bwdupd_nc_ck( handle, kind, start_thread, tid );
        } else if ( (handle->desc.buffer_format == LIBXS_DNN_TENSOR_FORMAT_NCNC) && (handle->desc.filter_format == LIBXS_DNN_TENSOR_FORMAT_KCCK)  ) {
          status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
          /*status = libxs_dnn_rnncell_st_bwdupd_ncnc_kcck( handle, kind, start_thread, tid );*/
        } else {
          status = LIBXS_DNN_ERR_INVALID_FORMAT_GENERAL;
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
