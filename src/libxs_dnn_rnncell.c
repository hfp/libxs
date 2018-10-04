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
    /* Need to allocate space for scratch libxs_dnn_tensor's */
    handle->z  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->deltat = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->zi = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->barrier = libxs_barrier_create(handle->desc.nThreads, 1);
    if (NULL == handle->deltat || NULL == handle->z || NULL == handle->zi ||
        NULL == handle->barrier)
    {
      free(handle->deltat); free(handle->z); free(handle->zi);
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
    free(handle->deltat); free(handle->z); free(handle->zi);
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
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* zi */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* deltat */
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
  size_t scratch_size = 0;
  const size_t sizeof_datatype = sizeof(float);

  if (scratch == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->zi->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->zi->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltat->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltat->data = (void*)(address+offset);
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->zi->data = 0;
                                           handle->deltat->data = 0;
                                           handle->zi = 0;
                                           handle->deltat = 0;
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
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
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
                                           handle->z->data = 0;
                                           handle->z = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->z->data = 0;
                                           handle->z = 0;
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
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z, (LIBXS_DNN_ELTWISE_FTYPE*)handle->z->data, K * N);
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
  if ( (type != LIBXS_DNN_RNN_REGULAR_INPUT)       && (type != LIBXS_DNN_RNN_GRADIENT_INPUT)  &&
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
        handle->h = (libxs_dnn_tensor*)tensor;
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
  if ( (type != LIBXS_DNN_RNN_REGULAR_INPUT)       && (type != LIBXS_DNN_RNN_GRADIENT_INPUT)  &&
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
      tensor = handle->h;
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
      handle->h = 0;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_fwd(libxs_dnn_rnncell* rnn, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint K = rnn->desc.K;
  libxs_blasint N = rnn->desc.N;
  libxs_blasint C = rnn->desc.C;
  libxs_blasint t = rnn->desc.t;
  libxs_blasint bk = rnn->bk;
  libxs_blasint bn = rnn->bn;
  libxs_blasint bc = rnn->bc;
  int nonlin = rnn->desc.nonlin;
  /* The following code should be in template */
  LIBXS_DNN_ELTWISE_FTYPE *wD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *uD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXS_DNN_ELTWISE_FTYPE *b = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->b->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *zt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, w, wD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, u, uD, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, z, zt, N, K);
  libxs_blasint i, ik, in, ic;
  libxs_smmfunction gemmkernela = libxs_smmdispatch( bk, bn, bc, &K, &C, &K, NULL, NULL, NULL, NULL );
  libxs_smmfunction gemmkernelb = libxs_smmdispatch( bk, bn, bk, &K, &K, &K, NULL, NULL, NULL, NULL );
  LIBXS_UNUSED(tid); LIBXS_UNUSED(start_thread); /* TODO: remove */
  /* All data is in column-major format */
  for (i = 0; i < t; ++i) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        /* z = per_col(b) */
        libxs_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &b[ik] );

        /* z += W.x */
        for (ic = 0; ic < C; ic += bc) {
          /* this is a small matmul */
          gemmkernela( &LIBXS_VLA_ACCESS(2, w, ic, ik, K), &LIBXS_VLA_ACCESS(3, x, i, in, ic, N, C), &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K) );
        }
        /* z2 += U.h */
        for (ic = 0; ic < K; ic += bk) {
          /* this is a small matmul */
          gemmkernelb( &LIBXS_VLA_ACCESS(2, u, ic, ik, K), &LIBXS_VLA_ACCESS(3, h, i, in, ic, N, K), &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K) );
        }
        if (1 == nonlin) {
          libxs_internal_matrix_relu_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXS_VLA_ACCESS(3, h, i+1, in, ik, N, K) );
        } else if (2 == nonlin) {
          libxs_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXS_VLA_ACCESS(3, h, i+1, in, ik, N, K) );
        } else {
          libxs_internal_matrix_tanh_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXS_VLA_ACCESS(3, h, i+1, in, ik, N, K) );
        }
      }
    }
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bwd_upd_bu(libxs_dnn_rnncell* rnn, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint K = rnn->desc.K;
  libxs_blasint N = rnn->desc.N;
  libxs_blasint C = rnn->desc.C;
  libxs_blasint t = rnn->desc.t;
  libxs_blasint bk = rnn->bk;
  libxs_blasint bn = rnn->bn;
  libxs_blasint bc = rnn->bc;
  int nonlin = rnn->desc.nonlin;
  int nThreads = rnn->desc.nThreads;
  LIBXS_DNN_ELTWISE_FTYPE *djdht = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdht->data;
  LIBXS_DNN_ELTWISE_FTYPE *zt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z->data;
  LIBXS_DNN_ELTWISE_FTYPE *deltat = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->deltat->data;
  LIBXS_DNN_ELTWISE_FTYPE *uD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *wD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXS_DNN_ELTWISE_FTYPE *djduD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdu->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdw->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdb = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdb->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdxt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdxt->data;
  LIBXS_DNN_ELTWISE_FTYPE* ziD = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->zi->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, u, uD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, w, wD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdu, djduD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdw, djdwD, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, djdh, djdht, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, z, zt, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, delta, deltat, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, djdx, djdxt, N, C);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, zi, ziD, K);

  libxs_blasint i, ik, in, ic, jk, jn, jc, ek, en, ec;
  /*const int ltid = tid - start_thread;*/
  /* initialization is done at the beginning */
  libxs_internal_matrix_zero(N*C*t, djdxt, start_thread, tid, nThreads);
  libxs_internal_matrix_zero(C*K,   djdwD, start_thread, tid, nThreads);
  libxs_internal_matrix_zero(K*K,   djduD, start_thread, tid, nThreads);
  /* The following code is for time step t-1 */
  for (in = 0; in < N; in += bn) {
    for (ik = 0; ik < K; ik += bk) {
      if (1 == nonlin) {
        libxs_internal_matrix_relu_inverse_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXS_VLA_ACCESS(2, zi, in, ik, K) );
      } else if (2 == nonlin) {
        libxs_internal_matrix_sigmoid_inverse_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXS_VLA_ACCESS(2, zi, in, ik, K) );
      } else {
        libxs_internal_matrix_tanh_inverse_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXS_VLA_ACCESS(2, zi, in, ik, K) );
      }
      libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(2, zi, in, ik, K),
                                                          &LIBXS_VLA_ACCESS(3, djdh,  t-1, in, ik, N, K),
                                                          &LIBXS_VLA_ACCESS(3, delta, t-1, in, ik, N, K) );
      for (jn = 0; jn < bn; jn++) {
        for (jk = 0; jk < bk; jk++) {
          en = in + jn;
          ek = ik + jk;
          if (1 == pass || 3 == pass) {
            /* djdx = W^T * delta */
            for (ic = 0; ic < C; ic += bc) {
              for (jc = 0; jc < bc; jc++) {
                ec = ic + jc;
                LIBXS_VLA_ACCESS(3, djdx, t-1, en, ec, N, C) += LIBXS_VLA_ACCESS(3, delta, t-1, en, ek, N, K) * LIBXS_VLA_ACCESS(2, w, ec, ek, K);
              }
            }
          }
          if (2 == pass || 3 == pass) {
            /* djdu = delta * h^T */
            for (ic = 0; ic < K; ic += bk) {
              for (jc = 0; jc < bk; jc++) {
                ec = ic + jc;
                LIBXS_VLA_ACCESS(2, djdu, ec, ek, K) += LIBXS_VLA_ACCESS(3, h, t-1, en, ec, N, K) * LIBXS_VLA_ACCESS(3, delta, t-1, en, ek, N, K);
              }
            }
            /* djdw = delta * x^T */
            for (ic = 0; ic < C; ic += bc) {
              for (jc = 0; jc < bc; jc++) {
                ec = ic + jc;
                LIBXS_VLA_ACCESS(2, djdw, ec, ek, K) += LIBXS_VLA_ACCESS(3, x, t-1, en, ec, N, C) * LIBXS_VLA_ACCESS(3, delta, t-1, en, ek, N, K);
              }
            }
            djdb[ek] += LIBXS_VLA_ACCESS(3, delta, t-1, en, ek, N, K);
          }
        }
      }
    }
  }
  for (i = t-2; i >= 0; --i) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        if (1 == nonlin) {
          libxs_internal_matrix_relu_inverse_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXS_VLA_ACCESS(2, zi, in, ik, K) );
        } else if (2 == nonlin) {
          libxs_internal_matrix_sigmoid_inverse_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXS_VLA_ACCESS(2, zi, in, ik, K) );
        } else {
          libxs_internal_matrix_tanh_inverse_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXS_VLA_ACCESS(2, zi, in, ik, K) );
        }
        /* di1 = U^T * delta */
        for (jn = 0; jn < bn; jn++) {
          for (jk = 0; jk < bk; jk++) {
            en = in + jn;
            ek = ik + jk;
            LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K) = (LIBXS_DNN_ELTWISE_FTYPE)0;
            for (ic = 0; ic < K; ic += bk) {
              for (jc = 0; jc < bk; jc++) {
                ec = ic + jc;
                LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K) += LIBXS_VLA_ACCESS(3, delta, i+1, en, ec, N, K) * LIBXS_VLA_ACCESS(2, u, ek, ec, K);
              }
            }
            LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K) += LIBXS_VLA_ACCESS(3, djdh, i, en, ek, N, K);
            LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K) *= LIBXS_VLA_ACCESS(2, zi, en, ek, K);
            if (1 == pass || 3 == pass) {
              /* djdx = W^T * delta */
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXS_VLA_ACCESS(3, djdx, i, en, ec, N, C) += LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K) * LIBXS_VLA_ACCESS(2, w, ec, ek, K);
                }
              }
            }
            if (2 == pass || 3 == pass) {
              /* djdu = delta * h^T */
              for (ic = 0; ic < K; ic += bk) {
                for (jc = 0; jc < bk; jc++) {
                  ec = ic + jc;
                  LIBXS_VLA_ACCESS(2, djdu, ec, ek, K) += LIBXS_VLA_ACCESS(3, h, i, en, ec, N, K) * LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K);
                }
              }
              /* djdw = delta * x^T */
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXS_VLA_ACCESS(2, djdw, ec, ek, K) += LIBXS_VLA_ACCESS(3, x, i, en, ec, N, C) * LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K);
                }
              }
              djdb[ek] += LIBXS_VLA_ACCESS(3, delta, i, en, ek, N, K);
            }
          }
        }
      }
    }
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
                                           status = libxs_dnn_rnncell_fwd(handle, start_thread, tid);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                           status = libxs_dnn_rnncell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                           status = libxs_dnn_rnncell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           status = libxs_dnn_rnncell_bwd_upd_bu(handle, start_thread, tid, 3);
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
