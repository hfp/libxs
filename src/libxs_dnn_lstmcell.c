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
#include "libxs_dnn_elementwise.h"
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API libxs_dnn_lstmcell* libxs_dnn_create_lstmcell(libxs_dnn_lstmcell_desc lstmcell_desc, libxs_dnn_err_t* status)
{
  libxs_dnn_lstmcell* handle = 0;
  handle = (libxs_dnn_lstmcell*)malloc(sizeof(libxs_dnn_lstmcell));
  if (0 != handle) {
    *status = LIBXS_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = lstmcell_desc;
    if ( (lstmcell_desc.datatype_in != LIBXS_DNN_DATATYPE_F32) || (lstmcell_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->bk = 64;
    handle->bn = 64;
    handle->bc = 64;
    if (lstmcell_desc.t < 1) {
      *status = LIBXS_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    /* Need to allocate space for scratch and internalstate libxs_dnn_tensor's */
    handle->dit  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->dft  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->dot  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->dct  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->deltat = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->doutt  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->t1 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->t2 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->barrier = libxs_barrier_create(handle->desc.nThreads, 1);
    if (NULL == handle->dit  || NULL == handle->dft    || NULL == handle->dot  ||
        NULL == handle->dct  || NULL == handle->deltat || NULL == handle->doutt ||
        NULL == handle->t1  || NULL == handle->t2      || NULL == handle->barrier)
    {
      free(handle->dit);    free(handle->dft);   free(handle->dot); free(handle->dct);
      free(handle->deltat); free(handle->doutt); free(handle->t1);  free(handle->t2);
      *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_lstmcell(const libxs_dnn_lstmcell* handle)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  if (0 != handle) {
    free(handle->dit);    free(handle->dft);   free(handle->dot); free(handle->dct);
    free(handle->deltat); free(handle->doutt); free(handle->t1);  free(handle->t2);
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_lstmcell*)handle);
  }
  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_lstmcell_create_tensor_datalayout(const libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor_datalayout* layout = 0;
  *status = LIBXS_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXS_DNN_LSTM_REGULAR_INPUT)             || (type == LIBXS_DNN_LSTM_GRADIENT_INPUT)             ||
           (type == LIBXS_DNN_LSTM_REGULAR_CS_PREV)           || (type == LIBXS_DNN_LSTM_GRADIENT_CS_PREV)           ||
           (type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) ||
           (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT)            || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT)            ||
           (type == LIBXS_DNN_LSTM_REGULAR_BIAS)              || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS)              ||
           (type == LIBXS_DNN_LSTM_REGULAR_CS)                || (type == LIBXS_DNN_LSTM_GRADIENT_CS)                ||
           (type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)      || (type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE)      ||
           (type == LIBXS_DNN_LSTM_INTERNAL_I)                || (type == LIBXS_DNN_LSTM_INTERNAL_F)                 ||
           (type == LIBXS_DNN_LSTM_INTERNAL_O)                || (type == LIBXS_DNN_LSTM_INTERNAL_C) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXS_DNN_ACTIVATION;

        if ((handle->desc.buffer_format & LIBXS_DNN_TENSOR_FORMAT_LIBXS) > 0) {
          if ( ((handle->desc.datatype_in == LIBXS_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXS_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXS_DNN_DATATYPE_F32;
            if (1 /*handle->custom_format_type == LIBXS_DNN_TENSOR_FORMAT_LIBXS_1*/) {
              layout->dim_type = (libxs_dnn_tensor_dimtype*) malloc(4*sizeof(libxs_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 4;
                if ( (type == LIBXS_DNN_LSTM_REGULAR_INPUT) || (type == LIBXS_DNN_LSTM_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = (unsigned int)handle->bc;
                  layout->dim_size[1] = (unsigned int)handle->bn;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_CS_PREV)           || (type == LIBXS_DNN_LSTM_GRADIENT_CS_PREV)           ||
                            (type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)      || (type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE)      ||
                            (type == LIBXS_DNN_LSTM_REGULAR_CS)                || (type == LIBXS_DNN_LSTM_GRADIENT_CS)                ||
                            (type == LIBXS_DNN_LSTM_INTERNAL_I)                || (type == LIBXS_DNN_LSTM_INTERNAL_F)                 ||
                            (type == LIBXS_DNN_LSTM_INTERNAL_O)                || (type == LIBXS_DNN_LSTM_INTERNAL_C) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bn;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT) || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_BIAS) || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
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


LIBXS_API size_t libxs_dnn_lstmcell_get_scratch_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size += 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dit */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dft */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dot */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dct */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* deltat */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* doutt */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* t1 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* t2 */
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* scratch)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;
  size_t scratch_size = 0;
  const size_t sizeof_datatype = sizeof(float);

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
        /* forward only has no scratch need */
      } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
        if (scratch == 0) {
          status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
          return status;
        }

        if (address % 64 == 0) {
          handle->dit->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dit->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->dft->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dft->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->dot->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dot->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->dct->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dct->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->deltat->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->deltat->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->doutt->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->doutt->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->t1->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->t1->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->t2->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->t2->data = (void*)(address+offset);
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->dit->data = 0;
                                           handle->dft->data = 0;
                                           handle->dot->data = 0;
                                           handle->dct->data = 0;
                                           handle->deltat->data = 0;
                                           handle->doutt->data = 0;
                                           handle->t1->data = 0;
                                           handle->t2->data = 0;
                                           handle->dit = 0;
                                           handle->dft = 0;
                                           handle->dot = 0;
                                           handle->dct = 0;
                                           handle->deltat = 0;
                                           handle->doutt = 0;
                                           handle->t1 = 0;
                                           handle->t2 = 0;
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


LIBXS_API size_t libxs_dnn_lstmcell_get_internalstate_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;
  LIBXS_UNUSED(sizeof_datatype); LIBXS_UNUSED(size);

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           /* with i, f, o, c, d exposed as i/o, there is currently no need for internal state */
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           /* with i, f, o, c, d exposed as i/o, there is currently no need for internal state */
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* internalstate)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)internalstate;
  size_t offset = 0;
  size_t scratch_size = 0;
  const size_t sizeof_datatype = sizeof(float);
  LIBXS_UNUSED(sizeof_datatype); LIBXS_UNUSED(address); LIBXS_UNUSED(offset); LIBXS_UNUSED(scratch_size);

  /*
  if (internalstate == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }
  */
  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_assign_internalstate(libxs_dnn_lstmcell* handle, const void* igoldtb, const void* fgoldtb, const void* ogoldtb, const void* cgoldtb, const void* dgoldtb)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && igoldtb != 0 && fgoldtb != 0 && ogoldtb != 0 && cgoldtb != 0 && dgoldtb != 0) {
    const libxs_blasint K = handle->desc.K, N = handle->desc.N, t = handle->desc.t;
    LIBXS_VLA_DECL(2, /*const*/ LIBXS_DNN_ELTWISE_FTYPE, igold, (/*const*/ LIBXS_DNN_ELTWISE_FTYPE*)igoldtb, K * N);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i, (LIBXS_DNN_ELTWISE_FTYPE*)handle->it->data, K * N);
    LIBXS_VLA_DECL(2, /*const*/ LIBXS_DNN_ELTWISE_FTYPE, fgold, (/*const*/ LIBXS_DNN_ELTWISE_FTYPE*)fgoldtb, K * N);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f, (LIBXS_DNN_ELTWISE_FTYPE*)handle->ft->data, K * N);
    LIBXS_VLA_DECL(2, /*const*/ LIBXS_DNN_ELTWISE_FTYPE, ogold, (/*const*/ LIBXS_DNN_ELTWISE_FTYPE*)ogoldtb, K * N);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o, (LIBXS_DNN_ELTWISE_FTYPE*)handle->ot->data, K * N);
    LIBXS_VLA_DECL(2, /*const*/ LIBXS_DNN_ELTWISE_FTYPE, cgold, (/*const*/ LIBXS_DNN_ELTWISE_FTYPE*)cgoldtb, K * N);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c, (LIBXS_DNN_ELTWISE_FTYPE*)handle->ct->data, K * N);
    LIBXS_VLA_DECL(2, /*const*/ LIBXS_DNN_ELTWISE_FTYPE, dgold, (/*const*/ LIBXS_DNN_ELTWISE_FTYPE*)dgoldtb, K * N);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, cs, (LIBXS_DNN_ELTWISE_FTYPE*)handle->cst->data, K * N);
    libxs_blasint it;
    for (it = 0; it < t; ++it) {
      libxs_internal_matrix_copy(K*N, &LIBXS_VLA_ACCESS(2, igold, it, 0, K * N), &LIBXS_VLA_ACCESS(2, i, it, 0, K * N), 0, 0, 1);
      libxs_internal_matrix_copy(K*N, &LIBXS_VLA_ACCESS(2, fgold, it, 0, K * N), &LIBXS_VLA_ACCESS(2, f, it, 0, K * N), 0, 0, 1);
      libxs_internal_matrix_copy(K*N, &LIBXS_VLA_ACCESS(2, ogold, it, 0, K * N), &LIBXS_VLA_ACCESS(2, o, it, 0, K * N), 0, 0, 1);
      libxs_internal_matrix_copy(K*N, &LIBXS_VLA_ACCESS(2, cgold, it, 0, K * N), &LIBXS_VLA_ACCESS(2, c, it, 0, K * N), 0, 0, 1);
      libxs_internal_matrix_copy(K*N, &LIBXS_VLA_ACCESS(2, dgold, it, 0, K * N), &LIBXS_VLA_ACCESS(2, cs, it, 0, K * N), 0, 0, 1);
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_LSTM_REGULAR_INPUT)             && (type != LIBXS_DNN_LSTM_GRADIENT_INPUT)             &&
       (type != LIBXS_DNN_LSTM_REGULAR_CS_PREV)           && (type != LIBXS_DNN_LSTM_GRADIENT_CS_PREV)           &&
       (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT)            && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT)            &&
       (type != LIBXS_DNN_LSTM_REGULAR_BIAS)              && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS)              &&
       (type != LIBXS_DNN_LSTM_REGULAR_CS)                && (type != LIBXS_DNN_LSTM_GRADIENT_CS)                &&
       (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)      && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXS_DNN_LSTM_INTERNAL_I)                && (type != LIBXS_DNN_LSTM_INTERNAL_F)                 &&
       (type != LIBXS_DNN_LSTM_INTERNAL_O) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_lstmcell_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_LSTM_REGULAR_INPUT ) {
        handle->xt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_INPUT ) {
        handle->dxt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_CS_PREV ) {
        handle->csp = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_CS_PREV ) {
        handle->dcsp = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) {
        handle->hp = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) {
        handle->dhp = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT ) {
        handle->w = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT ) {
        handle->dw = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS ) {
        handle->b = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS ) {
        handle->db = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_CS ) {
        handle->cst = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_CS ) {
        handle->dcst = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
        handle->ht = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
        handle->dht = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_INTERNAL_I ) {
        handle->it = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_INTERNAL_F ) {
        handle->ft = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_INTERNAL_O ) {
        handle->ot = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_INTERNAL_C ) {
        handle->ct = (libxs_dnn_tensor*)tensor;
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


LIBXS_API libxs_dnn_tensor* libxs_dnn_lstmcell_get_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor* tensor = 0;
  LIBXS_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXS_DNN_LSTM_REGULAR_INPUT)             && (type != LIBXS_DNN_LSTM_GRADIENT_INPUT)             &&
       (type != LIBXS_DNN_LSTM_REGULAR_CS_PREV)           && (type != LIBXS_DNN_LSTM_GRADIENT_CS_PREV)           &&
       (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT)            && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT)            &&
       (type != LIBXS_DNN_LSTM_REGULAR_BIAS)              && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS)              &&
       (type != LIBXS_DNN_LSTM_REGULAR_CS)                && (type != LIBXS_DNN_LSTM_GRADIENT_CS)                &&
       (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)      && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXS_DNN_LSTM_INTERNAL_I)                && (type != LIBXS_DNN_LSTM_INTERNAL_F)                 &&
       (type != LIBXS_DNN_LSTM_INTERNAL_O) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_LSTM_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_INPUT ) {
      tensor = handle->dxt;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_CS_PREV ) {
      tensor = handle->csp;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_CS_PREV ) {
      tensor = handle->dcsp;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) {
      tensor = handle->hp;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) {
      tensor = handle->dhp;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT ) {
      tensor = handle->w;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT ) {
      tensor = handle->dw;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS ) {
      tensor = handle->b;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS ) {
      tensor = handle->db;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_CS ) {
      tensor = handle->cst;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_CS ) {
      tensor = handle->dcst;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      tensor = handle->ht;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->dht;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_I ) {
      tensor = handle->it;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_F ) {
      tensor = handle->ft;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_O ) {
      tensor = handle->ot;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_C ) {
      tensor = handle->ct;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_LSTM_REGULAR_INPUT)             && (type != LIBXS_DNN_LSTM_GRADIENT_INPUT)             &&
       (type != LIBXS_DNN_LSTM_REGULAR_CS_PREV)           && (type != LIBXS_DNN_LSTM_GRADIENT_CS_PREV)           &&
       (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT)            && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT)            &&
       (type != LIBXS_DNN_LSTM_REGULAR_BIAS)              && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS)              &&
       (type != LIBXS_DNN_LSTM_REGULAR_CS)                && (type != LIBXS_DNN_LSTM_GRADIENT_CS)                &&
       (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)      && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXS_DNN_LSTM_INTERNAL_I)                && (type != LIBXS_DNN_LSTM_INTERNAL_F)                 &&
       (type != LIBXS_DNN_LSTM_INTERNAL_O) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_LSTM_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_INPUT ) {
      handle->dxt = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_CS_PREV ) {
      handle->csp = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_CS_PREV ) {
      handle->dcsp = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) {
      handle->hp = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) {
      handle->dhp = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT ) {
      handle->w = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT ) {
      handle->dw = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS ) {
      handle->b = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS ) {
      handle->db = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_CS ) {
      handle->cst = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_CS ) {
      handle->dcst = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      handle->ht = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      handle->dht = 0;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_I ) {
      handle->it = 0;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_F ) {
      handle->ft = 0;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_O ) {
      handle->ot = 0;
    } else if ( type == LIBXS_DNN_LSTM_INTERNAL_C ) {
      handle->ct = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_fwd(libxs_dnn_lstmcell* lstm, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint K = lstm->desc.K;
  libxs_blasint N = lstm->desc.N;
  libxs_blasint C = lstm->desc.C;
  libxs_blasint t = lstm->desc.t;
  libxs_blasint bk = lstm->bk;
  libxs_blasint bn = lstm->bn;
  libxs_blasint bc = lstm->bc;
  LIBXS_DNN_ELTWISE_FTYPE *xt  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *csp = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->csp->data;
  LIBXS_DNN_ELTWISE_FTYPE *hpD = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->hp->data;
  LIBXS_DNN_ELTWISE_FTYPE *w   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->w->data;
  LIBXS_DNN_ELTWISE_FTYPE *wiD = &(w[0]);
  LIBXS_DNN_ELTWISE_FTYPE *wcD = &(w[K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *wfD = &(w[2*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *woD = &(w[3*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *riD = &(w[4*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *rcD = &(w[4*K*N + K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *rfD = &(w[4*K*N + 2*K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *roD = &(w[4*K*N + 3*K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *b   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->b->data;
  LIBXS_DNN_ELTWISE_FTYPE *bi  = &(b[0]);
  LIBXS_DNN_ELTWISE_FTYPE *bd  = &(b[K]);
  LIBXS_DNN_ELTWISE_FTYPE *bf  = &(b[2*K]);
  LIBXS_DNN_ELTWISE_FTYPE *bo  = &(b[3*K]);
  LIBXS_DNN_ELTWISE_FTYPE *cst = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->cst->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ht->data;
  LIBXS_DNN_ELTWISE_FTYPE *it  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->it->data;
  LIBXS_DNN_ELTWISE_FTYPE *ft  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ft->data;
  LIBXS_DNN_ELTWISE_FTYPE *ot  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ot->data;
  LIBXS_DNN_ELTWISE_FTYPE *ct  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ct->data;
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, cp, csp, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, hp, hpD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wi, wiD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wf, wfD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wo, woD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wc, wcD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, ri, riD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, rf, rfD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, ro, roD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, rc, rcD, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, cs, cst, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, i, it, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, f, ft, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, o, ot, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, c, ct, N, K);
  libxs_blasint j, ik, in, ic;
  libxs_smmfunction gemmkernela = libxs_smmdispatch( bk, bn, bc, &K, &C, &K, NULL, NULL, NULL, NULL );
  libxs_smmfunction gemmkernelb = libxs_smmdispatch( bk, bn, bk, &K, &K, &K, NULL, NULL, NULL, NULL );
  LIBXS_UNUSED(tid); LIBXS_UNUSED(start_thread); /* TODO: remove */
  /* All data is in column-major format */
  for (j = 0; j < t; ++j) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        /* initialize with bias */
        libxs_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K), &bi[ik] );
        /* i += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXS_VLA_ACCESS(2, wi, ic, ik, K), &LIBXS_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K) );
        }
        /* i += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXS_VLA_ACCESS(2, ri, ic, ik, K), &LIBXS_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K) );
        }
        libxs_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K) );

        /* initialize with bias */
        libxs_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K), &bf[ik] );
        /* f += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXS_VLA_ACCESS(2, wf, ic, ik, K), &LIBXS_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K) );
        }
        /* f += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXS_VLA_ACCESS(2, rf, ic, ik, K), &LIBXS_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K) );
        }
        libxs_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K) );

        /* initialize with bias */
        libxs_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &bo[ik] );
        /* o += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXS_VLA_ACCESS(2, wo, ic, ik, K), &LIBXS_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K) );
        }
        /* o += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXS_VLA_ACCESS(2, ro, ic, ik, K), &LIBXS_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K) );
        }
        libxs_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K) );

        /* initialize with bias */
        libxs_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K), &bd[ik] );
        /* c += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXS_VLA_ACCESS(2, wc, ic, ik, K), &LIBXS_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K) );
        }
        /* c += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXS_VLA_ACCESS(2, rc, ic, ik, K), &LIBXS_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K) );
        }
        libxs_internal_matrix_tanh_ld(    bk, bn, K, &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K) );

        /* cs = f.cs */
        if (0 == j) {
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, cp, in, ik, K), &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        } else {
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        }
        /* cs += i.c */
        libxs_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        /* h = o.tanh(d) */
        if (0 == j) {
          libxs_internal_matrix_elt_mult_tanh_ld(  bk, bn, K, &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, hp, in, ik, K) );
        } else {
          libxs_internal_matrix_elt_mult_tanh_ld(  bk, bn, K, &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, h, j, in, ik, N, K) );
        }
      }
    }
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bwd_upd_bu(libxs_dnn_lstmcell* lstm, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint K = lstm->desc.K;
  libxs_blasint N = lstm->desc.N;
  libxs_blasint C = lstm->desc.C;
  libxs_blasint t = lstm->desc.t;
  libxs_blasint bk = lstm->bk;
  libxs_blasint bn = lstm->bn;
  libxs_blasint bc = lstm->bc;
  int nThreads = lstm->desc.nThreads;
  LIBXS_DNN_ELTWISE_FTYPE *xt   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  /*LIBXS_DNN_ELTWISE_FTYPE *csp  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->csp->data;*/
  /*LIBXS_DNN_ELTWISE_FTYPE *hpD  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->hp->data;*/
  LIBXS_DNN_ELTWISE_FTYPE *w    = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->w->data;
  LIBXS_DNN_ELTWISE_FTYPE *wiD  = &(w[0]);
  LIBXS_DNN_ELTWISE_FTYPE *wcD  = &(w[K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *wfD  = &(w[2*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *woD  = &(w[3*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *riD  = &(w[4*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *rcD  = &(w[4*K*N + K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *rfD  = &(w[4*K*N + 2*K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *roD  = &(w[4*K*N + 3*K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *cst  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->cst->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ht->data;
  LIBXS_DNN_ELTWISE_FTYPE *it   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->it->data;
  LIBXS_DNN_ELTWISE_FTYPE *ft   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ft->data;
  LIBXS_DNN_ELTWISE_FTYPE *ot   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ot->data;
  LIBXS_DNN_ELTWISE_FTYPE *ct   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ct->data;
  LIBXS_DNN_ELTWISE_FTYPE *dxt  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dxt->data;
  /*LIBXS_DNN_ELTWISE_FTYPE *dcsp = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dcsp->data;*/
  /*LIBXS_DNN_ELTWISE_FTYPE *dhpD = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dhp->data;*/
  LIBXS_DNN_ELTWISE_FTYPE *dw   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dw->data;
  LIBXS_DNN_ELTWISE_FTYPE *dwiD = &(dw[0]);
  LIBXS_DNN_ELTWISE_FTYPE *dwcD = &(dw[K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *dwfD = &(dw[2*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *dwoD = &(dw[3*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *driD = &(dw[4*K*N]);
  LIBXS_DNN_ELTWISE_FTYPE *drcD = &(dw[4*K*N + K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *drfD = &(dw[4*K*N + 2*K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *droD = &(dw[4*K*N + 3*K*K]);
  LIBXS_DNN_ELTWISE_FTYPE *db   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->db->data;
  LIBXS_DNN_ELTWISE_FTYPE *dbi  = &(db[0]);
  LIBXS_DNN_ELTWISE_FTYPE *dbc  = &(db[K]);
  LIBXS_DNN_ELTWISE_FTYPE *dbf  = &(db[2*K]);
  LIBXS_DNN_ELTWISE_FTYPE *dbo  = &(db[3*K]);
  LIBXS_DNN_ELTWISE_FTYPE *dcst = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dcst->data;
  LIBXS_DNN_ELTWISE_FTYPE *dht  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dht->data;
  LIBXS_DNN_ELTWISE_FTYPE *dit  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dit->data;
  LIBXS_DNN_ELTWISE_FTYPE *dft  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dft->data;
  LIBXS_DNN_ELTWISE_FTYPE *dct  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dct->data;
  LIBXS_DNN_ELTWISE_FTYPE *dot  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dot->data;
  LIBXS_DNN_ELTWISE_FTYPE *deltat = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->deltat->data;
  LIBXS_DNN_ELTWISE_FTYPE *doutt  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->doutt->data;
  LIBXS_DNN_ELTWISE_FTYPE *t1D = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->t1->data;
  LIBXS_DNN_ELTWISE_FTYPE *t2D = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->t2->data;
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, x, xt, N, C);
  /*LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, cp, csp, K);*/
  /*LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, hp, hpD, K);*/
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wi, wiD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wf, wfD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wo, woD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, wc, wcD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, ri, riD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, rf, rfD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, ro, roD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, rc, rcD, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, cs, cst, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, i, it, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, f, ft, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, o, ot, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, c, ct, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, dx, dxt, N, C);
  /*LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dcp, dcsp, K);*/
  /*LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dhp, dhpD, K);*/
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dwi, dwiD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dwf, dwfD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dwo, dwoD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dwc, dwcD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dri, driD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, drf, drfD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dro, droD, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, drc, drcD, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, dcs, dcst, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, dh, dht, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, di, dit, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, df, dft, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, dp, dot, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, dc, dct, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, delta, deltat, N, K);
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, dout, doutt, N, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, t1, t1D, K);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, t2, t2D, K);
  libxs_blasint j, ik, in, ic, jk, jn, jc, ek, en, ec;
  /* const int ltid = tid - start_thread; */
  /* initialization is done at the beginning */
  if (1 == pass || 3 == pass) {
    libxs_internal_matrix_zero(N*C*t, dxt,  start_thread, tid, nThreads);
  }
  if (2 == pass || 3 == pass) {
    libxs_internal_matrix_zero(C*K,   dwiD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(C*K,   dwfD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(C*K,   dwoD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(C*K,   dwcD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K*K,   driD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K*K,   drfD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K*K,   droD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K*K,   drcD, start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K,     dbi,  start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K,     dbf,  start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K,     dbo,  start_thread, tid, nThreads);
    libxs_internal_matrix_zero(K,     dbc,  start_thread, tid, nThreads);
  }
  libxs_internal_matrix_zero(N*K*t, doutt,  start_thread, tid, nThreads);
  for (j = t-1; j >= 0; --j) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        /* compute delta */
        if (j == t-1) {
          libxs_internal_matrix_copy( bk*bn, &LIBXS_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXS_VLA_ACCESS(3, delta, t-1, in, ik, N, K), start_thread, tid, nThreads );
        } else {
          libxs_internal_matrix_add_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, dout, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, dh, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, delta, j, in, ik, N, K) );
        }
        /* compute dcs */
        libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, delta, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K) );
        libxs_internal_matrix_tanh_inverse_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K) );
        if (j == t-1) {
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(3, dcs, j, in, ik, N, K) );
        } else {
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(3, dcs, j, in, ik, N, K) );
          libxs_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXS_VLA_ACCESS(3, delta, j+1, in, ik, N, K), &LIBXS_VLA_ACCESS(3, f, j+1, in, ik, N, K), &LIBXS_VLA_ACCESS(3, dcs, j, in, ik, N, K) );
        }
        /* compute dc */
        libxs_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXS_VLA_ACCESS(3, dcs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K) );
        libxs_internal_matrix_complement_square_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K) );
        libxs_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(3, dc, j, in, ik, N, K) );
        /* compute di */
        libxs_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXS_VLA_ACCESS(3, dcs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K) );
        libxs_internal_matrix_complement_ld(        bk, bn, K, &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K) );
        libxs_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXS_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(3, di, j, in, ik, N, K) );
        libxs_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(3, di, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, di, j, in, ik, N, K) );
        /* compute df */
        if (j >= 1) {
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, dcs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K) );
          libxs_internal_matrix_complement_ld(   bk, bn, K, &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K) );
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(3, df, j, in, ik, N, K) );
          libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(3, df, j, in, ik, N, K), &LIBXS_VLA_ACCESS(3, df, j, in, ik, N, K) );
        } else {
          /* df is zero for j == 0 */
          libxs_internal_matrix_zero( bk*bn, &LIBXS_VLA_ACCESS(3, df, j, in, ik, N, K), start_thread, tid, nThreads );
        }
        /* compute do */
        libxs_internal_matrix_tanh_ld(         bk, bn, K, &LIBXS_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K) );
        libxs_internal_matrix_complement_ld(   bk, bn, K, &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K) );
        libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, delta, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(2, t1, in, ik, K) );
        libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K) );
        libxs_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXS_VLA_ACCESS(2, t1, in, ik, K), &LIBXS_VLA_ACCESS(2, t2, in, ik, K), &LIBXS_VLA_ACCESS(3, dp, j, in, ik, N, K) );
        for (jn = 0; jn < bn; jn++) {
          for (jk = 0; jk < bk; jk++) {
            en = in + jn;
            ek = ik + jk;
            /* compute dout */
            if (j > 0) {
              for (ic = 0; ic < K; ic += bk) {
                for (jc = 0; jc < bk; jc++) {
                  ec = ic + jc;
                  LIBXS_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXS_VLA_ACCESS(3, di, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, ri, ec, ek, K);
                  LIBXS_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXS_VLA_ACCESS(3, df, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, rf, ec, ek, K);
                  LIBXS_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXS_VLA_ACCESS(3, dp, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, ro, ec, ek, K);
                  LIBXS_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXS_VLA_ACCESS(3, dc, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, rc, ec, ek, K);
                }
              }
            }
            /* compute dx */
            if (1 == pass || 3 == pass) {
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXS_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXS_VLA_ACCESS(3, di, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, wi, ec, ek, K);
                  LIBXS_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXS_VLA_ACCESS(3, df, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, wf, ec, ek, K);
                  LIBXS_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXS_VLA_ACCESS(3, dp, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, wo, ec, ek, K);
                  LIBXS_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXS_VLA_ACCESS(3, dc, j, en, ek, N, K) * LIBXS_VLA_ACCESS(2, wc, ec, ek, K);
                }
              }
            }
            if (2 == pass || 3 == pass) {
              /* dr = delta * h^T */
              if (j > 0) {
                for (ic = 0; ic < K; ic += bk) {
                  for (jc = 0; jc < bk; jc++) {
                    ec = ic + jc;
                    LIBXS_VLA_ACCESS(2, dri, ec, ek, K) += LIBXS_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXS_VLA_ACCESS(3, di, j, en, ek, N, K);
                    LIBXS_VLA_ACCESS(2, drf, ec, ek, K) += LIBXS_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXS_VLA_ACCESS(3, df, j, en, ek, N, K);
                    LIBXS_VLA_ACCESS(2, dro, ec, ek, K) += LIBXS_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXS_VLA_ACCESS(3, dp, j, en, ek, N, K);
                    LIBXS_VLA_ACCESS(2, drc, ec, ek, K) += LIBXS_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXS_VLA_ACCESS(3, dc, j, en, ek, N, K);
                  }
                }
              }
              /* dw = delta * x^T */
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXS_VLA_ACCESS(2, dwi, ec, ek, K) += LIBXS_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXS_VLA_ACCESS(3, di, j, en, ek, N, K);
                  LIBXS_VLA_ACCESS(2, dwf, ec, ek, K) += LIBXS_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXS_VLA_ACCESS(3, df, j, en, ek, N, K);
                  LIBXS_VLA_ACCESS(2, dwo, ec, ek, K) += LIBXS_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXS_VLA_ACCESS(3, dp, j, en, ek, N, K);
                  LIBXS_VLA_ACCESS(2, dwc, ec, ek, K) += LIBXS_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXS_VLA_ACCESS(3, dc, j, en, ek, N, K);
                }
              }
              if (j > 0) {
                dbi[ek] += LIBXS_VLA_ACCESS(3, di, j, en, ek, N, K);
                dbf[ek] += LIBXS_VLA_ACCESS(3, df, j, en, ek, N, K);
                dbo[ek] += LIBXS_VLA_ACCESS(3, dp, j, en, ek, N, K);
                dbc[ek] += LIBXS_VLA_ACCESS(3, dc, j, en, ek, N, K);
              }
            }
          }
        }
      }
    }
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_execute_st(libxs_dnn_lstmcell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           status = libxs_dnn_lstmcell_fwd(handle, start_thread, tid);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                           status = libxs_dnn_lstmcell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                           status = libxs_dnn_lstmcell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           status = libxs_dnn_lstmcell_bwd_upd_bu(handle, start_thread, tid, 3);
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

