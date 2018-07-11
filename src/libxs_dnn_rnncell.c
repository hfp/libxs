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

#if defined(LSTM_TIMING)
#include <stdio.h>
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif


LIBXS_API libxs_dnn_rnncell* libxs_dnn_create_rnncell(libxs_dnn_rnncell_desc rnncell_desc, libxs_dnn_err_t* status)
{
  libxs_dnn_rnncell* handle = 0;
  *status = LIBXS_DNN_SUCCESS;

  handle = (libxs_dnn_rnncell*)malloc(sizeof(libxs_dnn_rnncell));
  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->nThreads = rnncell_desc.nThreads;
    handle->desc = rnncell_desc;
    handle->datatype_in = rnncell_desc.datatype_in;
    handle->datatype_out = rnncell_desc.datatype_out;
    handle->reuse = rnncell_desc.reuse;
    if ( (rnncell_desc.datatype_in != LIBXS_DNN_DATATYPE_F32) || (rnncell_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = rnncell_desc.buffer_format;
    handle->custom_format_type = LIBXS_DNN_TENSOR_FORMAT_LIBXS_1; /* required only for comparing layouts */
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
    handle->handlewx = rnncell_desc.handlewx;
    handle->handleuh = rnncell_desc.handleuh;
    handle->handlett = rnncell_desc.handlett;
    handle->handlewd = rnncell_desc.handlewd;
    /* Need to allocate space for scratch libxs_dnn_tensor's */
    handle->z   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->deltat = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->z1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->z2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->di1 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->di2 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->deltaMt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    if (NULL == handle->deltat || NULL == handle->deltaMt ||
        NULL == handle->z || NULL == handle->z1t || NULL == handle->z2 ||
        NULL == handle->di1 || NULL == handle->di2)
    {
      free(handle->deltat); free(handle->deltaMt);
      free(handle->z); free(handle->z1t); free(handle->z2);
      free(handle->di1); free(handle->di2);
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
    free(handle->deltat); free(handle->deltaMt);
    free(handle->z); free(handle->z1t); free(handle->z2);
    free(handle->di1); free(handle->di2);
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_rnncell*)handle);
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
      layout->custom_format = handle->custom_format_type;
      if ( (type == LIBXS_DNN_RNN_REGULAR_INPUT)        || (type == LIBXS_DNN_RNN_GRADIENT_INPUT)  ||
           (type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) ||
           (type == LIBXS_DNN_RNN_REGULAR_WEIGHT)       || (type == LIBXS_DNN_RNN_GRADIENT_WEIGHT) ||
           (type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
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
                if ( (type == LIBXS_DNN_RNN_REGULAR_INPUT) || (type == LIBXS_DNN_RNN_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bk;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXS_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXS_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bm;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->m / handle->bm;
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


LIBXS_API size_t libxs_dnn_rnncell_get_scratch_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z2 */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* z2, zi */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* deltat */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* di1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* di2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* deltaMt */
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
  size_t sizeof_datatype = sizeof(float);

  if (scratch == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           if (address % 64 == 0) {
                                             handle->z1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->z1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltat->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltat->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->di1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->di1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->di2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->di2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltaMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltaMt->data = (void*)(address+offset);
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
                                           handle->z1t->data = 0;
                                           handle->z2->data = 0;
                                           handle->z1t = 0;
                                           handle->z2 = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->z1t->data = 0;
                                           handle->z2->data = 0;
                                           handle->deltat->data = 0;
                                           handle->di1->data = 0;
                                           handle->di2->data = 0;
                                           handle->deltaMt->data = 0;
                                           handle->z1t = 0;
                                           handle->z2 = 0;
                                           handle->deltat = 0;
                                           handle->di1 = 0;
                                           handle->di2 = 0;
                                           handle->deltaMt = 0;
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
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* zt */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* zt */
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


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_assign_z(libxs_dnn_rnncell* handle, const void* zgoldtb)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && zgoldtb != 0) {
    const libxs_blasint m = handle->m, n = handle->n, t = handle->t;
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, zgold, (const LIBXS_DNN_ELTWISE_FTYPE*)zgoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z, (LIBXS_DNN_ELTWISE_FTYPE*)handle->z->data, m * n);
    libxs_blasint it;
    for (it = 0; it < t; ++it) {
      libxs_bgemm_copyin_b(handle->handlewx, &LIBXS_VLA_ACCESS(2, zgold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, z, it, 0, m * n));
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
      (type != LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
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
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXS_DNN_ERR_MISMATCH_TENSOR;
    }

    /* libxs_dnn_destroy_tensor_datalayout( handle_layout ); */
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
      (type != LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
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
      (type != LIBXS_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXS_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
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
  libxs_blasint m = rnn->m;
  libxs_blasint n = rnn->n;
  libxs_blasint k = rnn->k;
  libxs_blasint t = rnn->t;
#if defined(LSTM_TIMING)
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
#endif
  int reuse = rnn->reuse;
  /* The following code should be in template */
  LIBXS_DNN_ELTWISE_FTYPE *w = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *u = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXS_DNN_ELTWISE_FTYPE *h = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *z1t = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *z2t = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z2->data;
  LIBXS_DNN_ELTWISE_FTYPE *z = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z->data;
  /* libxs_bgemm_handle *handlewx = rnn->handlewx; */
  libxs_bgemm_handle *handleuh = rnn->handleuh;
  libxs_bgemm_handle *handlett = rnn->handlett;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z1, z1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z2, z2t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, znr, z, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  int i;

#if defined(LSTM_TIMING)
  start = libxs_timer_tick();
  Gbl_t_input = libxs_timer_tick();
#endif
  libxs_bgemm_st(handlett, w, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), start_thread, tid);
#if defined(LSTM_TIMING)
  Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
  Gbl_t_input_total += Gbl_duration_input;
#endif
  if (reuse) {
    for (i = 0; i < t; ++i) {
      /* printf("i %d, t %d\n", i, t); */
      libxs_internal_recursive_step(handleuh, u, h, &LIBXS_VLA_ACCESS(2, z2, i, 0, m * n), &LIBXS_VLA_ACCESS(2, z1, i, 0, m * n), z, h, 2, m * n, start_thread, tid); /*sigmoid*/
    }
  } else {
    for (i = 0; i < t; ++i) {
      libxs_internal_recursive_step(handleuh, u, &LIBXS_VLA_ACCESS(2, hnr, i, 0, m * n), &LIBXS_VLA_ACCESS(2, z2, i, 0, m * n), &LIBXS_VLA_ACCESS(2, z1, i, 0, m * n),
        &LIBXS_VLA_ACCESS(2, znr, i, 0, m * n), &LIBXS_VLA_ACCESS(2, hnr, i+1, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
    }
  }
#if defined(LSTM_TIMING)
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops / duration); /* *nrepeat */
  }
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif
  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bwd_upd_bu(libxs_dnn_rnncell* rnn, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = rnn->m;
  libxs_blasint n = rnn->n;
  libxs_blasint k = rnn->k;
  libxs_blasint t = rnn->t;
#ifdef LSTM_TIMING
  const double tflops = 12; /* transcendental flops */
  double gflops = m * m; /* U^T */
  gflops += (2.0 * m * n * m); /* U^T * delta */
  gflops += (m * n); /* dJdh + (U^T * delta) */
  gflops += (tflops * m * n); /* sigma'(Z) */
  gflops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
  gflops *= t; /* for t time steps */
  double tempflops;
  if (pass == 2 || pass == 3) {
    tempflops = m * n; /* h^T */
    tempflops += (2.0 * m * n * m); /* delta * h^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * m * (t-1)); /* for summation of dJdU */
    gflops += tempflops;
    tempflops = k * n; /* x^T */
    tempflops += (2.0 * m * n * k); /* delta * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * k * (t-1)); /* for summation of dJdW */
    gflops += tempflops;
  }
  if (pass == 1 || pass == 3) {
    tempflops = m * k; /* W^T */
    tempflops += (2.0 * m * n * k); /* W^T * delta */
    tempflops *= t; /* for t time steps of input */
    gflops += tempflops;
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
#endif
  LIBXS_DNN_ELTWISE_FTYPE *djdht = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdht->data;
  LIBXS_DNN_ELTWISE_FTYPE *zt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z->data;
  LIBXS_DNN_ELTWISE_FTYPE *deltat = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->deltat->data;
  LIBXS_DNN_ELTWISE_FTYPE *u = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *w = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdu = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdu->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdw = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdw->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdxt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->djdxt->data;
  LIBXS_DNN_ELTWISE_FTYPE* zi = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->z1t->data;
  LIBXS_DNN_ELTWISE_FTYPE* di1t = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->di1->data;
  LIBXS_DNN_ELTWISE_FTYPE* di2 = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->di2->data;
  LIBXS_DNN_ELTWISE_FTYPE* deltaMt = (LIBXS_DNN_ELTWISE_FTYPE*)rnn->deltaMt->data;
  libxs_bgemm_handle *handleud = rnn->handlewx;
  libxs_bgemm_handle *handledh = rnn->handleuh;
  libxs_bgemm_handle *handledx = rnn->handlett;
  libxs_bgemm_handle *handlewd = rnn->handlewd;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdh, djdht, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z, zt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, delta, deltat, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, h, ht, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdx, djdxt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, di1, di1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, deltaM, deltaMt, m * n);

  int i;
#ifdef LSTM_TIMING
  unsigned long long start;
  double duration;
  start = libxs_timer_tick();
#endif

  libxs_internal_matrix_zero(m * n, &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, rnn->nThreads);
  for (i = t-2; i >= 0; --i) {
    libxs_internal_matrix_sigmoid_inverse(m * n, &LIBXS_VLA_ACCESS(2, z, i+1, 0, m * n), zi, start_thread, tid, rnn->nThreads);
    libxs_bgemm_st(handleud, u, &LIBXS_VLA_ACCESS(2, delta, i+1, 0, m * n), &LIBXS_VLA_ACCESS(2, di1, i, 0, m * n), start_thread, tid);
    libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdh, i+1, 0, m * n), &LIBXS_VLA_ACCESS(2, di1, i, 0, m * n), di2, start_thread, tid, rnn->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, zi, di2, &LIBXS_VLA_ACCESS(2, delta, i, 0, m * n), start_thread, tid, rnn->nThreads);
  }
  if (pass == 1 || pass == 3) {
    for (i = 0; i < t; ++i) {
      libxs_bgemm_st(handlewd, w, &LIBXS_VLA_ACCESS(2, delta, i, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, i, 0, k * n), start_thread, tid);
    }
  }
  if (pass == 2 || pass == 3) {
    for (i = 0; i < t; ++i) {
      libxs_bgemm_convert_b_to_a(handleud, &LIBXS_VLA_ACCESS(2, delta, i, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, deltaM, i, 0, m * n));
    }
    for (i = 0; i < t; ++i) {
      libxs_bgemm_st(handledh, &LIBXS_VLA_ACCESS(2, deltaM, i, 0, m * n), &LIBXS_VLA_ACCESS(2, h, i, 0, m * n), djdu, start_thread, tid);
      libxs_bgemm_st(handledx, &LIBXS_VLA_ACCESS(2, deltaM, i, 0, m * n), &LIBXS_VLA_ACCESS(2, x, i, 0, k * m), djdw, start_thread, tid);
    }
  }
#ifdef LSTM_TIMING
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops / duration); /* *nrepeat */
  }
#endif

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
