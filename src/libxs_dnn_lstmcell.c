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
#include "libxs_dnn_elementwise.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(FTYPE)
# define FTYPE float /* TODO: undefine/remove generic symbol names as header-only interfers with user's code */
#endif

#if defined(LSTM_TIMING)
#include <stdio.h>
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif


LIBXS_API libxs_dnn_lstmcell* libxs_dnn_create_lstmcell(libxs_dnn_lstmcell_desc lstmcell_desc, libxs_dnn_err_t* status) {
  libxs_dnn_lstmcell* handle = 0;
  *status = LIBXS_DNN_SUCCESS;

  handle = (libxs_dnn_lstmcell*)malloc(sizeof(libxs_dnn_lstmcell));
  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = lstmcell_desc;
    handle->datatype_in = lstmcell_desc.datatype_in;
    handle->datatype_out = lstmcell_desc.datatype_out;
    if ( (lstmcell_desc.datatype_in != LIBXS_DNN_DATATYPE_F32) || (lstmcell_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = lstmcell_desc.buffer_format;
    handle->m = lstmcell_desc.m;
    handle->n = lstmcell_desc.n;
    handle->k = lstmcell_desc.k;
    handle->t = lstmcell_desc.t;
    if (lstmcell_desc.t < 2) {
      *status = LIBXS_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bm = lstmcell_desc.bm;
    handle->bn = lstmcell_desc.bn;
    handle->bk = lstmcell_desc.bk;
    handle->b_m1 = lstmcell_desc.b_m1;
    handle->b_n1 = lstmcell_desc.b_n1;
    handle->b_k1 = lstmcell_desc.b_k1;
    handle->b_m2 = lstmcell_desc.b_m2;
    handle->b_n2 = lstmcell_desc.b_n2;
    handle->b_k2 = lstmcell_desc.b_k2;
  } else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_lstmcell(const libxs_dnn_lstmcell* handle) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  if (0 != handle) {
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_lstmcell*)handle);
  }
  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_lstmcell_create_tensor_datalayout(const libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor_datalayout* layout = 0;
  *status = LIBXS_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXS_DNN_LSTM_REGULAR_INPUT)          || (type == LIBXS_DNN_LSTM_GRADIENT_INPUT)  ||
           (type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)   || (type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE) || 
           (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_I)       || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I) ||
           (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_F)       || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F) ||
           (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_O)       || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O) ||
           (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_C)       || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C) ||
           (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) ||
           (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) ||
           (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) ||
           (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) ||
           (type == LIBXS_DNN_LSTM_REGULAR_BIAS_I)         || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_I)   ||
           (type == LIBXS_DNN_LSTM_REGULAR_BIAS_F)         || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_F)   ||
           (type == LIBXS_DNN_LSTM_REGULAR_BIAS_O)         || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_O)   ||
           (type == LIBXS_DNN_LSTM_REGULAR_BIAS_C)         || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_C) ) {
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
                /* TODO: Check if the following layout works for bwd and upd passes */
                if ( (type == LIBXS_DNN_LSTM_REGULAR_INPUT) || (type == LIBXS_DNN_LSTM_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bk;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_I) || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I) || 
                            (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_F) || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_O) || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_C) || (type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) || 
                            (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) || (type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bm;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXS_DNN_LSTM_REGULAR_BIAS_I) || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_I) || 
                            (type == LIBXS_DNN_LSTM_REGULAR_BIAS_F) || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_F) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_BIAS_O) || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_O) ||
                            (type == LIBXS_DNN_LSTM_REGULAR_BIAS_C) || (type == LIBXS_DNN_LSTM_GRADIENT_BIAS_C) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
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


LIBXS_API size_t libxs_dnn_lstmcell_get_scratch_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status) {
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
#if defined(NON_FUSED_INPUT_GEMM)
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
#else
                                           size += handle->m * 4 * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += 0; /* f1t */
                                           size += 0; /* o1t */
                                           size += 0; /* c1t */
#endif
                                           size += handle->m * handle->n * sizeof_datatype; /* i2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* dh */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
#if defined(NON_FUSED_INPUT_GEMM)
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
#else
                                           size += handle->m * 4 * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += 0; /* f1t */
                                           size += 0; /* o1t */
                                           size += 0; /* c1t */
#endif
                                           size += handle->m * handle->n * sizeof_datatype; /* i1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* i2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* i3 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f3 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d3 (dh) */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d4 */
                                           size += 64;
                                           size += handle->m * handle->k * sizeof_datatype; /* wTp */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* rTp */
                                           size += 64;
                                           size += handle->k * handle->n * sizeof_datatype; /* xTp */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* deltaTp */
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* scratch) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  size_t address = (size_t)scratch;
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
#if defined(NON_FUSED_INPUT_GEMM)
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#else
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           handle->f1t->data = handle->i1t->data; /* not used */
                                           handle->o1t->data = handle->i1t->data; /* not used */
                                           handle->c1t->data = handle->i1t->data; /* not used */
                                           scratch_size = handle->m * 4 * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#endif
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dh->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dh->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
#if defined(NON_FUSED_INPUT_GEMM)
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#else
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           handle->f1t->data = handle->i1t->data; /* not used */
                                           handle->o1t->data = handle->i1t->data; /* not used */
                                           handle->c1t->data = handle->i1t->data; /* not used */
                                           scratch_size = handle->m * 4 * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#endif
                                           if (address % 64 == 0) {
                                             handle->i1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i3->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f3->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dh->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dh->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d4->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d4->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->wTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->wTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->k * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->rTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->rTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->xTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->xTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->k * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltaTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltaTp->data = (void*)(address+offset);
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           handle->i1t->data = 0;
                                           handle->i2->data = 0;
                                           handle->f1t->data = 0;
                                           handle->f2->data = 0;
                                           handle->o1t->data = 0;
                                           handle->o2->data = 0;
                                           handle->c1t->data = 0;
                                           handle->c2->data = 0;
                                           handle->d1->data = 0;
                                           handle->d2->data = 0;
                                           handle->dh->data = 0;
                                           handle->i1t = 0;
                                           handle->i2 = 0;
                                           handle->f1t = 0;
                                           handle->f2 = 0;
                                           handle->o1t = 0;
                                           handle->o2 = 0;
                                           handle->c1t = 0;
                                           handle->c2 = 0;
                                           handle->d1 = 0;
                                           handle->d2 = 0;
                                           handle->dh = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->i1t->data = 0;
                                           handle->i1b->data = 0;
                                           handle->i2->data = 0;
                                           handle->f1t->data = 0;
                                           handle->f1b->data = 0;
                                           handle->f2->data = 0;
                                           handle->o1t->data = 0;
                                           handle->o1b->data = 0;
                                           handle->o2->data = 0;
                                           handle->c1t->data = 0;
                                           handle->c1b->data = 0;
                                           handle->c2->data = 0;
                                           handle->d1->data = 0;
                                           handle->d2->data = 0;
                                           handle->dh->data = 0;
                                           handle->i3->data = 0;
                                           handle->f3->data = 0;
                                           handle->d4->data = 0;
                                           handle->rTp->data = 0;
                                           handle->wTp->data = 0;
                                           handle->xTp->data = 0;
                                           handle->deltaTp->data = 0;
                                           handle->i1t = 0;
                                           handle->i1b = 0;
                                           handle->i2 = 0;
                                           handle->f1t = 0;
                                           handle->f1b = 0;
                                           handle->f2 = 0;
                                           handle->o1t = 0;
                                           handle->o1b = 0;
                                           handle->o2 = 0;
                                           handle->c1t = 0;
                                           handle->c1b = 0;
                                           handle->c2 = 0;
                                           handle->d1 = 0;
                                           handle->d2 = 0;
                                           handle->dh = 0;
                                           handle->i3 = 0;
                                           handle->f3 = 0;
                                           handle->d4 = 0;
                                           handle->rTp = 0;
                                           handle->wTp = 0;
                                           handle->xTp = 0;
                                           handle->deltaTp = 0;
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


LIBXS_API size_t libxs_dnn_lstmcell_get_internalstate_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status) {
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype; /* i */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* d */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djddt */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdit */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdft */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdct */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdot */
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* internalstate) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  size_t address = (size_t)internalstate;
  size_t offset = 0;
  size_t scratch_size = 0;
  size_t sizeof_datatype = sizeof(float);

  if (internalstate == 0) {
    status = LIBXS_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           if (address % 64 == 0) {
                                             handle->i->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->i->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djddt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djddt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdit->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdit->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdft->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdft->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdot->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdot->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdct->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdct->data = (void*)(address+offset);
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           handle->i->data = 0;
                                           handle->f->data = 0;
                                           handle->o->data = 0;
                                           handle->c->data = 0;
                                           handle->d->data = 0;
                                           handle->i = 0;
                                           handle->f = 0;
                                           handle->o = 0;
                                           handle->c = 0;
                                           handle->d = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->i->data = 0;
                                           handle->f->data = 0;
                                           handle->o->data = 0;
                                           handle->c->data = 0;
                                           handle->d->data = 0;
                                           handle->djddt->data = 0;
                                           handle->djdit->data = 0;
                                           handle->djdft->data = 0;
                                           handle->djdot->data = 0;
                                           handle->djdct->data = 0;
                                           handle->i = 0;
                                           handle->f = 0;
                                           handle->o = 0;
                                           handle->c = 0;
                                           handle->d = 0;
                                           handle->djddt = 0;
                                           handle->djdit = 0;
                                           handle->djdft = 0;
                                           handle->djdot = 0;
                                           handle->djdct = 0;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_LSTM_REGULAR_INPUT)         && (type != LIBXS_DNN_LSTM_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)   && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_I)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_F)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_O)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_C)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_I)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_I)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_F)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_F)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_O)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_O)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_C)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_C) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_lstmcell_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_LSTM_REGULAR_INPUT ) {
        handle->xt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_INPUT ) {
        handle->djdxt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
        handle->h = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
        handle->djdht = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_I ) {
        handle->wi = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I ) {
        handle->djdwi = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_F ) {
        handle->wf = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F ) {
        handle->djdwf = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_O ) {
        handle->wo = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O ) {
        handle->djdwo = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_C ) {
        handle->wc = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C ) {
        handle->djdwc = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I ) {
        handle->ri = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I ) {
        handle->djdri = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F ) {
        handle->rf = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F ) {
        handle->djdrf = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O ) {
        handle->ro = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O ) {
        handle->djdro = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C ) {
        handle->rc = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C ) {
        handle->djdrc = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_I ) {
        handle->bi = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_I ) {
        handle->djdbi = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_F ) {
        handle->bf = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_F ) {
        handle->djdbf = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_O ) {
        handle->bo = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_O ) {
        handle->djdbo = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_C ) {
        handle->bc = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_C ) {
        handle->djdbc = (libxs_dnn_tensor*)tensor;
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


LIBXS_API libxs_dnn_tensor* libxs_dnn_lstmcell_get_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status) {
  libxs_dnn_tensor* tensor = 0;
  /* check for tensor type */
  if ( (type != LIBXS_DNN_LSTM_REGULAR_INPUT)         && (type != LIBXS_DNN_LSTM_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)   && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_I)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_F)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_O)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_C)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_I)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_I)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_F)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_F)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_O)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_O)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_C)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_C) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_LSTM_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_INPUT ) {
      tensor = handle->djdxt;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      tensor = handle->h;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->djdht;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_I ) {
      tensor = handle->wi;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I ) {
      tensor = handle->djdwi;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_F ) {
      tensor = handle->wf;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F ) {
      tensor = handle->djdwf;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_O ) {
      tensor = handle->wo;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O ) {
      tensor = handle->djdwo;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_C ) {
      tensor = handle->wc;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C ) {
      tensor = handle->djdwc;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I ) {
      tensor = handle->ri;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I ) {
      tensor = handle->djdri;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F ) {
      tensor = handle->rf;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F ) {
      tensor = handle->djdrf;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O ) {
      tensor = handle->ro;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O ) {
      tensor = handle->djdro;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C ) {
      tensor = handle->rc;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C ) {
      tensor = handle->djdrc;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_I ) {
      tensor = handle->bi;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_I ) {
      tensor = handle->djdbi;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_F ) {
      tensor = handle->bf;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_F ) {
      tensor = handle->djdbf;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_O ) {
      tensor = handle->bo;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_O ) {
      tensor = handle->djdbo;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_C ) {
      tensor = handle->bc;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_C ) {
      tensor = handle->djdbc;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type) {
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_LSTM_REGULAR_INPUT)         && (type != LIBXS_DNN_LSTM_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE)   && (type != LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_I)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_F)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_O)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O) &&
      (type != LIBXS_DNN_LSTM_REGULAR_WEIGHT_C)       && (type != LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) &&
      (type != LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) && (type != LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_I)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_I)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_F)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_F)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_O)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_O)   &&
      (type != LIBXS_DNN_LSTM_REGULAR_BIAS_C)         && (type != LIBXS_DNN_LSTM_GRADIENT_BIAS_C) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_LSTM_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_INPUT ) {
      handle->djdxt = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      handle->h = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      handle->djdht = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_I ) {
      handle->wi = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_I ) {
      handle->djdwi = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_F ) {
      handle->wf = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_F ) {
      handle->djdwf = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_O ) {
      handle->wo = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_O ) {
      handle->djdwo = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_WEIGHT_C ) {
      handle->wc = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_WEIGHT_C ) {
      handle->djdwc = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_I ) {
      handle->ri = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I ) {
      handle->djdri = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_F ) {
      handle->rf = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F ) {
      handle->djdrf = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_O ) {
      handle->ro = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O ) {
      handle->djdro = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_RECUR_WEIGHT_C ) {
      handle->rc = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C ) {
      handle->djdrc = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_I ) {
      handle->bi = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_I ) {
      handle->djdbi = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_F ) {
      handle->bf = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_F ) {
      handle->djdbf = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_O ) {
      handle->bo = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_O ) {
      handle->djdbo = 0;
    } else if ( type == LIBXS_DNN_LSTM_REGULAR_BIAS_C ) {
      handle->bc = 0;
    } else if ( type == LIBXS_DNN_LSTM_GRADIENT_BIAS_C ) {
      handle->djdbc = 0;
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
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  libxs_blasint k = lstm->k;
  libxs_blasint t = lstm->t;
#if defined(LSTM_TIMING)
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (5.0 * m * n)) * t * 1E-9;
#endif
  int reuse = 1;
  FTYPE *wi = (FTYPE*)lstm->wi->data;
#if defined(NON_FUSED_INPUT_GEMM)
  FTYPE *wf = (FTYPE*)lstm->wf->data;
  FTYPE *wo = (FTYPE*)lstm->wo->data;
  FTYPE *wc = (FTYPE*)lstm->wc->data;
#endif
  FTYPE *xt = (FTYPE*)lstm->xt->data;
  FTYPE *ri = (FTYPE*)lstm->ri->data;
  FTYPE *rf = (FTYPE*)lstm->rf->data;
  FTYPE *ro = (FTYPE*)lstm->ro->data;
  FTYPE *rc = (FTYPE*)lstm->rc->data;
  FTYPE *h = (FTYPE*)lstm->h->data;
  FTYPE *i1t = (FTYPE*)lstm->i1t->data;
  FTYPE *i2 = (FTYPE*)lstm->i2->data;
  FTYPE *f1t = (FTYPE*)lstm->f1t->data;
  FTYPE *f2 = (FTYPE*)lstm->f2->data;
  FTYPE *o1t = (FTYPE*)lstm->o1t->data;
  FTYPE *o2 = (FTYPE*)lstm->o2->data;
  FTYPE *c1t = (FTYPE*)lstm->c1t->data;
  FTYPE *c2 = (FTYPE*)lstm->c2->data;
  FTYPE *i = (FTYPE*)lstm->i->data;
  FTYPE *f = (FTYPE*)lstm->f->data;
  FTYPE *o = (FTYPE*)lstm->o->data;
  FTYPE *c = (FTYPE*)lstm->c->data;
  FTYPE *dh = (FTYPE*)lstm->dh->data;
  FTYPE *d1 = (FTYPE*)lstm->d1->data;
  FTYPE *d2 = (FTYPE*)lstm->d2->data;
  FTYPE *d = (FTYPE*)lstm->d->data;
  LIBXS_VLA_DECL(2, FTYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXS_VLA_DECL(2, FTYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, c1, c1t, m * n);
#else
  LIBXS_VLA_DECL(3, FTYPE, i4, i1t, t, m * n);
  i1t = &LIBXS_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n);
  f1t = &LIBXS_VLA_ACCESS(3, i4, 1, 0, 0, t, m * n);
  o1t = &LIBXS_VLA_ACCESS(3, i4, 2, 0, 0, t, m * n);
  c1t = &LIBXS_VLA_ACCESS(3, i4, 3, 0, 0, t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, FTYPE, c1, c1t, m * n);
#endif
  /* libxs_bgemm_handle *handlewx = lstm->handlewx; */
  libxs_bgemm_handle *handleuh = lstm->handleuh;
  libxs_bgemm_handle *handlett = lstm->handlett;
  LIBXS_VLA_DECL(2, FTYPE, hnr, h, m * n);
  LIBXS_VLA_DECL(2, FTYPE, inr, i, m * n);
  LIBXS_VLA_DECL(2, FTYPE, fnr, f, m * n);
  LIBXS_VLA_DECL(2, FTYPE, onr, o, m * n);
  LIBXS_VLA_DECL(2, FTYPE, cnr, c, m * n);
  LIBXS_VLA_DECL(2, FTYPE, dnr, d, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  /* int s; */
  int j;

  LIBXS_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxs_timer_tick();
#endif
  if (reuse) {
    /* for (s = 0; s < nrepeat; ++s) { */
#if defined(LSTM_TIMING)
      Gbl_t_input = libxs_timer_tick();
#endif
#if defined(NON_FUSED_INPUT_GEMM)
      libxs_bgemm(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), tid, lstm->nThreads);
      libxs_bgemm(handlett, wf, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), tid, lstm->nThreads);
      libxs_bgemm(handlett, wo, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), tid, lstm->nThreads);
      libxs_bgemm(handlett, wc, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), tid, lstm->nThreads);
#else
      libxs_bgemm(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), tid, lstm->nThreads);
#endif
#if defined(LSTM_TIMING)
      Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
#endif
      for (j = 0; j < t-1; ++j) {
        libxs_internal_recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxs_internal_recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxs_internal_recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxs_internal_recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_t_eltwise = libxs_timer_tick();
#endif
        libxs_internal_matrix_eltwise_mult(m*n, f, d, d1);
        libxs_internal_matrix_eltwise_mult(m*n, i, c, d2);
        libxs_internal_matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxs_timer_tick();
#endif
        libxs_internal_matrix_relu(m*n, d, dh); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxs_timer_tick();
#endif
        libxs_internal_matrix_eltwise_mult(m*n, o, dh, h);
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
      }
      libxs_internal_recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, t-1, 0, m * n), i, i, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, t-1, 0, m * n), f, f, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, t-1, 0, m * n), o, o, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, t-1, 0, m * n), c, c, 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      libxs_internal_matrix_eltwise_mult(m*n, f, d, d1);
      libxs_internal_matrix_eltwise_mult(m*n, i, c, d2);
      libxs_internal_matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    /* } */ /* end for nrepeat */
  } else {
    /* for (s = 0; s < nrepeat; ++s) { */
#if defined(LSTM_TIMING)
      Gbl_t_input = libxs_timer_tick();
#endif
#if defined(NON_FUSED_INPUT_GEMM)
      libxs_bgemm(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), tid, lstm->nThreads);
      libxs_bgemm(handlett, wf, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), tid, lstm->nThreads);
      libxs_bgemm(handlett, wo, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), tid, lstm->nThreads);
      libxs_bgemm(handlett, wc, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), tid, lstm->nThreads);
#else
      libxs_bgemm(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), tid, lstm->nThreads);
#endif
#if defined(LSTM_TIMING)
      Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
#endif
      for (j = 0; j < t-1; ++j) {
        libxs_internal_recursive_step(handleuh, ri, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), i2, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxs_internal_recursive_step(handleuh, rf, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), f2, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxs_internal_recursive_step(handleuh, ro, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), o2, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxs_internal_recursive_step(handleuh, rc, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), c2, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_t_eltwise = libxs_timer_tick();
#endif
        libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, dnr, j, 0, m * n), d1);
        libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), d2);
        libxs_internal_matrix_add(m*n, d1, d2, &LIBXS_VLA_ACCESS(2, dnr, j+1, 0, m * n));
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxs_timer_tick();
#endif
        libxs_internal_matrix_relu(m*n, &LIBXS_VLA_ACCESS(2, dnr, j+1, 0, m * n), dh); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxs_timer_tick();
#endif
        libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), dh, &LIBXS_VLA_ACCESS(2, hnr, j+1, 0, m * n));
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
      }
      libxs_internal_recursive_step(handleuh, ri, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), i2, &LIBXS_VLA_ACCESS(2, i1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rf, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), f2, &LIBXS_VLA_ACCESS(2, f1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, ro, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), o2, &LIBXS_VLA_ACCESS(2, o1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rc, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), c2, &LIBXS_VLA_ACCESS(2, c1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, dnr, t-1, 0, m * n), d1);
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, t-2, 0, m * n), d2);
      libxs_internal_matrix_add(m*n, d1, d2, &LIBXS_VLA_ACCESS(2, dnr, t-1, 0, m * n));
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    /* } */
  }
#if defined(LSTM_TIMING)
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bwd_upd_bu(libxs_dnn_lstmcell* lstm, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  libxs_blasint k = lstm->k;
  libxs_blasint t = lstm->t;
#if defined(LSTM_TIMING)
  const double tflops = 12; /* transcendental flops */
  double gflops = m * n; /* delta + delta_out */
  gflops += (6.0 * m * n + tflops * m * n); /* dJdd */
  gflops += (4.0 * m * n); /* dJdc */
  gflops += (4.0 * m * n); /* dJdi */
  gflops += (4.0 * m * n); /* dJdf */
  gflops += (4.0 * m * n + tflops * m * n); /* dJdo */
  double tempflops;
  if (pass == 1 || pass == 3) {
    tempflops += (4.0 * m * k); /* W^T */
    tempflops += (8.0 * m * n * k); /* W^T * dJd{c, i, f, o} */
    tempflops += (3.0 * m * k); /* summation */
    gflops += tempflops;
  }
  tempflops += (4.0 * m * m); /* R^T */
  tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
  gflops += tempflops;
  gflops *= t; /* for t time steps */
  if (pass == 2 || pass == 3) {
    tempflops = k * n; /* x^T */
    tempflops += (8.0 * m * n * k); /* delta{c, i, f, o} * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (4.0 * m * k * (t-1)); /* for summation of dJdW{c, i, f, o} */
    gflops += tempflops;
    tempflops = 4.0 * m * n; /* delta^T */
    tempflops += (8.0 * m * n * m); /* delta{c, i, f, o} * delta^T */
    tempflops *= (t - 1); /* for (t - 1) time steps */
    tempflops += (4.0 * m * n * (t-2)); /* for summation of dJdR{c, i, f, o} */
    gflops += tempflops;
    gflops += (4.0 * m * n * (t - 1)); /* delbias */
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
#endif
  FTYPE *wi = (FTYPE*)lstm->wi->data;
  FTYPE *wf = (FTYPE*)lstm->wf->data;
  FTYPE *wo = (FTYPE*)lstm->wo->data;
  FTYPE *wc = (FTYPE*)lstm->wc->data;
  FTYPE *xt = (FTYPE*)lstm->xt->data;
  FTYPE *ri = (FTYPE*)lstm->ri->data;
  FTYPE *rf = (FTYPE*)lstm->rf->data;
  FTYPE *ro = (FTYPE*)lstm->ro->data;
  FTYPE *rc = (FTYPE*)lstm->rc->data;
  /* FTYPE *ht = (FTYPE*)lstm->h->data; */
  FTYPE *i1 = (FTYPE*)lstm->i1t->data;
  FTYPE *i2 = (FTYPE*)lstm->i2->data;
  FTYPE *i3 = (FTYPE*)lstm->i3->data;
  FTYPE *f1 = (FTYPE*)lstm->f1t->data;
  FTYPE *f2 = (FTYPE*)lstm->f2->data;
  FTYPE *f3 = (FTYPE*)lstm->f3->data;
  FTYPE *o1 = (FTYPE*)lstm->o1t->data;
  FTYPE *o2 = (FTYPE*)lstm->o2->data;
  FTYPE *c1 = (FTYPE*)lstm->c1t->data;
  FTYPE *c2 = (FTYPE*)lstm->c2->data;
  FTYPE *it = (FTYPE*)lstm->i->data;
  FTYPE *ft = (FTYPE*)lstm->f->data;
  FTYPE *ot = (FTYPE*)lstm->o->data;
  FTYPE *ct = (FTYPE*)lstm->c->data;
  FTYPE *d1 = (FTYPE*)lstm->d1->data;
  FTYPE *d2 = (FTYPE*)lstm->d2->data;
  FTYPE *d3 = (FTYPE*)lstm->dh->data;
  FTYPE *d4 = (FTYPE*)lstm->d4->data;
  FTYPE *dt = (FTYPE*)lstm->d->data;
  FTYPE *djdht = (FTYPE*)lstm->djdht->data;
  FTYPE *deltat = (FTYPE*)lstm->deltat->data;
  FTYPE *djddt = (FTYPE*)lstm->djddt->data;
  FTYPE *djdit = (FTYPE*)lstm->djdit->data;
  FTYPE *djdft = (FTYPE*)lstm->djdft->data;
  FTYPE *djdct = (FTYPE*)lstm->djdct->data;
  FTYPE *djdot = (FTYPE*)lstm->djdot->data;
  FTYPE *djdxt = (FTYPE*)lstm->djdxt->data;
  FTYPE *djdwi = (FTYPE*)lstm->djdwi->data;
  FTYPE *djdwf = (FTYPE*)lstm->djdwf->data;
  FTYPE *djdwo = (FTYPE*)lstm->djdwo->data;
  FTYPE *djdwc = (FTYPE*)lstm->djdwc->data;
  FTYPE *djdri = (FTYPE*)lstm->djdri->data;
  FTYPE *djdrf = (FTYPE*)lstm->djdrf->data;
  FTYPE *djdro = (FTYPE*)lstm->djdro->data;
  FTYPE *djdrc = (FTYPE*)lstm->djdrc->data;
  FTYPE *djdbi = (FTYPE*)lstm->djdbi->data;
  FTYPE *djdbf = (FTYPE*)lstm->djdbf->data;
  FTYPE *djdbo = (FTYPE*)lstm->djdbo->data;
  FTYPE *djdbc = (FTYPE*)lstm->djdbc->data;
  /*
  FTYPE *rTp = (FTYPE*)lstm->rTp->data;
  FTYPE *wTp = (FTYPE*)lstm->wTp->data;
  FTYPE *deltaTp = (FTYPE*)lstm->deltaTp->data;
  FTYPE *xTp = (FTYPE*)lstm->xTp->data;
  */
  LIBXS_VLA_DECL(2, FTYPE, x, xt, k * n);
  /* LIBXS_VLA_DECL(2, FTYPE, h, ht, m * n); */
  LIBXS_VLA_DECL(2, FTYPE, i, it, m * n);
  LIBXS_VLA_DECL(2, FTYPE, f, ft, m * n);
  LIBXS_VLA_DECL(2, FTYPE, o, ot, m * n);
  LIBXS_VLA_DECL(2, FTYPE, c, ct, m * n);
  LIBXS_VLA_DECL(2, FTYPE, d, dt, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdh, djdht, m * n);
  LIBXS_VLA_DECL(2, FTYPE, delta, deltat, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdd, djddt, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdi, djdit, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdf, djdft, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdo, djdot, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdc, djdct, m * n);
  LIBXS_VLA_DECL(2, FTYPE, djdx, djdxt, k * n);
  libxs_bgemm_handle *handleud = lstm->handlewx;
  libxs_bgemm_handle *handledh = lstm->handleuh;
  libxs_bgemm_handle *handledx = lstm->handlett;
  libxs_bgemm_handle *handlewd = lstm->handlewd;
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
#endif
  /* int s; */
  int j;
  
  LIBXS_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxs_timer_tick();
#endif
  /* for (s = 0; s < nrepeat; ++s) { */
    /* compute delta */
    libxs_internal_matrix_copy(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n));
    /* compute djdd */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), d1);
    libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), d2);
    libxs_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n));
    /* compute djdc */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), c1);
    libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), c2);
    libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n));
    /* compute djdi */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), i1);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3);
    libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n));
    /* compute djdf */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, d, t-2, 0, m * n), f1);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3);
    libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n));
    /* compute djdo */
    libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), o1);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2);
    libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n));
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      /* libxs_internal_matrix_transpose(m, k, wi, wTp); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wi, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, k, wf, wTp); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wf, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, k, wo, wTp); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wo, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, k, wc, wTp); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wc, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
    }
    for (j = t-2; j >= 0; --j) {
      /* compute delta */
      /* libxs_internal_matrix_transpose(m, m, ri, rTp); - already taken care of in init */
      /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxs_bgemm(handleud, ri, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, m, rf, rTp); - already taken care of in init */
      /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxs_bgemm(handleud, rf, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, m, ro, rTp); - already taken care of in init */
      /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxs_bgemm(handleud, ro, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, m, rc, rTp); - already taken care of in init */
      /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxs_bgemm(handleud, rc, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n));
      /* compute djdd */
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), d1);
      libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), d2);
      libxs_internal_matrix_eltwise_mult(m * n, d1, d2, d3);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, f, j+1, 0, m * n), d4);
      libxs_internal_matrix_add(m * n, d3, d4, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n));
      /* compute djdc */
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), c1);
      libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), c2);
      libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n));
      /* compute djdi */
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), i1);
      libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, i3);
      libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n));
      /* compute djdf */
      if (j >= 1) {
        libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, d, j-1, 0, m * n), f1);
        libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2);
        libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, f3);
        libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n));
      } else {
        /* djdf is zero for j == 0 */
        libxs_internal_matinit( 0, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), m, n, m, 0.0);
      }
      /* compute djdo */
      libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), o1);
      libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, o2);
      libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n));
      if (pass == 1 || pass == 3) {
        /* compute djdx */
        /* libxs_internal_matrix_transpose(m, k, wi, wTp); - already taken care of in init */
        /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxs_bgemm(handlewd, wi, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
        /* libxs_internal_matrix_transpose(m, k, wf, wTp); - already taken care of in init */
        /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxs_bgemm(handlewd, wf, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
        /* libxs_internal_matrix_transpose(m, k, wo, wTp); - already taken care of in init */
        /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxs_bgemm(handlewd, wo, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
        /* libxs_internal_matrix_transpose(m, k, wc, wTp); - already taken care of in init */
        /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxs_bgemm(handlewd, wc, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
      }
    }
    if (pass == 2 || pass == 3) {
      /* compute djdw */
      for (j = 0; j < t; ++j) {
        /* libxs_internal_matrix_transpose(k, n, &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), xTp); - already taken care of in init */
        /*
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), xTp, djdwi, tid, lstm->nThreads);
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), xTp, djdwf, tid, lstm->nThreads);
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), xTp, djdwo, tid, lstm->nThreads);
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), xTp, djdwc, tid, lstm->nThreads);
        */
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwi, tid, lstm->nThreads);
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwf, tid, lstm->nThreads);
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwo, tid, lstm->nThreads);
        libxs_bgemm(handledx, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwc, tid, lstm->nThreads);
      }
      /* compute djdr */
      for (j = 0; j < t-1; ++j) {
        /* libxs_internal_matrix_transpose(m, n, &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), deltaTp); - already taken care of in init */
        libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdi, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdri, tid, lstm->nThreads);
        libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdf, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdrf, tid, lstm->nThreads);
        libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdo, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdro, tid, lstm->nThreads);
        libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdc, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdrc, tid, lstm->nThreads);
      }
      /* compute djdb */
      for (j = 0; j < t-1; j++) {
        libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi);
        libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf);
        libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo);
        libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc);
      }
    }
  /* } */
#if defined(LSTM_TIMING)
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#endif

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_execute_st(libxs_dnn_lstmcell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
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

