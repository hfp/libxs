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


LIBXS_API libxs_dnn_lstmcell* libxs_dnn_create_lstmcell(libxs_dnn_lstmcell_desc lstmcell_desc, libxs_dnn_err_t* status)
{
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
    handle->handlewx = lstmcell_desc.handlewx;
    handle->handleuh = lstmcell_desc.handleuh;
    handle->handlett = lstmcell_desc.handlett;
    handle->handlewd = lstmcell_desc.handlewd;
    /* Need to allocate space for scratch and internalstate libxs_dnn_tensor's */
    handle->i1t = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->i1b = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->i2 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->f1t = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->f1b = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->f2 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->o1t = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->o1b = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->o2 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->c1t = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->c1b = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->c2 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->i = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->f = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->o = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->c = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->dh = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->d1 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->d2 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->d = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->i3 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->f3 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->d4 = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdht = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->deltat = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djddt = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdit = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdft = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdct = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdot = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdxt = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdwi = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdwf = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdwo = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdwc = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdri = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdrf = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdro = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdrc = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdbi = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdbf = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdbo = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->djdbc = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->rTp = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->wTp = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->deltaTp = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
    handle->xTp = (libxs_dnn_tensor*)libxs_malloc(sizeof(libxs_dnn_tensor));
  } else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_lstmcell(const libxs_dnn_lstmcell* handle)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  if (0 != handle) {
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


LIBXS_API size_t libxs_dnn_lstmcell_get_scratch_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* scratch)
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind)
{
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


LIBXS_API size_t libxs_dnn_lstmcell_get_internalstate_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* internalstate)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)internalstate;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind)
{
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type)
{
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


LIBXS_API libxs_dnn_tensor* libxs_dnn_lstmcell_get_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor* tensor = 0;
  LIBXS_UNUSED(status/*TODO*/);

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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type)
{
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
  LIBXS_DNN_ELTWISE_FTYPE *wi = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wi->data;
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXS_DNN_ELTWISE_FTYPE *wf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXS_DNN_ELTWISE_FTYPE *wo = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXS_DNN_ELTWISE_FTYPE *wc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wc->data;
#endif
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ri = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXS_DNN_ELTWISE_FTYPE *rf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXS_DNN_ELTWISE_FTYPE *ro = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXS_DNN_ELTWISE_FTYPE *rc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXS_DNN_ELTWISE_FTYPE *h = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *i2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i2->data;
  LIBXS_DNN_ELTWISE_FTYPE *f2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f2->data;
  LIBXS_DNN_ELTWISE_FTYPE *o2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o2->data;
  LIBXS_DNN_ELTWISE_FTYPE *c2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c2->data;
  LIBXS_DNN_ELTWISE_FTYPE *i = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXS_DNN_ELTWISE_FTYPE *f = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXS_DNN_ELTWISE_FTYPE *o = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXS_DNN_ELTWISE_FTYPE *c = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXS_DNN_ELTWISE_FTYPE *dh = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dh->data;
  LIBXS_DNN_ELTWISE_FTYPE *d1 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d1->data;
  LIBXS_DNN_ELTWISE_FTYPE *d2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d2->data;
  LIBXS_DNN_ELTWISE_FTYPE *d = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXS_DNN_ELTWISE_FTYPE *i1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *f1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *o1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *c1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c1t->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c1, c1t, m * n);
#else
  LIBXS_VLA_DECL(3, LIBXS_DNN_ELTWISE_FTYPE, i4,
    (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i1t->data, t, m * n);
  LIBXS_DNN_ELTWISE_FTYPE *i1t = &LIBXS_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n);
  LIBXS_DNN_ELTWISE_FTYPE *f1t = &LIBXS_VLA_ACCESS(3, i4, 1, 0, 0, t, m * n);
  LIBXS_DNN_ELTWISE_FTYPE *o1t = &LIBXS_VLA_ACCESS(3, i4, 2, 0, 0, t, m * n);
  LIBXS_DNN_ELTWISE_FTYPE *c1t = &LIBXS_VLA_ACCESS(3, i4, 3, 0, 0, t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c1, c1t, m * n);
#endif
  /* libxs_bgemm_handle *handlewx = lstm->handlewx; */
  libxs_bgemm_handle *handleuh = lstm->handleuh;
  libxs_bgemm_handle *handlett = lstm->handlett;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, inr, i, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, fnr, f, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, onr, o, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, cnr, c, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dnr, d, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  int j;

  LIBXS_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxs_timer_tick();
#endif
  if (reuse) {
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
      libxs_internal_recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 1, m * n, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      libxs_internal_matrix_eltwise_mult(m*n, f, d, d1, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_eltwise_mult(m*n, i, c, d2, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m*n, d1, d2, d, start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
      Gbl_t_nonlin = libxs_timer_tick();
#endif
      libxs_internal_matrix_relu(m*n, d, dh, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
      Gbl_t_nonlin_total += Gbl_duration_nonlin;
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      libxs_internal_matrix_eltwise_mult(m*n, o, dh, h, start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
    libxs_internal_recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, t-1, 0, m * n), i, i, 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
    libxs_internal_recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, t-1, 0, m * n), f, f, 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
    libxs_internal_recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, t-1, 0, m * n), o, o, 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
    libxs_internal_recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, t-1, 0, m * n), c, c, 1, m * n, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
    Gbl_t_eltwise = libxs_timer_tick();
#endif
    libxs_internal_matrix_eltwise_mult(m*n, f, d, d1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m*n, i, c, d2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_add(m*n, d1, d2, d, start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
    Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
    Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
  } else {
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
      libxs_internal_recursive_step(handleuh, ri, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), i2, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rf, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), f2, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, ro, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), o2, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rc, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), c2, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, dnr, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m*n, d1, d2, &LIBXS_VLA_ACCESS(2, dnr, j+1, 0, m * n), start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
      Gbl_t_nonlin = libxs_timer_tick();
#endif
      libxs_internal_matrix_relu(m*n, &LIBXS_VLA_ACCESS(2, dnr, j+1, 0, m * n), dh, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
      Gbl_t_nonlin_total += Gbl_duration_nonlin;
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), dh, &LIBXS_VLA_ACCESS(2, hnr, j+1, 0, m * n), start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
    libxs_internal_recursive_step(handleuh, ri, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), i2, &LIBXS_VLA_ACCESS(2, i1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, t-2, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
    libxs_internal_recursive_step(handleuh, rf, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), f2, &LIBXS_VLA_ACCESS(2, f1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, t-2, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
    libxs_internal_recursive_step(handleuh, ro, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), o2, &LIBXS_VLA_ACCESS(2, o1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, t-2, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*sigmoid*/
    libxs_internal_recursive_step(handleuh, rc, &LIBXS_VLA_ACCESS(2, hnr, t-1, 0, m * n), c2, &LIBXS_VLA_ACCESS(2, c1, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, t-2, 0, m * n), 1, m * n, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
    Gbl_t_eltwise = libxs_timer_tick();
#endif
    libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, dnr, t-1, 0, m * n), d1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, t-2, 0, m * n), d2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_add(m*n, d1, d2, &LIBXS_VLA_ACCESS(2, dnr, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
    Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
    Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
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
  LIBXS_DNN_ELTWISE_FTYPE *wi = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wi->data;
  LIBXS_DNN_ELTWISE_FTYPE *wf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXS_DNN_ELTWISE_FTYPE *wo = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXS_DNN_ELTWISE_FTYPE *wc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wc->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ri = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXS_DNN_ELTWISE_FTYPE *rf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXS_DNN_ELTWISE_FTYPE *ro = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXS_DNN_ELTWISE_FTYPE *rc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  /* LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->h->data; */
  LIBXS_DNN_ELTWISE_FTYPE *i1 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *i2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i2->data;
  LIBXS_DNN_ELTWISE_FTYPE *i3 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i3->data;
  LIBXS_DNN_ELTWISE_FTYPE *f1 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *f2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f2->data;
  LIBXS_DNN_ELTWISE_FTYPE *f3 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f3->data;
  LIBXS_DNN_ELTWISE_FTYPE *o1 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *o2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o2->data;
  LIBXS_DNN_ELTWISE_FTYPE *c1 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *c2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c2->data;
  LIBXS_DNN_ELTWISE_FTYPE *it = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXS_DNN_ELTWISE_FTYPE *ft = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXS_DNN_ELTWISE_FTYPE *ot = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXS_DNN_ELTWISE_FTYPE *ct = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXS_DNN_ELTWISE_FTYPE *d1 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d1->data;
  LIBXS_DNN_ELTWISE_FTYPE *d2 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d2->data;
  LIBXS_DNN_ELTWISE_FTYPE *d3 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dh->data;
  LIBXS_DNN_ELTWISE_FTYPE *d4 = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d4->data;
  LIBXS_DNN_ELTWISE_FTYPE *dt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdht = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdht->data;
  LIBXS_DNN_ELTWISE_FTYPE *deltat = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->deltat->data;
  LIBXS_DNN_ELTWISE_FTYPE *djddt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djddt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdit = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdit->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdft = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdft->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdct = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdct->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdot = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdot->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdxt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdxt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwi = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdwi->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdwf->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwo = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdwo->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdwc->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdri = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdri->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdrf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdrf->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdro = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdro->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdrc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdrc->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbi = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdbi->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdbf->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbo = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdbo->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdbc->data;
  /*
  LIBXS_DNN_ELTWISE_FTYPE *rTp = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rTp->data;
  LIBXS_DNN_ELTWISE_FTYPE *wTp = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wTp->data;
  LIBXS_DNN_ELTWISE_FTYPE *deltaTp = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->deltaTp->data;
  LIBXS_DNN_ELTWISE_FTYPE *xTp = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xTp->data;
  */
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
  /* LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, h, ht, m * n); */
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i, it, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f, ft, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o, ot, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c, ct, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d, dt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdh, djdht, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, delta, deltat, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdd, djddt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdi, djdit, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdf, djdft, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdo, djdot, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdc, djdct, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdx, djdxt, k * n);
  libxs_bgemm_handle *handleud = lstm->handlewx;
  libxs_bgemm_handle *handledh = lstm->handleuh;
  libxs_bgemm_handle *handledx = lstm->handlett;
  libxs_bgemm_handle *handlewd = lstm->handlewd;
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
#endif
  int j;

  LIBXS_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxs_timer_tick();
#endif
  /* compute delta */
  libxs_internal_matrix_copy(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdd */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), d1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), d2, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdc */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), c1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), c2, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdi */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), i1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdf */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, d, t-2, 0, m * n), f1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdo */
  libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), o1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  if (pass == 1 || pass == 3) {
    /* compute djdx */
    /* libxs_internal_matrix_transpose(m, k, wi, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
    libxs_bgemm(handlewd, wi, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
    /* libxs_internal_matrix_transpose(m, k, wf, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
    libxs_bgemm(handlewd, wf, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
    /* libxs_internal_matrix_transpose(m, k, wo, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
    libxs_bgemm(handlewd, wo, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
    /* libxs_internal_matrix_transpose(m, k, wc, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
    libxs_bgemm(handlewd, wc, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
  }
  for (j = t-2; j >= 0; --j) {
    /* compute delta */
    /* libxs_internal_matrix_transpose(m, m, ri, rTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
    libxs_bgemm(handleud, ri, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
    /* libxs_internal_matrix_transpose(m, m, rf, rTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
    libxs_bgemm(handleud, rf, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
    /* libxs_internal_matrix_transpose(m, m, ro, rTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
    libxs_bgemm(handleud, ro, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
    /* libxs_internal_matrix_transpose(m, m, rc, rTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
    /* libxs_bgemm(handleud, rTp, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
    libxs_bgemm(handleud, rc, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
    libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdd */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, d1, d2, d3, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, f, j+1, 0, m * n), d4, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_add(m * n, d3, d4, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdc */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), c1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), c2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdi */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), i1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdf */
    if (j >= 1) {
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, d, j-1, 0, m * n), f1, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
    } else {
      /* djdf is zero for j == 0 */
      libxs_internal_matrix_zero(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
    }
    /* compute djdo */
    libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), o1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), start_thread, tid, lstm->nThreads);
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      /* libxs_internal_matrix_transpose(m, k, wi, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wi, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, k, wf, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wf, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, k, wo, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wo, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
      /* libxs_internal_matrix_transpose(m, k, wc, wTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
      /* libxs_bgemm(handlewd, wTp, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
      libxs_bgemm(handlewd, wc, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
    }
  }
  if (pass == 2 || pass == 3) {
    /* compute djdw */
    for (j = 0; j < t; ++j) {
      /* libxs_internal_matrix_transpose(k, n, &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), xTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
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
      /* libxs_internal_matrix_transpose(m, n, &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), deltaTp, start_thread, tid, lstm->nThreads); - already taken care of in init */
      libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdi, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdri, tid, lstm->nThreads);
      libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdf, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdrf, tid, lstm->nThreads);
      libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdo, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdro, tid, lstm->nThreads);
      libxs_bgemm(handledh, &LIBXS_VLA_ACCESS(2, djdc, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), djdrc, tid, lstm->nThreads);
    }
    /* compute djdb */
    for (j = 0; j < t-1; j++) {
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc, start_thread, tid, lstm->nThreads);
    }
  }
#if defined(LSTM_TIMING)
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#endif

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

