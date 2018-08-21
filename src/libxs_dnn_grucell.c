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

/* #define LSTM_TIMING */
#if defined(LSTM_TIMING)
#include <stdio.h>
extern double Gbl_t_input_total, Gbl_t_recur_total, Gbl_t_eltwise_total, Gbl_t_nonlin_total;
extern unsigned long long Gbl_t_input, Gbl_t_recur, Gbl_t_eltwise, Gbl_t_nonlin;
extern double Gbl_duration_input, Gbl_duration_recur, Gbl_duration_eltwise, Gbl_duration_nonlin;
#endif


LIBXS_API libxs_dnn_grucell* libxs_dnn_create_grucell(libxs_dnn_grucell_desc grucell_desc, libxs_dnn_err_t* status)
{
  libxs_dnn_grucell* handle = 0;
  const char *const env_b_m1 = getenv("LIBXS_BGEMM_M1");
  const int b_m1 = (0 == env_b_m1) ? 1 : atoi(env_b_m1);
  const char *const env_b_n1 = getenv("LIBXS_BGEMM_N1");
  const int b_n1 = (0 == env_b_n1) ? 1 : atoi(env_b_n1);
  const char *const env_b_k1 = getenv("LIBXS_BGEMM_K1");
  const int b_k1 = (0 == env_b_k1) ? 1 : atoi(env_b_k1);
  const char *const env_b_m2 = getenv("LIBXS_BGEMM_M2");
  const int b_m2 = (0 == env_b_m2) ? 1 : atoi(env_b_m2);
  const char *const env_b_n2 = getenv("LIBXS_BGEMM_N2");
  const int b_n2 = (0 == env_b_n2) ? 1 : atoi(env_b_n2);
  const char *const env_b_k2 = getenv("LIBXS_BGEMM_K2");
  const int b_k2 = (0 == env_b_k2) ? 1 : atoi(env_b_k2);
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const float alpha = 1, beta = 1;
  const libxs_bgemm_order order = (libxs_bgemm_order)0; /* denotes order of execution for bgemm */
  const libxs_gemm_prefetch_type strategy = (libxs_gemm_prefetch_type)LIBXS_PREFETCH_AUTO;

  handle = (libxs_dnn_grucell*)malloc(sizeof(libxs_dnn_grucell));
  if (0 != handle) {
    *status = LIBXS_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->nThreads = grucell_desc.nThreads;
    handle->desc = grucell_desc;
    handle->datatype_in = grucell_desc.datatype_in;
    handle->datatype_out = grucell_desc.datatype_out;
    handle->reuse = grucell_desc.reuse;
    handle->pass = grucell_desc.pass;
    if ( (grucell_desc.datatype_in != LIBXS_DNN_DATATYPE_F32) || (grucell_desc.datatype_out != LIBXS_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXS_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = grucell_desc.buffer_format;
    handle->m = grucell_desc.m;
    handle->n = grucell_desc.n;
    handle->k = grucell_desc.k;
    handle->t = grucell_desc.t;
    if (grucell_desc.t < 2) {
      *status = LIBXS_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bm = grucell_desc.bm;
    handle->bn = grucell_desc.bn;
    handle->bk = grucell_desc.bk;
    handle->b_m1 = b_m1;
    handle->b_n1 = b_n1;
    handle->b_k1 = b_k1;
    handle->b_m2 = b_m2;
    handle->b_n2 = b_n2;
    handle->b_k2 = b_k2;
    if (handle->pass == 0) {
      handle->handleux = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handlewh = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handlett = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
    } else {
      handle->handleux = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
      handle->handlewh = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->m, handle->n, &(handle->bm), &(handle->bm), &(handle->bn), &(handle->b_m1), &(handle->b_m1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
      handle->handlett = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->k, handle->n, &(handle->bm), &(handle->bk), &(handle->bn), &(handle->b_m1), &(handle->b_k1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
    }
    /* Need to allocate space for scratch and internalstate libxs_dnn_tensor's */
    handle->r1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->r2t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->z1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->z2t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->g1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->g2t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->g3  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->h1  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->h2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->h3  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->r   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->z   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->g   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwr = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwz = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwg = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdxt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdur = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djduz = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdug = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdht = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdrt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdzt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdgt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->barrier = libxs_barrier_create(handle->nThreads, 1);
    if (NULL == handle->r1t || NULL == handle->r2t || NULL == handle->z1t || NULL == handle->z2t || NULL == handle->g1t ||
        NULL == handle->g2t || NULL == handle->g3 || NULL == handle->h1 || NULL == handle->h2 || NULL == handle->h3 ||
        NULL == handle->r || NULL == handle->z || NULL == handle->g || NULL == handle->barrier ||
        NULL == handle->djdwr || NULL == handle->djdwz ||NULL == handle->djdwg || NULL == handle->djdxt ||
        NULL == handle->djdur || NULL == handle->djduz ||NULL == handle->djdug || NULL == handle->djdht ||
        NULL == handle->djdrt || NULL == handle->djdzt || NULL == handle->djdgt)
    {
      free(handle->r1t); free(handle->r2t); free(handle->z1t); free(handle->z2t); free(handle->g1t);
      free(handle->g2t); free(handle->g3); free(handle->h1); free(handle->h2); free(handle->h3);
      free(handle->r); free(handle->z); free(handle->g);
      free(handle->djdwr); free(handle->djdwz); free(handle->djdwg); free(handle->djdxt);
      free(handle->djdur); free(handle->djduz); free(handle->djdug); free(handle->djdht);
      free(handle->djdrt); free(handle->djdzt); free(handle->djdgt);
      *status = LIBXS_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXS_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_grucell(const libxs_dnn_grucell* handle)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  if (0 != handle) {
    free(handle->r1t); free(handle->r2t); free(handle->z1t); free(handle->z2t); free(handle->g1t);
    free(handle->g2t); free(handle->g3); free(handle->h1); free(handle->h2); free(handle->h3);
    free(handle->r); free(handle->z); free(handle->g);
    free(handle->djdwr); free(handle->djdwz); free(handle->djdwg); free(handle->djdxt);
    free(handle->djdur); free(handle->djduz); free(handle->djdug); free(handle->djdht);
    free(handle->djdrt); free(handle->djdzt); free(handle->djdgt);
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxs_barrier_release((const libxs_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxs_dnn_grucell*)handle);
  }
  return status;
}


LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_grucell_create_tensor_datalayout(const libxs_dnn_grucell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor_datalayout* layout = 0;
  *status = LIBXS_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxs_dnn_tensor_datalayout*) malloc(sizeof(libxs_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxs_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXS_DNN_GRU_REGULAR_INPUT)          || (type == LIBXS_DNN_GRU_GRADIENT_INPUT)  ||
           (type == LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE)   || (type == LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE) ||
           (type == LIBXS_DNN_GRU_REGULAR_WEIGHT_R)       || (type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_R) ||
           (type == LIBXS_DNN_GRU_REGULAR_WEIGHT_Z)       || (type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z) ||
           (type == LIBXS_DNN_GRU_REGULAR_WEIGHT_G)       || (type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_G) ||
           (type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R) || (type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) ||
           (type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) || (type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) ||
           (type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G) || (type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) ||
           (type == LIBXS_DNN_GRU_REGULAR_BIAS_R)         || (type == LIBXS_DNN_GRU_GRADIENT_BIAS_R)   ||
           (type == LIBXS_DNN_GRU_REGULAR_BIAS_Z)         || (type == LIBXS_DNN_GRU_GRADIENT_BIAS_Z)   ||
           (type == LIBXS_DNN_GRU_REGULAR_BIAS_G)         || (type == LIBXS_DNN_GRU_GRADIENT_BIAS_G) ) {
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
                if ( (type == LIBXS_DNN_GRU_REGULAR_INPUT) || (type == LIBXS_DNN_GRU_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bk;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE) || (type == LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXS_DNN_GRU_REGULAR_WEIGHT_R) || (type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_R) ||
                            (type == LIBXS_DNN_GRU_REGULAR_WEIGHT_Z) || (type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z) ||
                            (type == LIBXS_DNN_GRU_REGULAR_WEIGHT_G) || (type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_G) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R) || (type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) ||
                            (type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) || (type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) ||
                            (type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G) || (type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) ) {
                  layout->dim_type[0] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXS_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bm;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXS_DNN_GRU_REGULAR_BIAS_R) || (type == LIBXS_DNN_GRU_GRADIENT_BIAS_R) ||
                            (type == LIBXS_DNN_GRU_REGULAR_BIAS_Z) || (type == LIBXS_DNN_GRU_GRADIENT_BIAS_Z) ||
                            (type == LIBXS_DNN_GRU_REGULAR_BIAS_G) || (type == LIBXS_DNN_GRU_GRADIENT_BIAS_G) ) {
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


LIBXS_API size_t libxs_dnn_grucell_get_scratch_size(const libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r2t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z2t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g2t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* g3 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* h1 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* h2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* h3 */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
#if 0
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* i1b */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* i2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* i3 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* f1b */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* f2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* f3 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* o1b */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* o2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* c1b */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* c2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d3 (dh) */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d4 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* delta */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdiMt */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdfMt */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdcMt */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdoMt */
                                           size += 64;
#endif
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


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bind_scratch(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, const void* scratch)
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
                                             handle->r1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->r2t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r2t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g2t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g2t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g3->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h3->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
#if 0
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i3->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f3->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dh->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dh->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d4->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d4->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltat->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltat->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdiMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdiMt->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdfMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdfMt->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdoMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdoMt->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdcMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdcMt->data = (void*)(address+offset);
                                           }
#endif
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


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_release_scratch(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           handle->r1t->data = 0;
                                           handle->r2t->data = 0;
                                           handle->z1t->data = 0;
                                           handle->z2t->data = 0;
                                           handle->g1t->data = 0;
                                           handle->g2t->data = 0;
                                           handle->g3->data = 0;
                                           handle->h1->data = 0;
                                           handle->h2->data = 0;
                                           handle->h3->data = 0;
                                           handle->r1t = 0;
                                           handle->r2t = 0;
                                           handle->z1t = 0;
                                           handle->z2t = 0;
                                           handle->g1t = 0;
                                           handle->g2t = 0;
                                           handle->g3 = 0;
                                           handle->h1 = 0;
                                           handle->h2 = 0;
                                           handle->h3 = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->r1t->data = 0;
                                           handle->r2t->data = 0;
                                           handle->z1t->data = 0;
                                           handle->z2t->data = 0;
                                           handle->g1t->data = 0;
                                           handle->g2t->data = 0;
                                           handle->g3->data = 0;
                                           handle->h1->data = 0;
                                           handle->h2->data = 0;
                                           handle->h3->data = 0;
                                           handle->r1t = 0;
                                           handle->r2t = 0;
                                           handle->z1t = 0;
                                           handle->z2t = 0;
                                           handle->g1t = 0;
                                           handle->g2t = 0;
                                           handle->g3 = 0;
                                           handle->h1 = 0;
                                           handle->h2 = 0;
                                           handle->h3 = 0;
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


LIBXS_API size_t libxs_dnn_grucell_get_internalstate_size(const libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
#if 0
        size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* i */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* f */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* o */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* c */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* d */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djddt */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdit */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdft */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdct */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* djdot */
                                           size += 64;
#endif
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


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bind_internalstate(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, const void* internalstate)
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
                                             handle->r->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
#if 0
                                           if (address % 64 == 0) {
                                             handle->i->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djddt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djddt->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdit->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdit->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdft->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdft->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdot->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdot->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdct->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdct->data = (void*)(address+offset);
                                           }
#endif
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


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_release_internalstate(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           handle->r->data = 0;
                                           handle->z->data = 0;
                                           handle->g->data = 0;
                                           handle->r = 0;
                                           handle->z = 0;
                                           handle->g = 0;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           handle->r->data = 0;
                                           handle->z->data = 0;
                                           handle->g->data = 0;
                                           handle->r = 0;
                                           handle->z = 0;
                                           handle->g = 0;
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


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_assign_internalstate(libxs_dnn_grucell* handle, const void* rgoldtb, const void* zgoldtb, const void* ggoldtb)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && rgoldtb != 0 && zgoldtb != 0 && ggoldtb != 0) {
    const libxs_blasint m = handle->m, n = handle->n, t = handle->t;
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, rgold, (const LIBXS_DNN_ELTWISE_FTYPE*)rgoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, r, (LIBXS_DNN_ELTWISE_FTYPE*)handle->r->data, m * n);
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, zgold, (const LIBXS_DNN_ELTWISE_FTYPE*)zgoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z, (LIBXS_DNN_ELTWISE_FTYPE*)handle->z->data, m * n);
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, ggold, (const LIBXS_DNN_ELTWISE_FTYPE*)ggoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, g, (LIBXS_DNN_ELTWISE_FTYPE*)handle->g->data, m * n);
    libxs_blasint it;
    for (it = 0; it < t; ++it) {
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, rgold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, r, it, 0, m * n));
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, zgold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, z, it, 0, m * n));
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, ggold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, g, it, 0, m * n));
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bind_tensor(libxs_dnn_grucell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_GRU_REGULAR_INPUT)         && (type != LIBXS_DNN_GRU_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE)   && (type != LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_R)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_R) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_Z)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_G)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_G) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_R)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_R)   &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_Z)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_Z)   &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_G)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_G) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxs_dnn_tensor_datalayout* handle_layout = libxs_dnn_grucell_create_tensor_datalayout(handle, type, &status);

    if ( libxs_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXS_DNN_GRU_REGULAR_INPUT ) {
        handle->xt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_INPUT ) {
        handle->djdxt = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE ) {
        handle->h = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE ) {
        handle->djdht = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_R ) {
        handle->ur = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_R ) {
        handle->djdur = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_Z ) {
        handle->uz = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z ) {
        handle->djduz = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_G ) {
        handle->ug = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_G ) {
        handle->djdug = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) {
        handle->wr = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) {
        handle->djdwr = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) {
        handle->wz = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) {
        handle->djdwz = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) {
        handle->wg = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) {
        handle->djdwg = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_R ) {
        handle->br = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_R ) {
        handle->djdbr = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_Z ) {
        handle->bz = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_Z ) {
        handle->djdbz = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_G ) {
        handle->bg = (libxs_dnn_tensor*)tensor;
      } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_G ) {
        handle->djdbg = (libxs_dnn_tensor*)tensor;
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


LIBXS_API libxs_dnn_tensor* libxs_dnn_grucell_get_tensor(libxs_dnn_grucell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status)
{
  libxs_dnn_tensor* tensor = 0;
  LIBXS_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXS_DNN_GRU_REGULAR_INPUT)         && (type != LIBXS_DNN_GRU_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE)   && (type != LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_R)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_R) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_Z)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_G)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_G) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_R)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_R)   &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_Z)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_Z)   &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_G)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_G) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_GRU_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_INPUT ) {
      tensor = handle->djdxt;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE ) {
      tensor = handle->h;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->djdht;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_R ) {
      tensor = handle->ur;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_R ) {
      tensor = handle->djdur;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_Z ) {
      tensor = handle->uz;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z ) {
      tensor = handle->djduz;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_G ) {
      tensor = handle->ug;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_G ) {
      tensor = handle->djdug;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) {
      tensor = handle->wr;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) {
      tensor = handle->djdwr;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) {
      tensor = handle->wz;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) {
      tensor = handle->djdwz;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) {
      tensor = handle->wg;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) {
      tensor = handle->djdwg;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_R ) {
      tensor = handle->br;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_R ) {
      tensor = handle->djdbr;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_Z ) {
      tensor = handle->bz;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_Z ) {
      tensor = handle->djdbz;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_G ) {
      tensor = handle->bg;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_G ) {
      tensor = handle->djdbg;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_release_tensor(libxs_dnn_grucell* handle, const libxs_dnn_tensor_type type)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXS_DNN_GRU_REGULAR_INPUT)         && (type != LIBXS_DNN_GRU_GRADIENT_INPUT)  &&
      (type != LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE)   && (type != LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_R)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_R) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_Z)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z) &&
      (type != LIBXS_DNN_GRU_REGULAR_WEIGHT_G)       && (type != LIBXS_DNN_GRU_GRADIENT_WEIGHT_G) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) &&
      (type != LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G) && (type != LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_R)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_R)   &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_Z)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_Z)   &&
      (type != LIBXS_DNN_GRU_REGULAR_BIAS_G)         && (type != LIBXS_DNN_GRU_GRADIENT_BIAS_G) ) {
    status = LIBXS_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXS_DNN_GRU_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_INPUT ) {
      handle->djdxt = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_HIDDEN_STATE ) {
      handle->h = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_HIDDEN_STATE ) {
      handle->djdht = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_R ) {
      handle->ur = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_R ) {
      handle->djdur = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_Z ) {
      handle->uz = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_Z ) {
      handle->djduz = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_WEIGHT_G ) {
      handle->ug = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_WEIGHT_G ) {
      handle->djdug = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) {
      handle->wr = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) {
      handle->djdwr = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) {
      handle->wz = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) {
      handle->djdwz = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) {
      handle->wg = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) {
      handle->djdwg = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_R ) {
      handle->br = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_R ) {
      handle->djdbr = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_Z ) {
      handle->bz = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_Z ) {
      handle->djdbz = 0;
    } else if ( type == LIBXS_DNN_GRU_REGULAR_BIAS_G ) {
      handle->bg = 0;
    } else if ( type == LIBXS_DNN_GRU_GRADIENT_BIAS_G ) {
      handle->djdbg = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_fwd(libxs_dnn_grucell* gru, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = gru->m;
  libxs_blasint n = gru->n;
  libxs_blasint k = gru->k;
  libxs_blasint t = gru->t;
#if defined(LSTM_TIMING)
  libxs_blasint k = gru->k;
  const double tflops = 12;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n) + (tflops * m * n)) * 2.0; /* r and z */
  gflops += (m * n) + (2.0 * m * n * k) + (2.0 * m * n * m) + (tflops * m * n); /* g */
  gflops += 4.0 * (m * n); /* h */
  gflops *= (double)t * 1E-9; /* t time steps */
#endif
  int reuse = gru->reuse;
  LIBXS_DNN_ELTWISE_FTYPE *wr  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wr->data;
  LIBXS_DNN_ELTWISE_FTYPE *wz  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wz->data;
  LIBXS_DNN_ELTWISE_FTYPE *wg  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wg->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ur  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->ur->data;
  LIBXS_DNN_ELTWISE_FTYPE *uz  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->uz->data;
  LIBXS_DNN_ELTWISE_FTYPE *ug  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->ug->data;
  LIBXS_DNN_ELTWISE_FTYPE *h   = (LIBXS_DNN_ELTWISE_FTYPE*)gru->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *br  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->br->data;
  LIBXS_DNN_ELTWISE_FTYPE *bz  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->bz->data;
  LIBXS_DNN_ELTWISE_FTYPE *bg  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->bg->data;
  LIBXS_DNN_ELTWISE_FTYPE *r1t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->r1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *r2t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->r2t->data;
  LIBXS_DNN_ELTWISE_FTYPE *z1t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->z1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *z2t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->z2t->data;
  LIBXS_DNN_ELTWISE_FTYPE *g1t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->g1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *g2t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->g2t->data;
  LIBXS_DNN_ELTWISE_FTYPE *g3  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->g3->data;
  LIBXS_DNN_ELTWISE_FTYPE *h1  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->h1->data;
  LIBXS_DNN_ELTWISE_FTYPE *h2  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->h2->data;
  LIBXS_DNN_ELTWISE_FTYPE *h3  = (LIBXS_DNN_ELTWISE_FTYPE*)gru->h3->data;
  LIBXS_DNN_ELTWISE_FTYPE *r   = (LIBXS_DNN_ELTWISE_FTYPE*)gru->r->data;
  LIBXS_DNN_ELTWISE_FTYPE *z   = (LIBXS_DNN_ELTWISE_FTYPE*)gru->z->data;
  LIBXS_DNN_ELTWISE_FTYPE *g   = (LIBXS_DNN_ELTWISE_FTYPE*)gru->g->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, r1, r1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z1, z1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, g1, g1t, m * n);
  /*libxs_bgemm_handle *handleux = gru->handleux;*/
  libxs_bgemm_handle *handlewh = gru->handlewh;
  libxs_bgemm_handle *handlett = gru->handlett;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, rnr, r, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, znr, z, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, gnr, g, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, r2, r2t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z2, z2t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, g2, g2t, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  int j;
  const int ltid = tid - start_thread;

  libxs_barrier_init(gru->barrier, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) { start = libxs_timer_tick(); }
#endif

  if (reuse) {
#if defined(LSTM_TIMING)
    if (ltid == 0) { Gbl_t_input = libxs_timer_tick(); }
#endif
    libxs_bgemm_st(handlett, ur, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, r1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, uz, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, ug, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, g1, 0, 0, m * n), start_thread, tid);
#if defined(LSTM_TIMING)
    if (ltid == 0) {
      Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
    }
#endif
    for (j = 0; j < t; ++j) {
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, r1, j, 0, m * n), br, &LIBXS_VLA_ACCESS(2, r1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, z1, j, 0, m * n), bz, &LIBXS_VLA_ACCESS(2, z1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, g1, j, 0, m * n), bg, &LIBXS_VLA_ACCESS(2, g1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxs_internal_recursive_step(handlewh, wr, h, &LIBXS_VLA_ACCESS(2, r2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, r1, j, 0, m * n), r, r, 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handlewh, wz, h, &LIBXS_VLA_ACCESS(2, z2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, z1, j, 0, m * n), z, z, 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, h, r, g3, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxs_internal_recursive_step(handlewh, wg, g3, &LIBXS_VLA_ACCESS(2, g2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, g1, j, 0, m * n), g, g, 3, m * n, start_thread, tid); /*tanh*/
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, z, g, h3, start_thread, tid, gru->nThreads);
      libxs_internal_matrix_complement(m*n, z, h2, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_internal_matrix_eltwise_mult(m*n, h, h2, h1, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_internal_matrix_add(m*n, h1, h3, h, start_thread, tid, gru->nThreads);
#if defined(LSTM_TIMING)
      libxs_barrier_wait(gru->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
    }
  } else {
#if defined(LSTM_TIMING)
    if (ltid == 0) { Gbl_t_input = libxs_timer_tick(); }
#endif
    libxs_bgemm_st(handlett, ur, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, r1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, uz, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, ug, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, g1, 0, 0, m * n), start_thread, tid);
#if defined(LSTM_TIMING)
    if (ltid == 0) {
      Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
    }
#endif
    for (j = 0; j < t; ++j) {
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, r1, j, 0, m * n), br, &LIBXS_VLA_ACCESS(2, r1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, z1, j, 0, m * n), bz, &LIBXS_VLA_ACCESS(2, z1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, g1, j, 0, m * n), bg, &LIBXS_VLA_ACCESS(2, g1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxs_internal_recursive_step(handlewh, wr, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, r2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, r1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, rnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, rnr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handlewh, wz, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, z2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, z1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, znr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, znr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, rnr, j, 0, m * n), g3, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxs_internal_recursive_step(handlewh, wg, g3, &LIBXS_VLA_ACCESS(2, g2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, g1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, gnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, gnr, j, 0, m * n), 3, m * n, start_thread, tid); /*tanh*/
      libxs_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, znr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, gnr, j, 0, m * n), h3, start_thread, tid, gru->nThreads);
      libxs_internal_matrix_complement(m*n, &LIBXS_VLA_ACCESS(2, znr, j, 0, m * n), h2, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m *n), h2, h1, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_internal_matrix_add(m*n, h1, h3, &LIBXS_VLA_ACCESS(2, hnr, j+1, 0, m * n), start_thread, tid, gru->nThreads);
#if defined(LSTM_TIMING)
      libxs_barrier_wait(gru->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
    }
  }
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    duration = libxs_timer_duration(start, libxs_timer_tick());
    if (0 < duration) {
      fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops / duration);
      double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
      fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
      fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
      fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
      fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
    }
  }
#endif

  return status;
}

#if 0
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bwd_upd_bu(libxs_dnn_grucell* gru, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = gru->m;
  libxs_blasint n = gru->n;
  libxs_blasint k = gru->k;
  libxs_blasint t = gru->t;
  LIBXS_DNN_ELTWISE_FTYPE *wi = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wi->data;
  LIBXS_DNN_ELTWISE_FTYPE *wf = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wf->data;
  LIBXS_DNN_ELTWISE_FTYPE *wo = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wo->data;
  LIBXS_DNN_ELTWISE_FTYPE *wc = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wc->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ri = (LIBXS_DNN_ELTWISE_FTYPE*)gru->ri->data;
  LIBXS_DNN_ELTWISE_FTYPE *rf = (LIBXS_DNN_ELTWISE_FTYPE*)gru->rf->data;
  LIBXS_DNN_ELTWISE_FTYPE *ro = (LIBXS_DNN_ELTWISE_FTYPE*)gru->ro->data;
  LIBXS_DNN_ELTWISE_FTYPE *rc = (LIBXS_DNN_ELTWISE_FTYPE*)gru->rc->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)gru->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *i1 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->i1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *i2 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->i2->data;
  LIBXS_DNN_ELTWISE_FTYPE *i3 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->i3->data;
  LIBXS_DNN_ELTWISE_FTYPE *f1 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->f1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *f2 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->f2->data;
  LIBXS_DNN_ELTWISE_FTYPE *f3 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->f3->data;
  LIBXS_DNN_ELTWISE_FTYPE *o1 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->o1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *o2 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->o2->data;
  LIBXS_DNN_ELTWISE_FTYPE *c1 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->c1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *c2 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->c2->data;
  LIBXS_DNN_ELTWISE_FTYPE *it = (LIBXS_DNN_ELTWISE_FTYPE*)gru->i->data;
  LIBXS_DNN_ELTWISE_FTYPE *ft = (LIBXS_DNN_ELTWISE_FTYPE*)gru->f->data;
  LIBXS_DNN_ELTWISE_FTYPE *ot = (LIBXS_DNN_ELTWISE_FTYPE*)gru->o->data;
  LIBXS_DNN_ELTWISE_FTYPE *ct = (LIBXS_DNN_ELTWISE_FTYPE*)gru->c->data;
  LIBXS_DNN_ELTWISE_FTYPE *d1 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d1->data;
  LIBXS_DNN_ELTWISE_FTYPE *d2 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d2->data;
  LIBXS_DNN_ELTWISE_FTYPE *d3 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->dh->data;
  LIBXS_DNN_ELTWISE_FTYPE *d4 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d4->data;
  LIBXS_DNN_ELTWISE_FTYPE *dt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdht = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdht->data;
  LIBXS_DNN_ELTWISE_FTYPE *deltat = (LIBXS_DNN_ELTWISE_FTYPE*)gru->deltat->data;
  LIBXS_DNN_ELTWISE_FTYPE *djddt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djddt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdit = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdit->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdft = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdft->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdct = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdct->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdot = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdot->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdxt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdxt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwi = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwi->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwf = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwf->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwo = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwo->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwc = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwc->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdri = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdri->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdrf = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdrf->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdro = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdro->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdrc = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdrc->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbi = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbi->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbf = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbf->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbo = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbo->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbc = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbc->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdiMt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdiMt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdfMt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdfMt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdcMt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdcMt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdoMt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdoMt->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, h, ht, m * n);
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
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdiM, djdiMt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdfM, djdfMt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdoM, djdoMt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdcM, djdcMt, m * n);
  libxs_bgemm_handle *handleud = gru->handlewx;
  libxs_bgemm_handle *handledh = gru->handleuh;
  libxs_bgemm_handle *handledx = gru->handlett;
  libxs_bgemm_handle *handlewd = gru->handlewd;
  int j;
  const int ltid = tid - start_thread;

  libxs_barrier_init(gru->barrier, ltid);
  /* compute delta */
  libxs_internal_matrix_copy(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdd */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), d1, start_thread, tid, gru->nThreads);
  libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), d2, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdc */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), c1, start_thread, tid, gru->nThreads);
  libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), c2, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdi */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), i1, start_thread, tid, gru->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdf */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, d, t-2, 0, m * n), f1, start_thread, tid, gru->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdo */
  libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), o1, start_thread, tid, gru->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1, start_thread, tid, gru->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2, start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  libxs_barrier_wait(gru->barrier, ltid);
  if (pass == 1 || pass == 3) {
    /* compute djdx */
    libxs_bgemm_st(handlewd, wi, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
    libxs_bgemm_st(handlewd, wf, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
    libxs_bgemm_st(handlewd, wo, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
    libxs_bgemm_st(handlewd, wc, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
  }
  for (j = t-2; j >= 0; --j) {
    /* compute delta */
    libxs_bgemm_st(handleud, ri, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handleud, rf, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handleud, ro, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handleud, rc, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdd */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), d1, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), d2, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, d1, d2, d3, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, f, j+1, 0, m * n), d4, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_add(m * n, d3, d4, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdc */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), c1, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), c2, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdi */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), i1, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, i3, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdf */
    if (j >= 1) {
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, d, j-1, 0, m * n), f1, start_thread, tid, gru->nThreads);
      libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, f3, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, gru->nThreads);
    } else {
      /* djdf is zero for j == 0 */
      libxs_internal_matrix_zero(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, gru->nThreads);
    }
    /* compute djdo */
    libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), o1, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, o2, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      libxs_bgemm_st(handlewd, wi, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxs_bgemm_st(handlewd, wf, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxs_bgemm_st(handlewd, wo, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxs_bgemm_st(handlewd, wc, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
    }
  }
  if (pass == 2 || pass == 3) {
    /* Reorganizing djdi, djdf, dfdo, djdc */
    for (j = 0; j < t; ++j) {
      libxs_bgemm_convert_b_to_a(handleud, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, djdiM, j, 0, m * n));
      libxs_bgemm_convert_b_to_a(handleud, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, djdfM, j, 0, m * n));
      libxs_bgemm_convert_b_to_a(handleud, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, djdoM, j, 0, m * n));
      libxs_bgemm_convert_b_to_a(handleud, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, djdcM, j, 0, m * n));
      libxs_barrier_wait(gru->barrier, ltid);
    }
    /* compute djdw */
    for (j = 0; j < t; ++j) {
      libxs_bgemm_st(handledx, &LIBXS_VLA_ACCESS(2, djdiM, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwi, start_thread, tid);
      libxs_bgemm_st(handledx, &LIBXS_VLA_ACCESS(2, djdfM, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwf, start_thread, tid);
      libxs_bgemm_st(handledx, &LIBXS_VLA_ACCESS(2, djdoM, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwo, start_thread, tid);
      libxs_bgemm_st(handledx, &LIBXS_VLA_ACCESS(2, djdcM, j, 0, m * n), &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdwc, start_thread, tid);
    }
    /* compute djdr */
    for (j = 0; j < t-1; ++j) {
      libxs_bgemm_st(handledh, &LIBXS_VLA_ACCESS(2, djdiM, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), djdri, start_thread, tid);
      libxs_bgemm_st(handledh, &LIBXS_VLA_ACCESS(2, djdfM, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), djdrf, start_thread, tid);
      libxs_bgemm_st(handledh, &LIBXS_VLA_ACCESS(2, djdoM, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), djdro, start_thread, tid);
      libxs_bgemm_st(handledh, &LIBXS_VLA_ACCESS(2, djdcM, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), djdrc, start_thread, tid);
    }
    /* compute djdb */
    for (j = 0; j < t-1; j++) {
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi, start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf, start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo, start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc, start_thread, tid, gru->nThreads);
    }
    libxs_barrier_wait(gru->barrier, ltid);
  }

  return status;
}
#endif /* if 0 */
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_execute_st(libxs_dnn_grucell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           status = libxs_dnn_grucell_fwd(handle, start_thread, tid);
                                         } break;
#if 0
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                           status = libxs_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                           status = libxs_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           status = libxs_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 3);
                                         } break;
#endif
      default: {
                  status = LIBXS_DNN_ERR_INVALID_KIND;
               }
    }
  } else {
    status = LIBXS_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

