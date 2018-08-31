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
      handle->handlewd = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */
      handle->handlewh = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->m, handle->n, &(handle->bm), &(handle->bm), &(handle->bn), &(handle->b_m1), &(handle->b_m1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
      handle->handlett = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->k, handle->n, &(handle->bm), &(handle->bk), &(handle->bn), &(handle->b_m1), &(handle->b_k1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
      handle->handleux = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->k, handle->n, handle->m, &(handle->bk), &(handle->bn), &(handle->bm), &(handle->b_k1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
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
    handle->d3  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d4  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d5  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d6  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d7  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d8  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d9  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d10 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d11 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d12 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d13 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d14 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d15 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d16 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d17 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d18 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d19 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d20 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d21 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d22 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d23 = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d10M  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d11M  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d18M  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->hrTp  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwr = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwz = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwg = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdxt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdur = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djduz = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdug = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdht = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbr = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbz = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbg = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->barrier = libxs_barrier_create(handle->nThreads, 1);
    if (NULL == handle->r1t || NULL == handle->r2t || NULL == handle->z1t || NULL == handle->z2t || NULL == handle->g1t ||
        NULL == handle->g2t || NULL == handle->g3 || NULL == handle->h1 || NULL == handle->h2 || NULL == handle->h3 ||
        NULL == handle->r || NULL == handle->z || NULL == handle->g || NULL == handle->barrier ||
        NULL == handle->djdwr || NULL == handle->djdwz ||NULL == handle->djdwg || NULL == handle->djdxt ||
        NULL == handle->djdur || NULL == handle->djduz ||NULL == handle->djdug || NULL == handle->djdht ||
        NULL == handle->djdbr || NULL == handle->djdbz ||NULL == handle->djdbg || NULL == handle->hrTp || NULL == handle->d3 ||
        NULL == handle->d4 || NULL == handle->d5 || NULL == handle->d6 || NULL == handle->d7 || NULL == handle->d8 ||
        NULL == handle->d9 || NULL == handle->d10 || NULL == handle->d11 || NULL == handle->d12 || NULL == handle->d13 ||
        NULL == handle->d14 || NULL == handle->d15 || NULL == handle->d16 || NULL == handle->d17 || NULL == handle->d18 ||
        NULL == handle->d19 || NULL == handle->d20 || NULL == handle->d21 || NULL == handle->d22 || NULL == handle->d23 ||
        NULL == handle->d10M || NULL == handle->d11M || NULL == handle->d18M)
    {
      free(handle->r1t); free(handle->r2t); free(handle->z1t); free(handle->z2t); free(handle->g1t);
      free(handle->g2t); free(handle->g3); free(handle->h1); free(handle->h2); free(handle->h3);
      free(handle->r); free(handle->z); free(handle->g);
      free(handle->djdwr); free(handle->djdwz); free(handle->djdwg); free(handle->djdxt);
      free(handle->djdur); free(handle->djduz); free(handle->djdug); free(handle->djdht);
      free(handle->djdbr); free(handle->djdbz); free(handle->djdbg); free(handle->hrTp); free(handle->d3);
      free(handle->d4); free(handle->d5); free(handle->d6); free(handle->d7); free(handle->d8); free(handle->d9);
      free(handle->d10); free(handle->d11); free(handle->d12); free(handle->d13); free(handle->d14); free(handle->d15);
      free(handle->d16); free(handle->d17); free(handle->d18); free(handle->d19); free(handle->d20); free(handle->d21);
      free(handle->d22); free(handle->d23); free(handle->d10M); free(handle->d11M); free(handle->d18M);
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
    free(handle->djdbr); free(handle->djdbz); free(handle->djdbg); free(handle->hrTp); free(handle->d3);
    free(handle->d4); free(handle->d5); free(handle->d6); free(handle->d7); free(handle->d8); free(handle->d9);
    free(handle->d10); free(handle->d11); free(handle->d12); free(handle->d13); free(handle->d14); free(handle->d15);
    free(handle->d16); free(handle->d17); free(handle->d18); free(handle->d19); free(handle->d20); free(handle->d21);
    free(handle->d22); free(handle->d23); free(handle->d10M); free(handle->d11M); free(handle->d18M);
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
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d3 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d4 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d5 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d6 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d7 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d8 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d9 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d10 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d11 */
                                           size += 64;
                                           size += (size_t)handle->k * handle->n * sizeof_datatype * handle->t; /* d12 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* d13 */
                                           size += 64;
                                           size += (size_t)handle->k * handle->n * sizeof_datatype * handle->t; /* d14 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* d15 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d16 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d17 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d18 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d19 */
                                           size += 64;
                                           size += (size_t)handle->k * handle->n * sizeof_datatype * handle->t; /* d20 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* d21 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d22 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d23 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d10M */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d11M */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d18M */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* hrTp */
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
                                           if (address % 64 == 0) {
                                             handle->d3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d3->data = (void*)(address+offset);
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
                                             handle->d5->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d5->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d6->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d6->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d7->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d7->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d8->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d8->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d9->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d9->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d10->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d10->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d11->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d11->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d12->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d12->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->k * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d13->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d13->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d14->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d14->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->k * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d15->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d15->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d16->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d16->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d17->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d17->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d18->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d18->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d19->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d19->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d20->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d20->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->k * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d21->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d21->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d22->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d22->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d23->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d23->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d10M->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d10M->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d11M->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d11M->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d18M->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d18M->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->hrTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->hrTp->data = (void*)(address+offset);
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
                                           handle->d3->data = 0;
                                           handle->d4->data = 0;
                                           handle->d5->data = 0;
                                           handle->d6->data = 0;
                                           handle->d7->data = 0;
                                           handle->d8->data = 0;
                                           handle->d9->data = 0;
                                           handle->d10->data = 0;
                                           handle->d11->data = 0;
                                           handle->d12->data = 0;
                                           handle->d13->data = 0;
                                           handle->d14->data = 0;
                                           handle->d15->data = 0;
                                           handle->d16->data = 0;
                                           handle->d17->data = 0;
                                           handle->d18->data = 0;
                                           handle->d19->data = 0;
                                           handle->d20->data = 0;
                                           handle->d21->data = 0;
                                           handle->d22->data = 0;
                                           handle->d23->data = 0;
                                           handle->d10M->data = 0;
                                           handle->d11M->data = 0;
                                           handle->d18M->data = 0;
                                           handle->hrTp->data = 0;
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
                                           handle->d3 = 0;
                                           handle->d4 = 0;
                                           handle->d5 = 0;
                                           handle->d6 = 0;
                                           handle->d7 = 0;
                                           handle->d8 = 0;
                                           handle->d9 = 0;
                                           handle->d10 = 0;
                                           handle->d11 = 0;
                                           handle->d12 = 0;
                                           handle->d13 = 0;
                                           handle->d14 = 0;
                                           handle->d15 = 0;
                                           handle->d16 = 0;
                                           handle->d17 = 0;
                                           handle->d18 = 0;
                                           handle->d19 = 0;
                                           handle->d20 = 0;
                                           handle->d21 = 0;
                                           handle->d22 = 0;
                                           handle->d23 = 0;
                                           handle->d10M = 0;
                                           handle->d11M = 0;
                                           handle->d18M = 0;
                                           handle->hrTp = 0;
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
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g */
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


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bwd_upd_bu(libxs_dnn_grucell* gru, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = gru->m;
  libxs_blasint n = gru->n;
  libxs_blasint k = gru->k;
  libxs_blasint t = gru->t;
  LIBXS_DNN_ELTWISE_FTYPE *wr = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wr->data;
  LIBXS_DNN_ELTWISE_FTYPE *wz = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wz->data;
  LIBXS_DNN_ELTWISE_FTYPE *wg = (LIBXS_DNN_ELTWISE_FTYPE*)gru->wg->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ur = (LIBXS_DNN_ELTWISE_FTYPE*)gru->ur->data;
  LIBXS_DNN_ELTWISE_FTYPE *uz = (LIBXS_DNN_ELTWISE_FTYPE*)gru->uz->data;
  LIBXS_DNN_ELTWISE_FTYPE *ug = (LIBXS_DNN_ELTWISE_FTYPE*)gru->ug->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)gru->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *rt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->r->data;
  LIBXS_DNN_ELTWISE_FTYPE *zt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->z->data;
  LIBXS_DNN_ELTWISE_FTYPE *gt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->g->data;
  LIBXS_DNN_ELTWISE_FTYPE *d3 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d3->data;
  LIBXS_DNN_ELTWISE_FTYPE *d4 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d4->data;
  LIBXS_DNN_ELTWISE_FTYPE *d5 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d5->data;
  LIBXS_DNN_ELTWISE_FTYPE *d6 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d6->data;
  LIBXS_DNN_ELTWISE_FTYPE *d7 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d7->data;
  LIBXS_DNN_ELTWISE_FTYPE *d8 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d8->data;
  LIBXS_DNN_ELTWISE_FTYPE *d9 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d9->data;
  LIBXS_DNN_ELTWISE_FTYPE *d10 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d10->data;
  LIBXS_DNN_ELTWISE_FTYPE *d11 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d11->data;
  LIBXS_DNN_ELTWISE_FTYPE *d12t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d12->data;
  LIBXS_DNN_ELTWISE_FTYPE *d13t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d13->data;
  LIBXS_DNN_ELTWISE_FTYPE *d14t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d14->data;
  LIBXS_DNN_ELTWISE_FTYPE *d15t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d15->data;
  LIBXS_DNN_ELTWISE_FTYPE *d16 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d16->data;
  LIBXS_DNN_ELTWISE_FTYPE *d17 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d17->data;
  LIBXS_DNN_ELTWISE_FTYPE *d18 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d18->data;
  LIBXS_DNN_ELTWISE_FTYPE *d19 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d19->data;
  LIBXS_DNN_ELTWISE_FTYPE *d20t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d20->data;
  LIBXS_DNN_ELTWISE_FTYPE *d21t = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d21->data;
  LIBXS_DNN_ELTWISE_FTYPE *d22 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d22->data;
  LIBXS_DNN_ELTWISE_FTYPE *d23 = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d23->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdht = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdht->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdxt = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdxt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwr = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwr->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwz = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwz->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdwg = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdwg->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdur = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdur->data;
  LIBXS_DNN_ELTWISE_FTYPE *djduz = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djduz->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdug = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdug->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbr = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbr->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbz = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbz->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdbg = (LIBXS_DNN_ELTWISE_FTYPE*)gru->djdbg->data;
  LIBXS_DNN_ELTWISE_FTYPE *d10M = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d10M->data;
  LIBXS_DNN_ELTWISE_FTYPE *d11M = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d11M->data;
  LIBXS_DNN_ELTWISE_FTYPE *d18M = (LIBXS_DNN_ELTWISE_FTYPE*)gru->d18M->data;
  LIBXS_DNN_ELTWISE_FTYPE *hrTp = (LIBXS_DNN_ELTWISE_FTYPE*)gru->hrTp->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, h, ht, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, r, rt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, z, zt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, g, gt, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdh, djdht, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, djdx, djdxt, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d12, d12t, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d14, d14t, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d20, d20t, k * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d13, d13t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d15, d15t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d21, d21t, m * n);
  libxs_bgemm_handle *handleud = gru->handleux;
  libxs_bgemm_handle *handledh = gru->handlewh;
  libxs_bgemm_handle *handledx = gru->handlett;
  libxs_bgemm_handle *handlewd = gru->handlewd;
  int j;
  const int ltid = tid - start_thread;

  libxs_barrier_init(gru->barrier, ltid);
  /* libxs_internal_matrix_zero(m * n, d23, start_thread, tid, gru->nThreads); */
  for (j = t-1; j >= 0; j--) {
    /* d3 = djdh + d23 (delta) */
    libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), d23, d3, start_thread, tid, gru->nThreads);
    /* d4 = (1 - z).d3 */
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, z, j, 0, m * n), d4, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, d4, d3, d4, start_thread, tid, gru->nThreads);
    /* d5 = d3.h */
    libxs_internal_matrix_eltwise_mult(m * n, d3, &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), d5, start_thread, tid, gru->nThreads);
    /* d6 = -d5 */
    libxs_internal_matrix_inverse(m * n, d5, d6, start_thread, tid, gru->nThreads);
    /* d7 = d3.g */
    libxs_internal_matrix_eltwise_mult(m * n, d3, &LIBXS_VLA_ACCESS(2, g, j, 0, m * n), d7, start_thread, tid, gru->nThreads);
    /* d8 = d3.z */
    libxs_internal_matrix_eltwise_mult(m * n, d3, &LIBXS_VLA_ACCESS(2, z, j, 0, m * n), d8, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    /* d9 = d7 + d8 */
    libxs_internal_matrix_add(m * n, d7, d8, d9, start_thread, tid, gru->nThreads);
    /* d10 = d8.tanh'(g) */
    libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, g, j, 0, m * n), d10, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, d8, d10, d10, start_thread, tid, gru->nThreads);
    /* d11 = d9.sig'(z) */
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, z, j, 0, m * n), d11, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, z, j, 0, m * n), d11, d11, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, d9, d11, d11, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    /* d13 = Wg^T * d10 */
    libxs_bgemm_st(handlewd, wg, d10, &LIBXS_VLA_ACCESS(2, d13, j, 0, m * n), start_thread, tid);
    /* d15 = Wz^T * d11 */
    libxs_bgemm_st(handlewd, wz, d11, &LIBXS_VLA_ACCESS(2, d15, j, 0, m * n), start_thread, tid);
    /* d16 = d13.h */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, d13, j, 0, m * n), &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), d16, start_thread, tid, gru->nThreads);
    /* d17 = d13.r */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, d13, j, 0, m * n), &LIBXS_VLA_ACCESS(2, r, j, 0, m * n), d17, start_thread, tid, gru->nThreads);
    /* d18 = d16.sig'(r) */
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, r, j, 0, m * n), d18, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, r, j, 0, m * n), d18, d18, start_thread, tid, gru->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, d16, d18, d18, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid);
    /* d19 = d17 + d4 */
    libxs_internal_matrix_add(m * n, d17, d4, d19, start_thread, tid, gru->nThreads);
    /* d21 = Wr^T * d18 */
    libxs_bgemm_st(handlewd, wr, d18, &LIBXS_VLA_ACCESS(2, d21, j, 0, m * n), start_thread, tid);
    /* d22 = d21 + d15 */
    libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, d21, j, 0, m * n), &LIBXS_VLA_ACCESS(2, d15, j, 0, m * n), d22, start_thread, tid, gru->nThreads);
    libxs_barrier_wait(gru->barrier, ltid); /* This barrier may be removed */
    /* d23 = d19 + d22 */
    libxs_internal_matrix_add(m * n, d19, d22, d23, start_thread, tid, gru->nThreads);
    if (1 == pass || 3 == pass) {
      /* d12 = Ug^T * d10 */
      libxs_bgemm_st(handleud, ug, d10, &LIBXS_VLA_ACCESS(2, d12, j, 0, k * n), start_thread, tid);
      /* d14 = Uz^T * d11 */
      libxs_bgemm_st(handleud, uz, d11, &LIBXS_VLA_ACCESS(2, d14, j, 0, k * n), start_thread, tid);
      /* d20 = Ur^T * d18 */
      libxs_bgemm_st(handleud, ur, d18, &LIBXS_VLA_ACCESS(2, d20, j, 0, k * n), start_thread, tid);
      /* djdx = d12 + d14 + d20 */
      libxs_internal_matrix_add(k * n, &LIBXS_VLA_ACCESS(2, d12, j, 0, k * n), &LIBXS_VLA_ACCESS(2, d14, j, 0, k * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid, gru->nThreads);
      libxs_internal_matrix_add(k * n, &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), &LIBXS_VLA_ACCESS(2, d20, j, 0, k * n), &LIBXS_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid, gru->nThreads);
    }
    if (2 == pass || 3 == pass) {
      /* Reorganize d10, d11, d18 */
      libxs_bgemm_convert_b_to_a(handlewd, d10, &m, d10M);
      libxs_bgemm_convert_b_to_a(handlewd, d11, &m, d11M);
      libxs_bgemm_convert_b_to_a(handlewd, d18, &m, d18M);
      /* djdwr = djdwr + d18 * h^T */
      libxs_bgemm_transpose_b(handledh, &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), &m, hrTp);
      libxs_bgemm_st(handledh, d18M, hrTp, djdwr, start_thread, tid);
      /* djdwz = djdwz + d11 * h^T */
      libxs_bgemm_st(handledh, d11M, hrTp, djdwz, start_thread, tid);
      /* djdwg = djdwg + d10 * (h.r)^T */
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, h, j, 0, m * n), &LIBXS_VLA_ACCESS(2, r, j, 0, m * n), d4, start_thread, tid, gru->nThreads);
      libxs_barrier_wait(gru->barrier, ltid);
      libxs_bgemm_transpose_b(handledh, d4, &m, hrTp);
      libxs_bgemm_st(handledh, d10M, hrTp, djdwg, start_thread, tid);
      /* djdur = djdur + d18 * x^T */
      libxs_bgemm_st(handledx, d18M, &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdur, start_thread, tid);
      /* djduz = djduz + d11 * x^T */
      libxs_bgemm_st(handledx, d11M, &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djduz, start_thread, tid);
      /* djdug = djdug + d10 * x^T */
      libxs_bgemm_st(handledx, d10M, &LIBXS_VLA_ACCESS(2, x, j, 0, k * n), djdug, start_thread, tid);
      /* djdbr = djdbr + d18 */
      libxs_internal_matrix_add(m * n, djdbr, d18, djdbr, start_thread, tid, gru->nThreads);
      /* djdbz = djdbz + d11 */
      libxs_internal_matrix_add(m * n, djdbz, d11, djdbz, start_thread, tid, gru->nThreads);
      /* djdbg = djdbg + d10 */
      libxs_internal_matrix_add(m * n, djdbg, d10, djdbg, start_thread, tid, gru->nThreads);
    }
    libxs_barrier_wait(gru->barrier, ltid);
  }

  return status;
}


LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_execute_st(libxs_dnn_grucell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXS_DNN_COMPUTE_KIND_FWD: {
                                           status = libxs_dnn_grucell_fwd(handle, start_thread, tid);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD: {
                                           status = libxs_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_UPD: {
                                           status = libxs_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
                                           status = libxs_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 3);
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

