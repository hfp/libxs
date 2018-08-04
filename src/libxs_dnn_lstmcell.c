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

/* #define NON_FUSED_INPUT_GEMM */
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
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif


LIBXS_API libxs_dnn_lstmcell* libxs_dnn_create_lstmcell(libxs_dnn_lstmcell_desc lstmcell_desc, libxs_dnn_err_t* status)
{
  libxs_dnn_lstmcell* handle = 0;
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

  handle = (libxs_dnn_lstmcell*)malloc(sizeof(libxs_dnn_lstmcell));
  if (0 != handle) {
    *status = LIBXS_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->nThreads = lstmcell_desc.nThreads;
    handle->desc = lstmcell_desc;
    handle->datatype_in = lstmcell_desc.datatype_in;
    handle->datatype_out = lstmcell_desc.datatype_out;
    handle->reuse = lstmcell_desc.reuse;
    handle->pass = lstmcell_desc.pass;
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
    handle->b_m1 = b_m1;
    handle->b_n1 = b_n1;
    handle->b_k1 = b_k1;
    handle->b_m2 = b_m2;
    handle->b_n2 = b_n2;
    handle->b_k2 = b_k2;
    if (handle->pass == 0) {
      handle->handlewx = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handleuh = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
#if defined(NON_FUSED_INPUT_GEMM)
      handle->handlett = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
#else
      handle->handlett = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m*4, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
#endif
    } else {
      handle->handlewx = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
      handle->handleuh = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->m, handle->n, &(handle->bm), &(handle->bm), &(handle->bn), &(handle->b_m1), &(handle->b_m1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
      handle->handlett = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->m, handle->k, handle->n, &(handle->bm), &(handle->bk), &(handle->bn), &(handle->b_m1), &(handle->b_k1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
      handle->handlewd = libxs_bgemm_handle_create(handle->nThreads, LIBXS_GEMM_PRECISION(float), LIBXS_GEMM_PRECISION(float),
        handle->k, handle->n, handle->m, &(handle->bk), &(handle->bn), &(handle->bm), &(handle->b_k1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */
    }
    /* Need to allocate space for scratch and internalstate libxs_dnn_tensor's */
    handle->i1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->i1b = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->i2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->f1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->f1b = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->f2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->o1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->o1b = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->o2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->c1t = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->c1b = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->c2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->i   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->f   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->o   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->c   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->dh  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d1  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d2  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d   = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->i3  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->f3  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->d4  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdht  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->deltat = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djddt  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdit  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdft  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdct  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdot  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdxt  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwi  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwf  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwo  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdwc  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdri  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdrf  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdro  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdrc  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbi  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbf  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbo  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdbc  = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->i4t    = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdiMt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdfMt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdcMt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->djdoMt = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
    handle->barrier = libxs_barrier_create(handle->nThreads, 1);
    if (NULL == handle->i1t || NULL == handle->i1b || NULL == handle->i2 || NULL == handle->f1t || NULL == handle->f1b ||
        NULL == handle->f2 || NULL == handle->o1t || NULL == handle->o1b || NULL == handle->o2 || NULL == handle->c1t ||
        NULL == handle->c1b || NULL == handle->c2 || NULL == handle->i || NULL == handle->f || NULL == handle->o ||
        NULL == handle->c || NULL == handle->dh || NULL == handle->d1 || NULL == handle->d2 || NULL == handle->d ||
        NULL == handle->i3 || NULL == handle->f3 || NULL == handle->d4 || NULL == handle->djdht || NULL == handle->deltat ||
        NULL == handle->djddt || NULL == handle->djdit || NULL == handle->djdft || NULL == handle->djdct ||
        NULL == handle->djdot || NULL == handle->djdxt || NULL == handle->djdwi || NULL == handle->djdwf ||
        NULL == handle->djdwo || NULL == handle->djdwc || NULL == handle->djdri || NULL == handle->djdrf ||
        NULL == handle->djdro || NULL == handle->djdrc || NULL == handle->djdbi || NULL == handle->djdbf ||
        NULL == handle->djdbo || NULL == handle->djdbc || NULL == handle->i4t   || NULL == handle->djdiMt||
        NULL == handle->djdfMt|| NULL == handle->djdoMt|| NULL == handle->djdcMt|| NULL == handle->barrier)
    {
      free(handle->i1t); free(handle->i1b); free(handle->i2); free(handle->f1t); free(handle->f1b);
      free(handle->f2); free(handle->o1t); free(handle->o1b); free(handle->o2); free(handle->c1t);
      free(handle->c1b); free(handle->c2); free(handle->i); free(handle->f); free(handle->o);
      free(handle->c); free(handle->dh); free(handle->d1); free(handle->d2); free(handle->d);
      free(handle->i3); free(handle->f3); free(handle->d4); free(handle->djdht); free(handle->deltat);
      free(handle->djddt); free(handle->djdit); free(handle->djdft); free(handle->djdct);
      free(handle->djdot); free(handle->djdxt); free(handle->djdwi); free(handle->djdwf);
      free(handle->djdwo); free(handle->djdwc); free(handle->djdri); free(handle->djdrf);
      free(handle->djdro); free(handle->djdrc); free(handle->djdbi); free(handle->djdbf);
      free(handle->djdbo); free(handle->djdbc); free(handle->i4t);
      free(handle->djdiMt);free(handle->djdfMt);free(handle->djdoMt);free(handle->djdcMt);
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
    free(handle->i1t); free(handle->i1b); free(handle->i2); free(handle->f1t); free(handle->f1b);
    free(handle->f2); free(handle->o1t); free(handle->o1b); free(handle->o2); free(handle->c1t);
    free(handle->c1b); free(handle->c2); free(handle->i); free(handle->f); free(handle->o);
    free(handle->c); free(handle->dh); free(handle->d1); free(handle->d2); free(handle->d);
    free(handle->i3); free(handle->f3); free(handle->d4); free(handle->djdht); free(handle->deltat);
    free(handle->djddt); free(handle->djdit); free(handle->djdft); free(handle->djdct);
    free(handle->djdot); free(handle->djdxt); free(handle->djdwi); free(handle->djdwf);
    free(handle->djdwo); free(handle->djdwc); free(handle->djdri); free(handle->djdrf);
    free(handle->djdro); free(handle->djdrc); free(handle->djdbi); free(handle->djdbf);
    free(handle->djdbo); free(handle->djdbc); free(handle->i4t);
    free(handle->djdiMt);free(handle->djdfMt);free(handle->djdoMt);free(handle->djdcMt);
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
#if defined(NON_FUSED_INPUT_GEMM)
                  layout->dim_size[3] = handle->m / handle->bm;
#else
                  layout->dim_size[3] = (handle->m * 4) / handle->bm;
#endif
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
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* i2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* f2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* o2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* c2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* dh */
                                           size += 64;
#if !defined(NON_FUSED_INPUT_GEMM)
                                           size += (size_t)handle->m * 4 * handle->n * sizeof_datatype * handle->t; /* i4t */
                                           size += 64;
#endif
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
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
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
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
#if !defined(NON_FUSED_INPUT_GEMM)
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i4t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i4t->data = (void*)(address+offset);
                                           }
#endif
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
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
                                           handle->i4t->data = 0;
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
                                           handle->i4t = 0;
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
                                           handle->deltat->data = 0;
                                           handle->djdiMt->data = 0;
                                           handle->djdfMt->data = 0;
                                           handle->djdoMt->data = 0;
                                           handle->djdcMt->data = 0;
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
                                           handle->deltat = 0;
                                           handle->djdiMt = 0;
                                           handle->djdfMt = 0;
                                           handle->djdoMt = 0;
                                           handle->djdcMt = 0;
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
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* i */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* f */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* o */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* c */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * ((size_t)handle->t + 1); /* d */
                                           size += 64;
                                         } break;
      case LIBXS_DNN_COMPUTE_KIND_BWD:
      case LIBXS_DNN_COMPUTE_KIND_UPD:
      case LIBXS_DNN_COMPUTE_KIND_ALL: {
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_assign_internalstate(libxs_dnn_lstmcell* handle, const void* igoldtb, const void* fgoldtb, const void* ogoldtb, const void* cgoldtb, const void* dgoldtb)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;

  if (handle != 0 && igoldtb != 0 && fgoldtb != 0 && ogoldtb != 0 && cgoldtb != 0 && dgoldtb != 0) {
    const libxs_blasint m = handle->m, n = handle->n, t = handle->t;
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, igold, (const LIBXS_DNN_ELTWISE_FTYPE*)igoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i, (LIBXS_DNN_ELTWISE_FTYPE*)handle->i->data, m * n);
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, fgold, (const LIBXS_DNN_ELTWISE_FTYPE*)fgoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f, (LIBXS_DNN_ELTWISE_FTYPE*)handle->f->data, m * n);
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, ogold, (const LIBXS_DNN_ELTWISE_FTYPE*)ogoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o, (LIBXS_DNN_ELTWISE_FTYPE*)handle->o->data, m * n);
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, cgold, (const LIBXS_DNN_ELTWISE_FTYPE*)cgoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c, (LIBXS_DNN_ELTWISE_FTYPE*)handle->c->data, m * n);
    LIBXS_VLA_DECL(2, const LIBXS_DNN_ELTWISE_FTYPE, dgold, (const LIBXS_DNN_ELTWISE_FTYPE*)dgoldtb, m * n);
    LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, d, (LIBXS_DNN_ELTWISE_FTYPE*)handle->d->data, m * n);
    libxs_blasint it;
    for (it = 0; it < t; ++it) {
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, igold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, i, it, 0, m * n));
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, fgold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, f, it, 0, m * n));
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, ogold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, o, it, 0, m * n));
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, cgold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, c, it, 0, m * n));
      libxs_bgemm_copyin_b(handle->handlewd, &LIBXS_VLA_ACCESS(2, dgold, it, 0, m * n), &m, &LIBXS_VLA_ACCESS(2, d, it, 0, m * n));
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


LIBXS_API void libxs_dnn_lstmcell_split_wx(libxs_dnn_lstmcell* lstm, libxs_blasint offset, void* src, void* dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  int size;
  libxs_blasint job, i, j, k, l, p;
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  libxs_blasint t = lstm->t;
  libxs_blasint bm = lstm->bm;
  libxs_blasint bn = lstm->bn;
  libxs_blasint mb = m / bm;
  libxs_blasint nb = n / bn;
  LIBXS_VLA_DECL(5, LIBXS_DNN_ELTWISE_FTYPE, real_src, (LIBXS_DNN_ELTWISE_FTYPE*)src, nb, mb*4, bn, bm);
  LIBXS_VLA_DECL(5, LIBXS_DNN_ELTWISE_FTYPE, real_dst, (LIBXS_DNN_ELTWISE_FTYPE*)dst, nb, mb, bn, bm);
  ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  size = t*n*m;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  for (job = thr_begin; job < thr_end; job++) {
    i = job / (n * m);
    j = (job % (n * m)) / (m * bn);
    k = ((job % (n * m)) % (m * bn)) / (bn * bm);
    l = (((job % (n * m)) % (m * bn)) % (bn * bm)) / bm;
    p = (((job % (n * m)) % (m * bn)) % (bn * bm)) % bm;
    LIBXS_VLA_ACCESS(5, real_dst, i, j, k, l, p, nb, mb, bn, bm) =
      LIBXS_VLA_ACCESS(5, real_src, i, j, mb*offset + k, l, p, nb, mb*4, bn, bm);
  }
}

LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_fwd(libxs_dnn_lstmcell* lstm, int start_thread, int tid)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  /*libxs_blasint k = lstm->k;*/
  libxs_blasint t = lstm->t;
#if defined(LSTM_TIMING)
  libxs_blasint k = lstm->k;
  const double tflops = 12;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n) + (tflops * m * n)) * 4.0 + (4.0 * m * n) + (tflops * m * n)) * (double)t * 1E-9;
#endif
  int reuse = lstm->reuse;
  LIBXS_DNN_ELTWISE_FTYPE *wi  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wi->data;
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXS_DNN_ELTWISE_FTYPE *wf  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXS_DNN_ELTWISE_FTYPE *wo  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXS_DNN_ELTWISE_FTYPE *wc  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wc->data;
#endif
  LIBXS_DNN_ELTWISE_FTYPE *xt  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ri  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXS_DNN_ELTWISE_FTYPE *rf  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXS_DNN_ELTWISE_FTYPE *ro  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXS_DNN_ELTWISE_FTYPE *rc  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXS_DNN_ELTWISE_FTYPE *h   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->h->data;
  LIBXS_DNN_ELTWISE_FTYPE *bi  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->bi->data;
  LIBXS_DNN_ELTWISE_FTYPE *bf  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->bf->data;
  LIBXS_DNN_ELTWISE_FTYPE *bo  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->bo->data;
  LIBXS_DNN_ELTWISE_FTYPE *bc  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->bc->data;
  LIBXS_DNN_ELTWISE_FTYPE *i2t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i2->data;
  LIBXS_DNN_ELTWISE_FTYPE *f2t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f2->data;
  LIBXS_DNN_ELTWISE_FTYPE *o2t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o2->data;
  LIBXS_DNN_ELTWISE_FTYPE *c2t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c2->data;
  LIBXS_DNN_ELTWISE_FTYPE *i   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXS_DNN_ELTWISE_FTYPE *f   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXS_DNN_ELTWISE_FTYPE *o   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXS_DNN_ELTWISE_FTYPE *c   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXS_DNN_ELTWISE_FTYPE *dh  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->dh->data;
  LIBXS_DNN_ELTWISE_FTYPE *d1  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d1->data;
  LIBXS_DNN_ELTWISE_FTYPE *d2  = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d2->data;
  LIBXS_DNN_ELTWISE_FTYPE *d   = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->d->data;
  /* LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, x, xt, k * n); */
  LIBXS_DNN_ELTWISE_FTYPE *i1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *f1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->f1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *o1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->o1t->data;
  LIBXS_DNN_ELTWISE_FTYPE *c1t = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->c1t->data;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c1, c1t, m * n);
  /* libxs_bgemm_handle *handlewx = lstm->handlewx; */
  libxs_bgemm_handle *handleuh = lstm->handleuh;
  libxs_bgemm_handle *handlett = lstm->handlett;
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, inr, i, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, fnr, f, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, onr, o, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, cnr, c, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dnr, d, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, i2, i2t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, f2, f2t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, o2, o2t, m * n);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, c2, c2t, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  int j;
  const int ltid = tid - start_thread;

  libxs_barrier_init(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) { start = libxs_timer_tick(); }
#endif

  if (reuse) {
#if defined(LSTM_TIMING)
    if (ltid == 0) { Gbl_t_input = libxs_timer_tick(); }
#endif
#if defined(NON_FUSED_INPUT_GEMM)
    libxs_bgemm_st(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, wf, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, wo, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, wc, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), start_thread, tid);
#else
    libxs_bgemm_st(handlett, wi, xt, (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i4t->data, start_thread, tid);
    libxs_dnn_lstmcell_split_wx(lstm, 0, lstm->i4t->data, lstm->i1t->data, start_thread, tid, lstm->nThreads);
    libxs_dnn_lstmcell_split_wx(lstm, 1, lstm->i4t->data, lstm->f1t->data, start_thread, tid, lstm->nThreads);
    libxs_dnn_lstmcell_split_wx(lstm, 2, lstm->i4t->data, lstm->o1t->data, start_thread, tid, lstm->nThreads);
    libxs_dnn_lstmcell_split_wx(lstm, 3, lstm->i4t->data, lstm->c1t->data, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
#endif
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
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), bi, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), bf, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), bo, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), bc, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxs_internal_recursive_step(handleuh, ri, h, &LIBXS_VLA_ACCESS(2, i2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rf, h, &LIBXS_VLA_ACCESS(2, f2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, ro, h, &LIBXS_VLA_ACCESS(2, o2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rc, h, &LIBXS_VLA_ACCESS(2, c2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 3, m * n, start_thread, tid); /*tanh*/
      libxs_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, f, d, d1, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_eltwise_mult(m*n, i, c, d2, start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
      libxs_internal_matrix_add(m*n, d1, d2, d, start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      libxs_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxs_timer_tick();
      }
#endif
      libxs_internal_matrix_tanh(m*n, d, dh, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      libxs_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxs_timer_tick();
      }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, o, dh, h, start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
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
#if defined(NON_FUSED_INPUT_GEMM)
    libxs_bgemm_st(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, wf, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, wo, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), start_thread, tid);
    libxs_bgemm_st(handlett, wc, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), start_thread, tid);
#else
    libxs_bgemm_st(handlett, wi, xt, (LIBXS_DNN_ELTWISE_FTYPE*)lstm->i4t->data, start_thread, tid);
    libxs_dnn_lstmcell_split_wx(lstm, 0, lstm->i4t->data, lstm->i1t->data, start_thread, tid, lstm->nThreads);
    libxs_dnn_lstmcell_split_wx(lstm, 1, lstm->i4t->data, lstm->f1t->data, start_thread, tid, lstm->nThreads);
    libxs_dnn_lstmcell_split_wx(lstm, 2, lstm->i4t->data, lstm->o1t->data, start_thread, tid, lstm->nThreads);
    libxs_dnn_lstmcell_split_wx(lstm, 3, lstm->i4t->data, lstm->c1t->data, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
#endif
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
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), bi, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), bf, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), bo, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), bc, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxs_internal_recursive_step(handleuh, ri, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rf, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, f2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, ro, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxs_internal_recursive_step(handleuh, rc, &LIBXS_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c2, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), 3, m * n, start_thread, tid); /*tanh*/
      libxs_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxs_timer_tick(); }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, dnr, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXS_VLA_ACCESS(2, cnr, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
      libxs_internal_matrix_add(m*n, d1, d2, &LIBXS_VLA_ACCESS(2, dnr, j+1, 0, m * n), start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      libxs_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxs_timer_tick();
      }
#endif
      libxs_internal_matrix_tanh(m*n, &LIBXS_VLA_ACCESS(2, dnr, j+1, 0, m * n), dh, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      libxs_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxs_timer_tick();
      }
#endif
      libxs_internal_matrix_eltwise_mult(m*n, &LIBXS_VLA_ACCESS(2, onr, j, 0, m * n), dh, &LIBXS_VLA_ACCESS(2, hnr, j+1, 0, m * n), start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
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


LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bwd_upd_bu(libxs_dnn_lstmcell* lstm, int start_thread, int tid, int pass)
{
  libxs_dnn_err_t status = LIBXS_DNN_SUCCESS;
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  libxs_blasint k = lstm->k;
  libxs_blasint t = lstm->t;
  LIBXS_DNN_ELTWISE_FTYPE *wi = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wi->data;
  LIBXS_DNN_ELTWISE_FTYPE *wf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXS_DNN_ELTWISE_FTYPE *wo = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXS_DNN_ELTWISE_FTYPE *wc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->wc->data;
  LIBXS_DNN_ELTWISE_FTYPE *xt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXS_DNN_ELTWISE_FTYPE *ri = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXS_DNN_ELTWISE_FTYPE *rf = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXS_DNN_ELTWISE_FTYPE *ro = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXS_DNN_ELTWISE_FTYPE *rc = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXS_DNN_ELTWISE_FTYPE *ht = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->h->data;
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
  LIBXS_DNN_ELTWISE_FTYPE *djdiMt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdiMt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdfMt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdfMt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdcMt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdcMt->data;
  LIBXS_DNN_ELTWISE_FTYPE *djdoMt = (LIBXS_DNN_ELTWISE_FTYPE*)lstm->djdoMt->data;
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
  libxs_bgemm_handle *handleud = lstm->handlewx;
  libxs_bgemm_handle *handledh = lstm->handleuh;
  libxs_bgemm_handle *handledx = lstm->handlett;
  libxs_bgemm_handle *handlewd = lstm->handlewd;
  int j;
  const int ltid = tid - start_thread;

  libxs_barrier_init(lstm->barrier, ltid);
  /* compute delta */
  libxs_internal_matrix_copy(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdd */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), d1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), d2, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdc */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), c1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), c2, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdi */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, c, t-1, 0, m * n), i1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdf */
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXS_VLA_ACCESS(2, d, t-2, 0, m * n), f1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdo */
  libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, t-1, 0, m * n), o1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
  libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
  libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  libxs_barrier_wait(lstm->barrier, ltid);
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
    libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdd */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_tanh_inverse(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, d1, d2, d3, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXS_VLA_ACCESS(2, f, j+1, 0, m * n), d4, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_add(m * n, d3, d4, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdc */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), c1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_complement_square(m * n, &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), c2, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdi */
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, c, j, 0, m * n), i1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, i, j, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdf */
    if (j >= 1) {
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXS_VLA_ACCESS(2, d, j-1, 0, m * n), f1, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
      libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, f, j, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
      libxs_barrier_wait(lstm->barrier, ltid);
      libxs_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
    } else {
      /* djdf is zero for j == 0 */
      libxs_internal_matrix_zero(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
    }
    /* compute djdo */
    libxs_internal_matrix_tanh(m * n, &LIBXS_VLA_ACCESS(2, d, j, 0, m * n), o1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_complement(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
    libxs_internal_matrix_eltwise_mult(m * n, &LIBXS_VLA_ACCESS(2, o, j, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
    libxs_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), start_thread, tid, lstm->nThreads);
    libxs_barrier_wait(lstm->barrier, ltid);
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
      libxs_barrier_wait(lstm->barrier, ltid);
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
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo, start_thread, tid, lstm->nThreads);
      libxs_internal_matrix_add(m * n, &LIBXS_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc, start_thread, tid, lstm->nThreads);
    }
    libxs_barrier_wait(lstm->barrier, ltid);
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

