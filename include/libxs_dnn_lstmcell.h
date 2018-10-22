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
#ifndef LIBXS_DNN_LSTMCELL_H
#define LIBXS_DNN_LSTMCELL_H

#include "libxs_macros.h"
#include "libxs_typedefs.h"
#include "libxs_dnn.h"


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_lstmcell_desc {
  int nThreads;
  int K;     /* number of outputs */
  int N;     /* size of the minibatch */
  int C;     /* number of inputs */
  int t;     /* number of time steps */
  int pass;  /* denotes whether it is FWD/BWD/UPD */
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for activation buffers */
  libxs_dnn_tensor_format filter_format;  /* format which is for filter buffers */
} libxs_dnn_lstmcell_desc;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_lstmcell {
  libxs_dnn_lstmcell_desc desc;
  libxs_dnn_internal_format custom_format_type; /* required only for comparing layouts  */
  libxs_blasint bk;
  libxs_blasint bn;
  libxs_blasint bc;
  libxs_dnn_tensor* xt;
  libxs_dnn_tensor* csp;
  libxs_dnn_tensor* hp;
  libxs_dnn_tensor* w;
  libxs_dnn_tensor* r;
  libxs_dnn_tensor* b;
  libxs_dnn_tensor* cst;
  libxs_dnn_tensor* ht;
  libxs_dnn_tensor* it;
  libxs_dnn_tensor* ft;
  libxs_dnn_tensor* ot;
  libxs_dnn_tensor* cit;
  libxs_dnn_tensor* cot;
  libxs_dnn_tensor* dxt;
  libxs_dnn_tensor* dcspt;
  libxs_dnn_tensor* dhpt;
  libxs_dnn_tensor* dw;
  libxs_dnn_tensor* dr;
  libxs_dnn_tensor* db;
  libxs_dnn_tensor* dcs;
  libxs_dnn_tensor* dht;
  libxs_dnn_tensor* dit;
  libxs_dnn_tensor* dft;
  libxs_dnn_tensor* dot;
  libxs_dnn_tensor* dcit;
  libxs_dnn_tensor* doutt;
  libxs_dnn_tensor* t1;
  libxs_dnn_tensor* t2;
  libxs_barrier* barrier;
} libxs_dnn_lstmcell;

LIBXS_API libxs_dnn_lstmcell* libxs_dnn_create_lstmcell(libxs_dnn_lstmcell_desc lstmcell_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_lstmcell(const libxs_dnn_lstmcell* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_lstmcell_create_tensor_datalayout(const libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_lstmcell_get_scratch_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_scratch(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API size_t libxs_dnn_lstmcell_get_internalstate_size(const libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind, const void* internalstate);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_internalstate(libxs_dnn_lstmcell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bind_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_lstmcell_get_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_release_tensor(libxs_dnn_lstmcell* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_fwd(libxs_dnn_lstmcell* lstm, int start_thread, int tid);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_bwd_upd_bu(libxs_dnn_lstmcell* lstm, int start_thread, int tid, int pass);
LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_execute_st(libxs_dnn_lstmcell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_LSTMCELL_H*/
