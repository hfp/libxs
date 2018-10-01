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
#ifndef LIBXS_DNN_RNNCELL_H
#define LIBXS_DNN_RNNCELL_H

#include "libxs_macros.h"
#include "libxs_typedefs.h"
#include "libxs_dnn.h"


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_rnncell_desc {
  int nThreads;
  int K;     /* number of outputs */
  int N;     /* size of the minibatch */
  int C;     /* number of inputs */
  int t;     /* number of time steps */
#if 0
  int bm;    /* blocksize for m */
  int bn;    /* blocksize for n */
  int bk;    /* blocksize for k */
#endif
  int pass;  /* denotes whether it is FWD/BWD/UPD */
  int nonlin;/* denotes which non-linear function to be used: relu, sigmoid, tanh */
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  libxs_dnn_tensor_format filter_format;  /* format which is for buffer buffers */
} libxs_dnn_rnncell_desc;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_rnncell {
  libxs_dnn_rnncell_desc desc;
  libxs_dnn_internal_format custom_format_type; /* required only for comparing layouts  */
  libxs_blasint bk;
  libxs_blasint bn;
  libxs_blasint bc;
#if 0
  libxs_blasint b_m1;
  libxs_blasint b_n1;
  libxs_blasint b_k1;
  libxs_blasint b_m2;
  libxs_blasint b_n2;
  libxs_blasint b_k2;
#endif
  libxs_dnn_tensor* w;
  libxs_dnn_tensor* xt;
  libxs_dnn_tensor* u;
  libxs_dnn_tensor* h;
  libxs_dnn_tensor* b;
  libxs_dnn_tensor* z;
  libxs_dnn_tensor* djdht;
  libxs_dnn_tensor* djdu;
  libxs_dnn_tensor* djdw;
  libxs_dnn_tensor* djdxt;
  libxs_dnn_tensor* djdb;
  libxs_dnn_tensor* bM;
  libxs_dnn_tensor* deltat;
  libxs_dnn_tensor* z1;
  libxs_dnn_tensor* z2;
  libxs_dnn_tensor* di1;
  libxs_dnn_tensor* di2;
  libxs_dnn_tensor* deltaMt;
  libxs_bgemm_handle* handlewx;
  libxs_bgemm_handle* handleuh;
  libxs_bgemm_handle* handlett;
  libxs_bgemm_handle* handlewd;
  libxs_barrier* barrier; /* barrier */
} libxs_dnn_rnncell;

LIBXS_API libxs_dnn_rnncell* libxs_dnn_create_rnncell(libxs_dnn_rnncell_desc rnncell_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_rnncell(const libxs_dnn_rnncell* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_rnncell_create_tensor_datalayout(const libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_rnncell_get_scratch_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_scratch(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API size_t libxs_dnn_rnncell_get_internalstate_size(const libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind, const void* internalstate);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_internalstate(libxs_dnn_rnncell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_assign_internalstate(libxs_dnn_rnncell* handle, const void* zgoldt);

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bind_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_rnncell_get_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_release_tensor(libxs_dnn_rnncell* handle, const libxs_dnn_tensor_type type);

LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_fwd(libxs_dnn_rnncell* rnn, int start_thread, int tid);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_bwd_upd_bu(libxs_dnn_rnncell* rnn, int start_thread, int tid, int pass);
LIBXS_API libxs_dnn_err_t libxs_dnn_rnncell_execute_st(libxs_dnn_rnncell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_RNNCELL_H*/
