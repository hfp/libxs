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
#ifndef LIBXS_DNN_GRUCELL_H
#define LIBXS_DNN_GRUCELL_H

#include "libxs_macros.h"
#include "libxs_typedefs.h"
#include "libxs_dnn.h"


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_grucell_desc {
  int N;
  int nThreads;
  int m;     /* number of outputs */
  int n;     /* size of the minibatch */
  int k;     /* number of inputs */
  int t;     /* number of time steps */
  int bm;    /* blocksize for m */
  int bn;    /* blocksize for n */
  int bk;    /* blocksize for k */
  int reuse; /* reuse/overwrite memory for FWD */
  int pass;  /* denotes whether it is FWD/BWD/UPD */
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
} libxs_dnn_grucell_desc;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_grucell {
  int N;
  int nThreads;
  libxs_dnn_grucell_desc desc;
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  int m;
  int n;
  int k;
  int t;
  int bm;
  int bn;
  int bk;
  int reuse;
  int pass;
  int b_m1;
  int b_n1;
  int b_k1;
  int b_m2;
  int b_n2;
  int b_k2;
  libxs_dnn_tensor* wr;
  libxs_dnn_tensor* wz;
  libxs_dnn_tensor* wg;
  libxs_dnn_tensor* xt;
  libxs_dnn_tensor* ur;
  libxs_dnn_tensor* uz;
  libxs_dnn_tensor* ug;
  libxs_dnn_tensor* br;
  libxs_dnn_tensor* bz;
  libxs_dnn_tensor* bg;
  libxs_dnn_tensor* h;
  libxs_dnn_tensor* r1t;
  libxs_dnn_tensor* r2t;
  libxs_dnn_tensor* z1t;
  libxs_dnn_tensor* z2t;
  libxs_dnn_tensor* g1t;
  libxs_dnn_tensor* g2t;
  libxs_dnn_tensor* g3;
  libxs_dnn_tensor* h1;
  libxs_dnn_tensor* h2;
  libxs_dnn_tensor* h3;
  libxs_dnn_tensor* r;
  libxs_dnn_tensor* z;
  libxs_dnn_tensor* g;
  libxs_dnn_tensor* d3;
  libxs_dnn_tensor* d4;
  libxs_dnn_tensor* d5;
  libxs_dnn_tensor* d6;
  libxs_dnn_tensor* d7;
  libxs_dnn_tensor* d8;
  libxs_dnn_tensor* d9;
  libxs_dnn_tensor* d10;
  libxs_dnn_tensor* d11;
  libxs_dnn_tensor* d12;
  libxs_dnn_tensor* d13;
  libxs_dnn_tensor* d14;
  libxs_dnn_tensor* d15;
  libxs_dnn_tensor* d16;
  libxs_dnn_tensor* d17;
  libxs_dnn_tensor* d18;
  libxs_dnn_tensor* d19;
  libxs_dnn_tensor* d20;
  libxs_dnn_tensor* d21;
  libxs_dnn_tensor* d22;
  libxs_dnn_tensor* d23;
  libxs_dnn_tensor* d10M;
  libxs_dnn_tensor* d11M;
  libxs_dnn_tensor* d18M;
  libxs_dnn_tensor* hrTp;
  libxs_dnn_tensor* djdwr;
  libxs_dnn_tensor* djdwz;
  libxs_dnn_tensor* djdwg;
  libxs_dnn_tensor* djdxt;
  libxs_dnn_tensor* djdur;
  libxs_dnn_tensor* djduz;
  libxs_dnn_tensor* djdug;
  libxs_dnn_tensor* djdht;
  libxs_dnn_tensor* djdbr;
  libxs_dnn_tensor* djdbz;
  libxs_dnn_tensor* djdbg;
  libxs_bgemm_handle* handleux;
  libxs_bgemm_handle* handlewh;
  libxs_bgemm_handle* handlett;
  libxs_bgemm_handle* handlewd;
  libxs_barrier* barrier; /* barrier */
} libxs_dnn_grucell;

LIBXS_API libxs_dnn_grucell* libxs_dnn_create_grucell(libxs_dnn_grucell_desc grucell_desc, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_destroy_grucell(const libxs_dnn_grucell* handle);

LIBXS_API libxs_dnn_tensor_datalayout* libxs_dnn_grucell_create_tensor_datalayout(const libxs_dnn_grucell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);

LIBXS_API size_t libxs_dnn_grucell_get_scratch_size(const libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bind_scratch(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, const void* scratch);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_release_scratch(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API size_t libxs_dnn_grucell_get_internalstate_size(const libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bind_internalstate(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind, const void* internalstate);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_release_internalstate(libxs_dnn_grucell* handle, const libxs_dnn_compute_kind kind);

LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_assign_internalstate(libxs_dnn_grucell* handle, const void* rgoldtb, const void* zgoldtb, const void* ggoldtb);

LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bind_tensor(libxs_dnn_grucell* handle, const libxs_dnn_tensor* tensor, const libxs_dnn_tensor_type type);
LIBXS_API libxs_dnn_tensor* libxs_dnn_grucell_get_tensor(libxs_dnn_grucell* handle, const libxs_dnn_tensor_type type, libxs_dnn_err_t* status);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_release_tensor(libxs_dnn_grucell* handle, const libxs_dnn_tensor_type type);

LIBXS_API void libxs_dnn_grucell_matrix_transpose_b(libxs_dnn_grucell* gru, void* src, void* dst, int start_thread, int tid, int nthreads);

LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_fwd(libxs_dnn_grucell* gru, int start_thread, int tid);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_bwd_upd_bu(libxs_dnn_grucell* gru, int start_thread, int tid, int pass);
LIBXS_API libxs_dnn_err_t libxs_dnn_grucell_execute_st(libxs_dnn_grucell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_GRUCELL_H*/
