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
#ifndef LIBXS_DNN_LSTMCELL_H
#define LIBXS_DNN_LSTMCELL_H

#include "libxs_macros.h"
#include "libxs_typedefs.h"
#include "libxs_dnn.h"


LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_lstmcell_desc {
  int N;
  int nThreads;
  int m; /* number of outputs */
  int n; /* size of the minibatch */
  int k; /* number of inputs */
  int t; /* number of time steps */
  int bm; /* blocksize for m */
  int bn; /* blocksize for n */
  int bk; /* blocksize for k */
  int b_m1; /* b_?? parameters are used in libxs_bgemm */
  int b_n1;
  int b_k1;
  int b_m2;
  int b_n2;
  int b_k2;
  libxs_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxs_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxs_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
} libxs_dnn_lstmcell_desc;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_dnn_lstmcell {
  int N;
  int nThreads;
  libxs_dnn_lstmcell_desc desc;
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
  int b_m1;
  int b_n1;
  int b_k1;
  int b_m2;
  int b_n2;
  int b_k2;
  libxs_dnn_tensor* wi;
  libxs_dnn_tensor* wf;
  libxs_dnn_tensor* wo;
  libxs_dnn_tensor* wc;
  libxs_dnn_tensor* xt;
  libxs_dnn_tensor* ri;
  libxs_dnn_tensor* rf;
  libxs_dnn_tensor* ro;
  libxs_dnn_tensor* rc;
  /* Currently we are not using the following 4 bias terms */
  libxs_dnn_tensor* bi;
  libxs_dnn_tensor* bf;
  libxs_dnn_tensor* bo;
  libxs_dnn_tensor* bc;
  libxs_dnn_tensor* h;
  libxs_dnn_tensor* i1t;
  libxs_dnn_tensor* i1b;
  libxs_dnn_tensor* i2;
  libxs_dnn_tensor* f1t;
  libxs_dnn_tensor* f1b;
  libxs_dnn_tensor* f2;
  libxs_dnn_tensor* o1t;
  libxs_dnn_tensor* o1b;
  libxs_dnn_tensor* o2;
  libxs_dnn_tensor* c1t;
  libxs_dnn_tensor* c1b;
  libxs_dnn_tensor* c2;
  libxs_dnn_tensor* i;
  libxs_dnn_tensor* f;
  libxs_dnn_tensor* o;
  libxs_dnn_tensor* c;
  libxs_dnn_tensor* dh;
  libxs_dnn_tensor* d1;
  libxs_dnn_tensor* d2;
  libxs_dnn_tensor* d;
  libxs_dnn_tensor* i3;
  libxs_dnn_tensor* f3;
  libxs_dnn_tensor* d4;
  libxs_dnn_tensor* djdht;
  libxs_dnn_tensor* deltat;
  libxs_dnn_tensor* djddt;
  libxs_dnn_tensor* djdit;
  libxs_dnn_tensor* djdft;
  libxs_dnn_tensor* djdct;
  libxs_dnn_tensor* djdot;
  libxs_dnn_tensor* djdxt;
  libxs_dnn_tensor* djdwi;
  libxs_dnn_tensor* djdwf;
  libxs_dnn_tensor* djdwo;
  libxs_dnn_tensor* djdwc;
  libxs_dnn_tensor* djdri;
  libxs_dnn_tensor* djdrf;
  libxs_dnn_tensor* djdro;
  libxs_dnn_tensor* djdrc;
  libxs_dnn_tensor* djdbi;
  libxs_dnn_tensor* djdbf;
  libxs_dnn_tensor* djdbo;
  libxs_dnn_tensor* djdbc;
  libxs_dnn_tensor* rTp;
  libxs_dnn_tensor* wTp;
  libxs_dnn_tensor* deltaTp;
  libxs_dnn_tensor* xTp;
  libxs_bgemm_handle* handlewx;
  libxs_bgemm_handle* handleuh;
  libxs_bgemm_handle* handlett;
  libxs_bgemm_handle* handlewd;
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

LIBXS_API libxs_dnn_err_t libxs_dnn_lstmcell_execute_st(libxs_dnn_lstmcell* handle, libxs_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXS_DNN_LSTMCELL_H*/
