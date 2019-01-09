/******************************************************************************
** Copyright (c) 2018-2019, Intel Corporation                                **
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
#ifndef LIBXS_DNN_ELEMENTWISE_H
#define LIBXS_DNN_ELEMENTWISE_H

#include <libxs_blocked_gemm.h>

#if !defined(LIBXS_DNN_ELTWISE_FTYPE)
# define LIBXS_DNN_ELTWISE_FTYPE float
#endif


LIBXS_API_INTERN void libxs_internal_matrix_zero(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_add(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *a, LIBXS_DNN_ELTWISE_FTYPE *b, LIBXS_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_eltwise_mult(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *a, LIBXS_DNN_ELTWISE_FTYPE *b, LIBXS_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_sigmoid(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_tanh(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_relu(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_tanh_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_transpose(libxs_blasint rows, libxs_blasint cols, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_copy(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_complement(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_complement_square(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_matrix_1D_2D(libxs_blasint m, libxs_blasint n, libxs_blasint bm, libxs_blasint bn, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads);
LIBXS_API_INTERN void libxs_internal_recursive_step(libxs_blocked_gemm_handle* handle, LIBXS_DNN_ELTWISE_FTYPE* u, LIBXS_DNN_ELTWISE_FTYPE* h, LIBXS_DNN_ELTWISE_FTYPE* op1, LIBXS_DNN_ELTWISE_FTYPE *op2,
  LIBXS_DNN_ELTWISE_FTYPE *temp, LIBXS_DNN_ELTWISE_FTYPE *dst, int act, libxs_blasint size, int start_thread, int tid);

LIBXS_API_INTERN void libxs_internal_matrix_zero_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst);
LIBXS_API_INTERN void libxs_internal_matrix_add_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_copy_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *srcdst);
LIBXS_API_INTERN void libxs_internal_matrix_eltwise_fma_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_add_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv);
LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv);
LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_const_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv, LIBXS_DNN_ELTWISE_FTYPE const_bias);
LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_tanh_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_relu_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_tanh_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_tanh_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);

LIBXS_API_INTERN void libxs_internal_matrix_complement_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_complement_square_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);

LIBXS_API_INTERN void libxs_internal_compute_dcp_dci_di_df_dp_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, int timestep, int t, LIBXS_DNN_ELTWISE_FTYPE *dout, LIBXS_DNN_ELTWISE_FTYPE *dh, LIBXS_DNN_ELTWISE_FTYPE *o, LIBXS_DNN_ELTWISE_FTYPE *co, LIBXS_DNN_ELTWISE_FTYPE *dcs, LIBXS_DNN_ELTWISE_FTYPE *ii, LIBXS_DNN_ELTWISE_FTYPE *ci, LIBXS_DNN_ELTWISE_FTYPE *dci, LIBXS_DNN_ELTWISE_FTYPE *di, LIBXS_DNN_ELTWISE_FTYPE *cps, LIBXS_DNN_ELTWISE_FTYPE *f, LIBXS_DNN_ELTWISE_FTYPE *df, LIBXS_DNN_ELTWISE_FTYPE *dp, LIBXS_DNN_ELTWISE_FTYPE *dcp);

LIBXS_API_INTERN void libxs_internal_compute_o_i_f_ci_cs_co_h_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *f, LIBXS_DNN_ELTWISE_FTYPE *cps, LIBXS_DNN_ELTWISE_FTYPE *cs, LIBXS_DNN_ELTWISE_FTYPE *ii, LIBXS_DNN_ELTWISE_FTYPE *ci,LIBXS_DNN_ELTWISE_FTYPE *co, LIBXS_DNN_ELTWISE_FTYPE *o, LIBXS_DNN_ELTWISE_FTYPE *h);

LIBXS_API_INTERN void libxs_internal_compute_dcp_dci_di_df_dp_ld_and_reformat_dci_di_df_dp_ld2(libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_blasint ld2, int timestep, int t, LIBXS_DNN_ELTWISE_FTYPE *dout, LIBXS_DNN_ELTWISE_FTYPE *dh, LIBXS_DNN_ELTWISE_FTYPE *o, LIBXS_DNN_ELTWISE_FTYPE *co, LIBXS_DNN_ELTWISE_FTYPE *dcs, LIBXS_DNN_ELTWISE_FTYPE *ii, LIBXS_DNN_ELTWISE_FTYPE *ci, LIBXS_DNN_ELTWISE_FTYPE *dci, LIBXS_DNN_ELTWISE_FTYPE *di, LIBXS_DNN_ELTWISE_FTYPE *cps, LIBXS_DNN_ELTWISE_FTYPE *f, LIBXS_DNN_ELTWISE_FTYPE *df, LIBXS_DNN_ELTWISE_FTYPE *dp, LIBXS_DNN_ELTWISE_FTYPE *dcp, LIBXS_DNN_ELTWISE_FTYPE *dciB, LIBXS_DNN_ELTWISE_FTYPE *diB, LIBXS_DNN_ELTWISE_FTYPE *dfB, LIBXS_DNN_ELTWISE_FTYPE *dpB);

#endif /*LIBXS_DNN_ELEMENTWISE_H*/
