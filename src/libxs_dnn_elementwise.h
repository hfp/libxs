/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_DNN_ELEMENTWISE_H
#define LIBXS_DNN_ELEMENTWISE_H

#include <libxs.h>

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

LIBXS_API_INTERN void libxs_internal_matrix_zero_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst);
LIBXS_API_INTERN void libxs_internal_matrix_add_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_sub_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_copy_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *srcdst);
LIBXS_API_INTERN void libxs_internal_matrix_eltwise_fma_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst);
LIBXS_API_INTERN void libxs_internal_matrix_add_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv);
LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv);
LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_const_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv, LIBXS_DNN_ELTWISE_FTYPE const_bias);
LIBXS_API_INTERN void libxs_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, libxs_bfloat16 *colv);
LIBXS_API_INTERN void libxs_internal_matrix_bcst_cvt_bf16_fp32_colvector_const_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, libxs_bfloat16 *colv, LIBXS_DNN_ELTWISE_FTYPE const_bias);
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
LIBXS_API_INTERN void libxs_internal_matrix_rne_mask_fp32_bfp16_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, float* src, float* dst);
LIBXS_API_INTERN void libxs_internal_matrix_rne_cvt_fp32_bfp16_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, float* src, libxs_bfloat16* dst);
LIBXS_API_INTERN void libxs_internal_matrix_cvt_bf16_fp32_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_bfloat16 *src, LIBXS_DNN_ELTWISE_FTYPE *dst);
#endif /*LIBXS_DNN_ELEMENTWISE_H*/
