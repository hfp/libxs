/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "libxs_dnn_elementwise.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API_INTERN void libxs_internal_matrix_zero(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    src[i] = (LIBXS_DNN_ELTWISE_FTYPE)0;
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_add(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *a, LIBXS_DNN_ELTWISE_FTYPE *b, LIBXS_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_eltwise_mult(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *a, LIBXS_DNN_ELTWISE_FTYPE *b, LIBXS_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_sigmoid(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    const LIBXS_DNN_ELTWISE_FTYPE exp_value = (LIBXS_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_tanh(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_relu(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    const LIBXS_DNN_ELTWISE_FTYPE exp_value = (LIBXS_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    const LIBXS_DNN_ELTWISE_FTYPE sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_tanh_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    const LIBXS_DNN_ELTWISE_FTYPE tanh_value = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (LIBXS_DNN_ELTWISE_FTYPE)(src[i] > 0.0f ? 1.0f : 0.0f);
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_transpose(libxs_blasint rows, libxs_blasint cols, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  const libxs_blasint size = rows * cols;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, src2D, src, cols);
  LIBXS_VLA_DECL(2, LIBXS_DNN_ELTWISE_FTYPE, dst2D, dst, rows);
  libxs_blasint job;

  for (job = thr_begin; job < thr_end; ++job) {
    const libxs_blasint i = job / cols;
    const libxs_blasint j = job % cols;
    LIBXS_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXS_VLA_ACCESS(2, src2D, i, j, cols);
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_copy(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = src[i];
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_complement(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = 1 - src[i];
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_complement_square(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_inverse(libxs_blasint size, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, size);
  libxs_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = -src[i];
  }
}


LIBXS_API_INTERN void libxs_internal_matrix_1D_2D(libxs_blasint m, libxs_blasint n, libxs_blasint bm, libxs_blasint bn, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxs_blasint chunksize = (m % nthreads == 0) ? (m / nthreads) : (m / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxs_blasint thr_begin = (ltid * chunksize < m) ? (ltid * chunksize) : m;
  const libxs_blasint thr_end = LIBXS_MIN(ltid * chunksize + chunksize, m);
  libxs_blasint i, j;
  LIBXS_VLA_DECL(4, LIBXS_DNN_ELTWISE_FTYPE, real_dst, (LIBXS_DNN_ELTWISE_FTYPE*)dst, m/bm, bn, bm);

  for (i = thr_begin; i < thr_end; i++) {
    const libxs_blasint mb = i/bm;
    const libxs_blasint ibm = i%bm;
    for (j = 0; j < n; j++) {
      const libxs_blasint nb = j/bn;
      const libxs_blasint ibn = j%bn;
      LIBXS_VLA_ACCESS(4, real_dst, nb, mb, ibn, ibm, m/bm, bn, bm) = src[i];
    }
  }
}


/* #define LSTM_TIMING */
#if defined(LSTM_TIMING)
extern double Gbl_t_input_total, Gbl_t_recur_total, Gbl_t_eltwise_total, Gbl_t_nonlin_total;
extern unsigned long long Gbl_t_input, Gbl_t_recur, Gbl_t_eltwise, Gbl_t_nonlin;
extern double Gbl_duration_input, Gbl_duration_recur, Gbl_duration_eltwise, Gbl_duration_nonlin;
#endif

LIBXS_API_INTERN void libxs_internal_matrix_zero_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)0;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_copy_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_add_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] + src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sub_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] - src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *srcdst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] *= src0[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_eltwise_fma_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] += src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_add_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] += colv[i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, libxs_bfloat16 *colv) {
  libxs_blasint i, j;
  libxs_bfloat16_hp t;

  t.i[0] = 0;
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      t.i[1] = colv[i];
      srcdst[(j*ld)+i] = t.f;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_const_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv, LIBXS_DNN_ELTWISE_FTYPE const_bias) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i] + const_bias;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_bcst_cvt_bf16_fp32_colvector_const_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, libxs_bfloat16 *colv, LIBXS_DNN_ELTWISE_FTYPE const_bias) {
  libxs_blasint i, j;
  libxs_bfloat16_hp t;

  t.i[0] = 0;
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      t.i[1] = colv[i];
      srcdst[(j*ld)+i] = t.f + const_bias;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      const LIBXS_DNN_ELTWISE_FTYPE mid_value = (LIBXS_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 / ((LIBXS_DNN_ELTWISE_FTYPE)1 + mid_value);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_tanh_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_relu_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXS_DNN_ELTWISE_FTYPE)0 : src[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXS_DNN_ELTWISE_FTYPE exp_value = (LIBXS_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      LIBXS_DNN_ELTWISE_FTYPE mid_value = (LIBXS_DNN_ELTWISE_FTYPE)1 / ((LIBXS_DNN_ELTWISE_FTYPE)1 + exp_value);
      dst[(j*ld)+i] = ((LIBXS_DNN_ELTWISE_FTYPE)1 - mid_value) * mid_value;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_tanh_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXS_DNN_ELTWISE_FTYPE tanh_value = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXS_DNN_ELTWISE_FTYPE)0 : (LIBXS_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXS_DNN_ELTWISE_FTYPE exp_value = (LIBXS_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      LIBXS_DNN_ELTWISE_FTYPE mid_value = (LIBXS_DNN_ELTWISE_FTYPE)1 / ((LIBXS_DNN_ELTWISE_FTYPE)1 + exp_value);
      dst[(j*ld)+i] *= ((LIBXS_DNN_ELTWISE_FTYPE)1 - mid_value) * mid_value;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_tanh_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXS_DNN_ELTWISE_FTYPE tanh_value = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
      dst[(j*ld)+i] *= (LIBXS_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] *= (src[(j*ld)+i] < 0) ? (LIBXS_DNN_ELTWISE_FTYPE)0 : (LIBXS_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_complement_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 - src[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_complement_square_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 - (src[(j*ld)+i] * src[(j*ld)+i]);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_rne_mask_fp32_bfp16_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, float* src, float* dst) {
  libxs_blasint i,j;

  /* rnaz buffer to bfp16 */
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      unsigned int int_round = 0;
      unsigned int do_round = 1;
      const void *const ptr = &int_round;

      int_round = *((unsigned int*)&(src[(j*ld)+i]));

      /* we don't round NaN and inf */
      if ( (int_round & 0x7f800000) == 0x7f800000 ) {
        do_round = 0;
      }

      /* perform round nearest tie even */
      if ( do_round != 0 ) {
        unsigned int fixup = (int_round >> 16) & 1;
        int_round = int_round + 0x00007fff + fixup;
      }

      /* chop bits to create BFP16 in FP32 */
      int_round = int_round & 0xffff0000;

      dst[(j*ld)+i] = *((float*)ptr);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_rne_cvt_fp32_bfp16_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, float* src, libxs_bfloat16* dst) {
  libxs_blasint i,j;

  /* truncate buffer to bfp16 */
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      unsigned int int_round = 0;
      unsigned int do_round = 1;
      int_round = *((unsigned int*)&(src[(j*ld)+i]));
      /* we don't round NaN and inf */
      if ( (int_round & 0x7f800000) == 0x7f800000 ) {
        do_round = 0;
      }
      /* perform round nearest tie even */
      if ( do_round != 0 ) {
        unsigned int fixup = (int_round >> 16) & 1;
        int_round = int_round + 0x00007fff + fixup;
      }
      /* create the bfp16 value by shifting out the lower 16bits */
      int_round = int_round >> 16;
      dst[(j*ld)+i] = (unsigned short)int_round;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_cvt_bf16_fp32_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, libxs_bfloat16 *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;
  libxs_bfloat16_hp t;

  t.i[0] = 0;
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      t.i[1] = src[(j*ld)+i];
      dst[(j*ld)+i] = t.f;
    }
  }
}

