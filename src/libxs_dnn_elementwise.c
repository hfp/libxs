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
#include "libxs_dnn_elementwise.h"
#include "libxs_bgemm_types.h"

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

LIBXS_API_INTERN void libxs_internal_recursive_step(libxs_bgemm_handle* handle, LIBXS_DNN_ELTWISE_FTYPE* u, LIBXS_DNN_ELTWISE_FTYPE* h, LIBXS_DNN_ELTWISE_FTYPE* op1, LIBXS_DNN_ELTWISE_FTYPE *op2,
  LIBXS_DNN_ELTWISE_FTYPE *temp, LIBXS_DNN_ELTWISE_FTYPE *dst, int act, libxs_blasint size, int start_thread, int tid)
{
  const int ltid = tid - start_thread;
#if defined(LSTM_TIMING)
  if (ltid == 0) { Gbl_t_recur = libxs_timer_tick(); }
#endif
  libxs_bgemm_st(handle, u, h, op1, start_thread, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    Gbl_duration_recur = libxs_timer_duration(Gbl_t_recur, libxs_timer_tick());
    Gbl_t_recur_total += Gbl_duration_recur;
    Gbl_t_eltwise = libxs_timer_tick();
  }
#endif
  libxs_internal_matrix_add(size, op1, op2, temp, start_thread, ltid, handle->nthreads);
#if defined(LSTM_TIMING)
  libxs_barrier_wait(handle->barrier, ltid); /* Additional barrier introduced to measure time */
  if (ltid == 0) {
    Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
    Gbl_t_eltwise_total += Gbl_duration_eltwise;
    Gbl_t_nonlin = libxs_timer_tick();
  }
#endif
  switch (act) {
    case 0:
      /* do nothing */
      dst = temp;
      break;
    case 1:
      libxs_internal_matrix_relu(size, temp, dst, start_thread, tid, handle->nthreads);
      break;
    case 2:
      libxs_internal_matrix_sigmoid(size, temp, dst, start_thread, tid, handle->nthreads);
      break;
    case 3:
      libxs_internal_matrix_tanh(size, temp, dst, start_thread, tid, handle->nthreads);
      break;
    default:
      /* fprintf(stdout, "Unsupported activation function: %d\n", act); */
      dst = temp;
  }
#if defined(LSTM_TIMING)
  libxs_barrier_wait(handle->barrier, ltid); /* Additional barrier introduced to measure time */
  if (ltid == 0) {
    Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
    Gbl_t_nonlin_total += Gbl_duration_nonlin;
  }
#endif
}

LIBXS_API_INTERN void libxs_internal_matrix_zero_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)0;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_copy_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_add_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] + src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *srcdst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] *= src0[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_eltwise_fma_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src0, LIBXS_DNN_ELTWISE_FTYPE *src1, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] += src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_add_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] += colv[i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_bcst_colvector_const_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *srcdst, LIBXS_DNN_ELTWISE_FTYPE *colv, LIBXS_DNN_ELTWISE_FTYPE const_bias) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i] + const_bias;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      const LIBXS_DNN_ELTWISE_FTYPE mid_value = (LIBXS_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 / ((LIBXS_DNN_ELTWISE_FTYPE)1 + mid_value);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_tanh_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_relu_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXS_DNN_ELTWISE_FTYPE)0 : src[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

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
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     LIBXS_DNN_ELTWISE_FTYPE tanh_value = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
     dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXS_DNN_ELTWISE_FTYPE)0 : (LIBXS_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

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
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     LIBXS_DNN_ELTWISE_FTYPE tanh_value = (LIBXS_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
     dst[(j*ld)+i] *= (LIBXS_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_relu_inverse_inplace_eltwise_mult_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] *= (src[(j*ld)+i] < 0) ? (LIBXS_DNN_ELTWISE_FTYPE)0 : (LIBXS_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_complement_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 - src[(j*ld)+i];
    }
  }
}

LIBXS_API_INTERN void libxs_internal_matrix_complement_square_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *src, LIBXS_DNN_ELTWISE_FTYPE *dst) {
  libxs_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     dst[(j*ld)+i] = (LIBXS_DNN_ELTWISE_FTYPE)1 - (src[(j*ld)+i] * src[(j*ld)+i]);
    }
  }
}

LIBXS_API_INTERN void libxs_internal_compute_dcp_dci_di_df_dp_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, int timestep, int t, LIBXS_DNN_ELTWISE_FTYPE *dout, LIBXS_DNN_ELTWISE_FTYPE *dh, LIBXS_DNN_ELTWISE_FTYPE *o, LIBXS_DNN_ELTWISE_FTYPE *co, LIBXS_DNN_ELTWISE_FTYPE *dcs, LIBXS_DNN_ELTWISE_FTYPE *ii, LIBXS_DNN_ELTWISE_FTYPE *ci, LIBXS_DNN_ELTWISE_FTYPE *dci, LIBXS_DNN_ELTWISE_FTYPE *di, LIBXS_DNN_ELTWISE_FTYPE *cps, LIBXS_DNN_ELTWISE_FTYPE *f, LIBXS_DNN_ELTWISE_FTYPE *df, LIBXS_DNN_ELTWISE_FTYPE *dp, LIBXS_DNN_ELTWISE_FTYPE *dcp) {
#if defined(LIBXS_INTRINSICS_AVX512) 
  libxs_blasint i, j;
  const float neg_ones[16] = { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 };
  const float ones[16]  = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

  __m512 _dout, _dh, _o, _t1, _t2, _co, _dcs, _dcp, _ii, _ci, _dci, _di, _cps, _f, _df, _dp;
  __m512 _neg_ones = LIBXS_INTRINSICS_MM512_LOAD_PS( neg_ones );
  __m512 _ones = LIBXS_INTRINSICS_MM512_LOAD_PS( ones );

  if (timestep == t-1) {
    for ( j = 0; j < n; ++j ) {
      LIBXS_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXS_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _o = LIBXS_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXS_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcs = LIBXS_INTRINSICS_MM512_LOAD_PS( &dcs[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcs, _t1 );
        _ii = LIBXS_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXS_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXS_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        _cps = LIBXS_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );     
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXS_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXS_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXS_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  } else {
    for ( j = 0; j < n; ++j ) {
       LIBXS_PRAGMA_UNROLL_N(4)
       for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXS_INTRINSICS_MM512_LOAD_PS( &dout[(j*ld)+i] );
        _dh = LIBXS_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _dout = _mm512_add_ps( _dout, _dh );
        _o = LIBXS_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXS_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcp = LIBXS_INTRINSICS_MM512_LOAD_PS( &dcp[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcp, _t1 );
        _ii = LIBXS_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXS_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXS_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        _cps = LIBXS_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );     
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXS_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXS_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXS_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }  
  }
#else
LIBXS_UNUSED(m);LIBXS_UNUSED(n);LIBXS_UNUSED(ld);LIBXS_UNUSED(timestep);
LIBXS_UNUSED(t);LIBXS_UNUSED(dout);LIBXS_UNUSED(dh);LIBXS_UNUSED(o);
LIBXS_UNUSED(co);LIBXS_UNUSED(dcs);LIBXS_UNUSED(ii);LIBXS_UNUSED(ci);
LIBXS_UNUSED(dci);LIBXS_UNUSED(di);LIBXS_UNUSED(cps);LIBXS_UNUSED(f);
LIBXS_UNUSED(df);LIBXS_UNUSED(dp);LIBXS_UNUSED(dcp);
#endif
}

LIBXS_API_INTERN void libxs_internal_compute_o_cs_co_h_ld(libxs_blasint m, libxs_blasint n, libxs_blasint ld, LIBXS_DNN_ELTWISE_FTYPE *f, LIBXS_DNN_ELTWISE_FTYPE *cps, LIBXS_DNN_ELTWISE_FTYPE *cs, LIBXS_DNN_ELTWISE_FTYPE *ii, LIBXS_DNN_ELTWISE_FTYPE *ci,LIBXS_DNN_ELTWISE_FTYPE *co, LIBXS_DNN_ELTWISE_FTYPE *o, LIBXS_DNN_ELTWISE_FTYPE *h) {
#if defined(LIBXS_INTRINSICS_AVX512) 
  libxs_blasint i, j;
  __m512 _f, _cps, _cs, _ii, _ci, _co, _o, _h;
  const float halves[16]  = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
  __m512 _halves = LIBXS_INTRINSICS_MM512_LOAD_PS( halves );

  for ( j = 0; j < n; ++j ) {
    LIBXS_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _o = LIBXS_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _o = _mm512_fmadd_ps( _mm512_tanh_ps( _mm512_mul_ps( _o, _halves ) ), _halves, _halves);
        _f = LIBXS_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _cps = LIBXS_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _cs = _mm512_mul_ps( _f, _cps );
        _ii = LIBXS_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _ci = LIBXS_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _cs = _mm512_fmadd_ps( _ii, _ci, _cs );
        _co = _mm512_tanh_ps( _cs );
        _h = _mm512_mul_ps( _o, _co );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &h[(j*ld)+i], _h );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &cs[(j*ld)+i], _cs );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &co[(j*ld)+i], _co );
        LIBXS_INTRINSICS_MM512_STREAM_PS( &o[(j*ld)+i], _o );
      }
  }
#else
LIBXS_UNUSED(m);LIBXS_UNUSED(n);LIBXS_UNUSED(ld);LIBXS_UNUSED(f);
LIBXS_UNUSED(cps);LIBXS_UNUSED(cs);LIBXS_UNUSED(ii);LIBXS_UNUSED(ci);
LIBXS_UNUSED(co);LIBXS_UNUSED(o);LIBXS_UNUSED(h);
#endif
}
