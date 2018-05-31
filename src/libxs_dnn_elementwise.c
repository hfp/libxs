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

#include <libxs.h>
#include <math.h>

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(FTYPE)
# define FTYPE float /* TODO: undefine/remove generic symbol names as header-only interfers with user's code */
#endif


void libxs_internal_matinit(int seed, FTYPE *dst,
  libxs_blasint nrows, libxs_blasint ncols, libxs_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxs_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (FTYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (FTYPE)seed;
    }
  }
}


void libxs_internal_matrix_add(libxs_blasint size, FTYPE *a, FTYPE *b, FTYPE *c)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


void libxs_internal_matrix_eltwise_mult(libxs_blasint size, FTYPE *a, FTYPE *b, FTYPE *c)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


void libxs_internal_matrix_sigmoid(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
  FTYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (FTYPE)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


void libxs_internal_matrix_tanh(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (FTYPE)tanh((double)src[i]);
  }
}


void libxs_internal_matrix_relu(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


void libxs_internal_matrix_sigmoid_inverse(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
  FTYPE exp_value;
  FTYPE sig_exp;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (FTYPE)exp((double) -src[i]);
    sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


void libxs_internal_matrix_tanh_inverse(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
  FTYPE tanh_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    tanh_value = (FTYPE)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


void libxs_internal_matrix_relu_inverse(libxs_blasint size, FTYPE *src, FTYPE *dst, FTYPE *input)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


void libxs_internal_matrix_transpose(libxs_blasint rows, libxs_blasint cols, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i, j;
  LIBXS_VLA_DECL(2, FTYPE, src2D, src, cols);
  LIBXS_VLA_DECL(2, FTYPE, dst2D, dst, rows);
#if defined(_OPENMP)
# pragma omp parallel for private(i, j) LIBXS_OPENMP_COLLAPSE(2)
#endif
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      LIBXS_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXS_VLA_ACCESS(2, src2D, i, j, cols);
    }
  }
}


void libxs_internal_matrix_copy(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}


void libxs_internal_matrix_complement(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - src[i];
  }
}


void libxs_internal_matrix_complement_square(libxs_blasint size, FTYPE *src, FTYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


void libxs_internal_recursive_step(libxs_bgemm_handle* handle, FTYPE* u, FTYPE* h, FTYPE* op1, FTYPE *op2,
  FTYPE *temp, FTYPE *dst, int act, libxs_blasint size, int tid, int nthreads)
{
#if defined(LSTM_TIMING)
  Gbl_t_recur = libxs_timer_tick();
#endif
  libxs_bgemm(handle, u, h, op1, tid, nthreads);
#if defined(LSTM_TIMING)
  Gbl_duration_recur = libxs_timer_duration(Gbl_t_recur, libxs_timer_tick());
  Gbl_t_recur_total += Gbl_duration_recur;
  Gbl_t_eltwise = libxs_timer_tick();
#endif
  libxs_internal_matrix_add(size, op1, op2, temp);
#if defined(LSTM_TIMING)
  Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
  Gbl_t_eltwise_total += Gbl_duration_eltwise;
  Gbl_t_nonlin = libxs_timer_tick();
#endif
  switch (act) {
    case 0:
      /* do nothing -- this is required for the last time step */
      dst = temp;
      break;
    case 1:
      libxs_internal_matrix_relu(size, temp, dst);
      break;
    case 2:
      libxs_internal_matrix_sigmoid(size, temp, dst);
      break;
    case 3:
      libxs_internal_matrix_tanh(size, temp, dst);
      break;
    default:
      /* fprintf(stdout, "Unsupported activation function: %d\n", act); */
      dst = temp;
  }
#if defined(LSTM_TIMING)
  Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
  Gbl_t_nonlin_total += Gbl_duration_nonlin;
#endif
}

