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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxs.h>
#include </nfs_home/kunalban/libxs_egeor/src/libxs_main.h>
#include <math.h>

#define CHKERR_LIBXS_DNN(A) if ( A != LIBXS_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxs_dnn_get_error(A) );

#define LSTM_TIMING

/* #define NON_FUSED_INPUT_GEMM */
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif

#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS evailable */ \
  (LIBXS_EQUAL(REAL_TYPE, float) || LIBXS_EQUAL(REAL_TYPE, double))
# define CHECK
#endif

#if defined(LSTM_TIMING)
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif

struct rnn_handle {
  libxs_blasint m;
  libxs_blasint n;
  libxs_blasint k;
  libxs_blasint t;
  libxs_dnn_tensor *w;
  libxs_dnn_tensor *xt;
  libxs_dnn_tensor *u;
  libxs_dnn_tensor *h;
  libxs_dnn_tensor *z1t;
  libxs_dnn_tensor *z2;
  libxs_dnn_tensor *z;
  libxs_bgemm_handle *handlewx;
  libxs_bgemm_handle *handleuh;
  libxs_bgemm_handle *handlett;
};

struct lstm_handle {
  libxs_blasint m;
  libxs_blasint n;
  libxs_blasint k;
  libxs_blasint t;
  libxs_dnn_tensor *wi;
  libxs_dnn_tensor *wf;
  libxs_dnn_tensor *wo;
  libxs_dnn_tensor *wc;
  libxs_dnn_tensor *xt;
  libxs_dnn_tensor *ri;
  libxs_dnn_tensor *rf;
  libxs_dnn_tensor *ro;
  libxs_dnn_tensor *rc;
  libxs_dnn_tensor *h;
  libxs_dnn_tensor *i1t;
  libxs_dnn_tensor *i2;
  libxs_dnn_tensor *f1t;
  libxs_dnn_tensor *f2;
  libxs_dnn_tensor *o1t;
  libxs_dnn_tensor *o2;
  libxs_dnn_tensor *c1t;
  libxs_dnn_tensor *c2;
  libxs_dnn_tensor *i;
  libxs_dnn_tensor *f;
  libxs_dnn_tensor *o;
  libxs_dnn_tensor *c;
  libxs_dnn_tensor *d0;
  libxs_dnn_tensor *d1;
  libxs_dnn_tensor *d2;
  libxs_dnn_tensor *d;
  libxs_bgemm_handle *handlewx;
  libxs_bgemm_handle *handleuh;
  libxs_bgemm_handle *handlett;
};


LIBXS_INLINE LIBXS_RETARGETABLE void init(int seed, REAL_TYPE *LIBXS_RESTRICT dst,
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
      dst[k] = (REAL_TYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxs_blasint k = i * ld + j;
      dst[k] = (REAL_TYPE)seed;
    }
  }
}


void matrix_add(libxs_blasint size, REAL_TYPE *a, REAL_TYPE *b, REAL_TYPE *c)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  /*LIBXS_PRAGMA_SIMD*/
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


void matrix_eltwise_mult(libxs_blasint size, REAL_TYPE *a, REAL_TYPE *b, REAL_TYPE *c)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  /*LIBXS_PRAGMA_SIMD*/
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


void matrix_sigmoid(libxs_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxs_blasint i;
  REAL_TYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  /*LIBXS_PRAGMA_SIMD*/
  for (i = 0; i < size; i++) {
    exp_value = (REAL_TYPE)exp( -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


void matrix_tanh(libxs_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  /*LIBXS_PRAGMA_SIMD*/
  for (i = 0; i < size; i++) {
    dst[i] = tanh(src[i]);
  }
}


void matrix_relu(libxs_blasint size, REAL_TYPE *src, REAL_TYPE *dst)
{
  libxs_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  /*LIBXS_PRAGMA_SIMD*/
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : -src[i];
  }
}


void recursive_step(libxs_bgemm_handle* handle, REAL_TYPE* u, REAL_TYPE* h, REAL_TYPE* op1, REAL_TYPE *op2,
  REAL_TYPE *temp, REAL_TYPE *dst, int act, libxs_blasint size)
{
#if defined(LSTM_TIMING)
  Gbl_t_recur = libxs_timer_tick();
#endif
  libxs_bgemm_omp(handle, u, h, op1, 1);
#if defined(LSTM_TIMING)
  Gbl_duration_recur = libxs_timer_duration(Gbl_t_recur, libxs_timer_tick());
  Gbl_t_recur_total += Gbl_duration_recur;
  Gbl_t_eltwise = libxs_timer_tick();
#endif
  matrix_add(size, op1, op2, temp);
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
      matrix_relu(size, temp, dst);
      break;
    case 2:
      matrix_sigmoid(size, temp, dst);
      break;
    case 3:
      matrix_tanh(size, temp, dst);
      break;
    default:
      fprintf(stdout, "Unsupported activation function: %d\n", act);
      dst = temp;
  }
#if defined(LSTM_TIMING)
  Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
  Gbl_t_nonlin_total += Gbl_duration_nonlin;
#endif
}


libxs_dnn_tensor* libxs_create_dnn_tensor_rnn( REAL_TYPE *data )
{
  libxs_dnn_tensor* tensor;
  tensor = (libxs_dnn_tensor*)malloc(sizeof(libxs_dnn_tensor));
  tensor->layout = NULL;
  tensor->data = (void*)data;
  tensor->scf = 1;
  return tensor;
}


void rnn_init(struct rnn_handle *rnn, REAL_TYPE *wgold, REAL_TYPE *xgoldt, REAL_TYPE *ugold,
  REAL_TYPE *hgold, REAL_TYPE *z1gold, REAL_TYPE *z2gold, REAL_TYPE *zgold,
  const libxs_blasint ldw, const libxs_blasint ldx, const libxs_blasint ldz,
  const libxs_blasint ldu, const libxs_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;
  libxs_blasint m = rnn->m;
  libxs_blasint n = rnn->n;
  libxs_blasint k = rnn->k;
  libxs_blasint t = rnn->t;
  REAL_TYPE *w = (REAL_TYPE*)rnn->w->data;
  REAL_TYPE *xt = (REAL_TYPE*)rnn->xt->data;
  REAL_TYPE *u = (REAL_TYPE*)rnn->u->data;
  REAL_TYPE *h = (REAL_TYPE*)rnn->h->data;
  REAL_TYPE *z1t = (REAL_TYPE*)rnn->z1t->data;
  REAL_TYPE *z2 = (REAL_TYPE*)rnn->z2->data;
  REAL_TYPE *z = (REAL_TYPE*)rnn->z->data;
  libxs_bgemm_handle *handlewx = rnn->handlewx;
  libxs_bgemm_handle *handleuh = rnn->handleuh;
  LIBXS_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, z1, z1t, m * n);

  init(42, wgold, m, k, ldw, 1.0);
  int it;
  for (it = 0; it < t; ++it) {
    init(24, &LIBXS_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
  }
  init(42, ugold, m, m, ldu, 1.0);
  init(24, hgold, m, n, ldh, 1.0);
  init( 0, z1gold, m, n, ldz, 1.0);
  init( 0, z2gold, m, n, ldz, 1.0);
  init( 0, zgold, m, n, ldz, 1.0);
  libxs_bgemm_copyin_a(handlewx, wgold, &ldw, w);
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_b(handlewx, &LIBXS_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXS_VLA_ACCESS(2, x, it, 0, k * n));
  }
  libxs_bgemm_copyin_a(handleuh, ugold, &ldu, u);
  libxs_bgemm_copyin_b(handleuh, hgold, &ldh, h);
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_c(handlewx, z1gold, &ldz, &LIBXS_VLA_ACCESS(2, z1, it, 0, m * n));
  }
  libxs_bgemm_copyin_c(handleuh, z2gold, &ldz, z2);
  libxs_bgemm_copyin_c(handlewx, zgold, &ldz, z);
#if defined(MKL_ENABLE_AVX512)
  mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
  /* warmup OpenMP (populate thread pool) */
  libxs_bgemm_omp(handlewx, w, x, &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), 1);
#if defined(CHECK)
  if (!LIBXS_FEQ(0, check)) {
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, z1gold, &ldz);
  }
#endif
  libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(REAL_TYPE),
    &transa, &transb, &m, &n, &k, &alpha, w, &ldw, x, &ldx, &beta, &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), &ldz);
  fprintf(stdout, "\n\n");
  /* warmup OpenMP (populate thread pool) */
  libxs_bgemm_omp(handleuh, u, h, z2, 1);
#if defined(CHECK)
  if (!LIBXS_FEQ(0, check)) {
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
  }
#endif
  libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(REAL_TYPE),
    &transa, &transb, &m, &n, &m, &alpha, u, &ldu, h, &ldh, &beta, z2, &ldz);
  fprintf(stdout, "\n\n");
}


void rnn_execute(struct rnn_handle *rnn, const int nrepeat)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;
  libxs_blasint m = rnn->m;
  libxs_blasint n = rnn->n;
  libxs_blasint k = rnn->k;
  libxs_blasint t = rnn->t;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
  REAL_TYPE *w = (REAL_TYPE*)rnn->w->data;
  REAL_TYPE *xt = (REAL_TYPE*)rnn->xt->data;
  REAL_TYPE *u = (REAL_TYPE*)rnn->u->data;
  REAL_TYPE *h = (REAL_TYPE*)rnn->h->data;
  REAL_TYPE *z1t = (REAL_TYPE*)rnn->z1t->data;
  REAL_TYPE *z2 = (REAL_TYPE*)rnn->z2->data;
  REAL_TYPE *z = (REAL_TYPE*)rnn->z->data;
  libxs_bgemm_handle *handlewx = rnn->handlewx;
  libxs_bgemm_handle *handleuh = rnn->handleuh;
  libxs_bgemm_handle *handlett = rnn->handlett;
  LIBXS_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, z1, z1t, m * n);
  unsigned long long start;
  double duration;
#if defined(LSTM_TIMING)
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  int s;
  int i;
  libxs_blasint nt = n*t;
  start = libxs_timer_tick();
  for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
    Gbl_t_input = libxs_timer_tick();
#endif
    /* The following loop may be absorbed into libxs_lstm_omp */
    libxs_bgemm_omp(handlett, w, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, z1, 0, 0, m * n), 1/*nrepeat*/);
    /*LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &nt, &k, &alpha, w, m, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), k, &beta, z1, m);*/
#if defined(LSTM_TIMING)
    Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
    Gbl_t_input_total += Gbl_duration_input;
#endif
    for (i = 0; i < t-1; ++i) {
      recursive_step(handleuh, u, h, z2, &LIBXS_VLA_ACCESS(2, z1, i, 0, m * n), z, h, 1, m * n); /*sigmoid*/
    }
    recursive_step(handleuh, u, h, z2, &LIBXS_VLA_ACCESS(2, z1, t-1, 0, m * n), z, z, 0, m * n); /*nop*/
  }
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#if defined(LSTM_TIMING)
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif
}


void rnn_destroy(struct rnn_handle *rnn)
{
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->w ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->xt ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->u ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->h ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->z1t ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->z2 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( rnn->z ) );
}


int rnn (const libxs_blasint m, const libxs_blasint n, const libxs_blasint k, const libxs_blasint t,
         const libxs_blasint bm, const libxs_blasint bn, const libxs_blasint bk, const libxs_bgemm_order order, const int nrepeat,
         const libxs_blasint b_m1, const libxs_blasint b_n1, const libxs_blasint b_k1, const libxs_blasint b_k2, const libxs_blasint b_m2,
         const libxs_blasint ldw, const libxs_blasint ldx, const libxs_blasint ldz, const libxs_blasint ldu, const libxs_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload target(LIBXS_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE* wgold = (REAL_TYPE*)libxs_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* xgoldt = (REAL_TYPE*)libxs_malloc(ldx * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* ugold = (REAL_TYPE*)libxs_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* hgold = (REAL_TYPE*)libxs_malloc(ldh * n * sizeof(REAL_TYPE));
    REAL_TYPE* z1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* z2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* zgold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));

    REAL_TYPE* w = (REAL_TYPE*)libxs_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* xt = (REAL_TYPE*)libxs_malloc(k * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* u = (REAL_TYPE*)libxs_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* h = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* z1t = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* z2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* z = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    LIBXS_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
    libxs_bgemm_handle* handlewx = 0;
    libxs_bgemm_handle* handleuh = 0;
    libxs_bgemm_handle* handlett = 0;
    const libxs_gemm_prefetch_type strategy = LIBXS_PREFETCH_AUTO;
    handlewx = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);

    struct rnn_handle rnn;
    rnn.m = m;
    rnn.n = n;
    rnn.k = k;
    rnn.t = t;
    rnn.w = libxs_create_dnn_tensor_rnn( w );
    rnn.xt = libxs_create_dnn_tensor_rnn( xt );
    rnn.u = libxs_create_dnn_tensor_rnn( u );
    rnn.h = libxs_create_dnn_tensor_rnn( h );
    rnn.z1t = libxs_create_dnn_tensor_rnn( z1t );
    rnn.z2 = libxs_create_dnn_tensor_rnn( z2 );
    rnn.z = libxs_create_dnn_tensor_rnn( z );
    rnn.handlewx = handlewx;
    rnn.handleuh = handleuh;
    rnn.handlett = handlett;

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      rnn_init(&rnn, wgold, xgoldt, ugold, hgold, z1gold, z2gold, zgold, ldw, ldx, ldz, ldu, ldh);
      rnn_execute(&rnn, nrepeat);
#if defined(CHECK)
      unsigned long long start;
      double duration;
      int s;
      if (!LIBXS_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        REAL_TYPE* ztest = 0;
        int i;
        start = libxs_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (i = 0; i < t-1; ++i) {
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, i, 0, k * n), &ldx, &beta, z1gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
            matrix_add(m*n, z1gold, z2gold, zgold);
            matrix_relu(m*n, zgold, hgold); /*sigmoid*/
          }
          LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, t-1, 0, k * n), &ldx, &beta, z1gold, &ldz);
          LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
          matrix_add(m*n, z1gold, z2gold, zgold);
        }
        duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxs_free(wgold); wgold = 0;
        libxs_free(xgoldt); xgoldt = 0;
        libxs_free(ugold); ugold = 0;
        libxs_free(hgold); hgold = 0;
        libxs_free(z1gold); z1gold = 0;
        libxs_free(z2gold); z2gold = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        ztest = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
        if (0 != ztest) {
          libxs_matdiff_info diff;
          libxs_bgemm_copyout_c(handleuh, z, &ldz, ztest);
          if (EXIT_SUCCESS == libxs_matdiff(LIBXS_DATATYPE(REAL_TYPE), m, n, zgold, ztest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxs_free(ztest);
        }
      }
#endif
      libxs_bgemm_handle_destroy(handlewx);
      libxs_bgemm_handle_destroy(handleuh);
      libxs_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXS_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxs_free(wgold);
    libxs_free(xgoldt);
    libxs_free(ugold);
    libxs_free(hgold);
    libxs_free(z1gold);
    libxs_free(z2gold);
    libxs_free(zgold);
    rnn_destroy(&rnn);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


void lstm_init(struct lstm_handle *lstm, REAL_TYPE *wigold, REAL_TYPE *wfgold, REAL_TYPE *wogold, REAL_TYPE *wcgold,
  REAL_TYPE *xgoldt, REAL_TYPE *rigold, REAL_TYPE *rfgold, REAL_TYPE *rogold, REAL_TYPE *rcgold, REAL_TYPE *hgold,
  REAL_TYPE *i1gold, REAL_TYPE *i2gold, REAL_TYPE *f1gold, REAL_TYPE *f2gold, REAL_TYPE *o1gold, REAL_TYPE *o2gold,
  REAL_TYPE *c1gold, REAL_TYPE *c2gold, REAL_TYPE *igold, REAL_TYPE *fgold, REAL_TYPE *ogold, REAL_TYPE *cgold,
  REAL_TYPE *d0gold, REAL_TYPE *d1gold, REAL_TYPE *d2gold, REAL_TYPE *dgold,
  const libxs_blasint ldw, const libxs_blasint ldx, const libxs_blasint ldz,
  const libxs_blasint ldu, const libxs_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  libxs_blasint k = lstm->k;
  libxs_blasint t = lstm->t;
  REAL_TYPE *wi = (REAL_TYPE*)lstm->wi->data;
  REAL_TYPE *wf = (REAL_TYPE*)lstm->wf->data;
  REAL_TYPE *wo = (REAL_TYPE*)lstm->wo->data;
  REAL_TYPE *wc = (REAL_TYPE*)lstm->wc->data;
  REAL_TYPE *xt = (REAL_TYPE*)lstm->xt->data;
  REAL_TYPE *ri = (REAL_TYPE*)lstm->ri->data;
  REAL_TYPE *rf = (REAL_TYPE*)lstm->rf->data;
  REAL_TYPE *ro = (REAL_TYPE*)lstm->ro->data;
  REAL_TYPE *rc = (REAL_TYPE*)lstm->rc->data;
  REAL_TYPE *h = (REAL_TYPE*)lstm->h->data;
  REAL_TYPE *i1t = (REAL_TYPE*)lstm->i1t->data;
  REAL_TYPE *i2 = (REAL_TYPE*)lstm->i2->data;
  REAL_TYPE *f1t = (REAL_TYPE*)lstm->f1t->data;
  REAL_TYPE *f2 = (REAL_TYPE*)lstm->f2->data;
  REAL_TYPE *o1t = (REAL_TYPE*)lstm->o1t->data;
  REAL_TYPE *o2 = (REAL_TYPE*)lstm->o2->data;
  REAL_TYPE *c1t = (REAL_TYPE*)lstm->c1t->data;
  REAL_TYPE *c2 = (REAL_TYPE*)lstm->c2->data;
  REAL_TYPE *i = (REAL_TYPE*)lstm->i->data;
  REAL_TYPE *f = (REAL_TYPE*)lstm->f->data;
  REAL_TYPE *o = (REAL_TYPE*)lstm->o->data;
  REAL_TYPE *c = (REAL_TYPE*)lstm->c->data;
  REAL_TYPE *d0 = (REAL_TYPE*)lstm->d0->data;
  REAL_TYPE *d1 = (REAL_TYPE*)lstm->d1->data;
  REAL_TYPE *d2 = (REAL_TYPE*)lstm->d2->data;
  REAL_TYPE *d = (REAL_TYPE*)lstm->d->data;
  LIBXS_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXS_VLA_DECL(2, REAL_TYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, c1, c1t, m * n);
#else
  LIBXS_VLA_DECL(3, REAL_TYPE, i1, i1t, t, m * n);
#endif
  libxs_bgemm_handle *handlewx = lstm->handlewx;
  libxs_bgemm_handle *handleuh = lstm->handleuh;

  init(42, wigold, m, k, ldw, 1.0);
  init(42, wfgold, m, k, ldw, 1.0);
  init(42, wogold, m, k, ldw, 1.0);
  init(42, wcgold, m, k, ldw, 1.0);
  int it;
  for (it = 0; it < t; ++it) {
    init(24, &LIBXS_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
  }
  init(42, rigold, m, m, ldu, 1.0);
  init(42, rfgold, m, m, ldu, 1.0);
  init(42, rogold, m, m, ldu, 1.0);
  init(42, rcgold, m, m, ldu, 1.0);
  init(24, hgold, m, n, ldh, 1.0);
  init(24, d0gold, m, n, ldh, 1.0);
  init( 0, i1gold, m, n, ldz, 1.0);
  init( 0, i2gold, m, n, ldz, 1.0);
  init( 0, f1gold, m, n, ldz, 1.0);
  init( 0, f2gold, m, n, ldz, 1.0);
  init( 0, o1gold, m, n, ldz, 1.0);
  init( 0, o2gold, m, n, ldz, 1.0);
  init( 0, c1gold, m, n, ldz, 1.0);
  init( 0, c2gold, m, n, ldz, 1.0);
  init( 0, igold, m, n, ldz, 1.0);
  init( 0, fgold, m, n, ldz, 1.0);
  init( 0, ogold, m, n, ldz, 1.0);
  init( 0, cgold, m, n, ldz, 1.0);
  init( 0, d1gold, m, n, ldz, 1.0);
  init( 0, d2gold, m, n, ldz, 1.0);
  init( 0, dgold, m, n, ldz, 1.0);
#if defined(NON_FUSED_INPUT_GEMM)
  libxs_bgemm_copyin_a(handlewx, wigold, &ldw, wi);
  libxs_bgemm_copyin_a(handlewx, wfgold, &ldw, wf);
  libxs_bgemm_copyin_a(handlewx, wogold, &ldw, wo);
  libxs_bgemm_copyin_a(handlewx, wcgold, &ldw, wc);
#else
  LIBXS_VLA_DECL(2, REAL_TYPE, wi4, wi, m * k);
  libxs_bgemm_copyin_a(handlewx, wigold, &ldw, &LIBXS_VLA_ACCESS(2, wi4, 0, 0, m * k));
  libxs_bgemm_copyin_a(handlewx, wfgold, &ldw, &LIBXS_VLA_ACCESS(2, wi4, 1, 0, m * k));
  libxs_bgemm_copyin_a(handlewx, wogold, &ldw, &LIBXS_VLA_ACCESS(2, wi4, 2, 0, m * k));
  libxs_bgemm_copyin_a(handlewx, wcgold, &ldw, &LIBXS_VLA_ACCESS(2, wi4, 3, 0, m * k));
#endif
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_b(handlewx, &LIBXS_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXS_VLA_ACCESS(2, x, it, 0, k * n));
  }
  libxs_bgemm_copyin_a(handleuh, rigold, &ldu, ri);
  libxs_bgemm_copyin_a(handleuh, rfgold, &ldu, rf);
  libxs_bgemm_copyin_a(handleuh, rogold, &ldu, ro);
  libxs_bgemm_copyin_a(handleuh, rcgold, &ldu, rc);
  libxs_bgemm_copyin_b(handleuh, hgold, &ldh, h);
#if defined(NON_FUSED_INPUT_GEMM)
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_c(handlewx, i1gold, &ldz, &LIBXS_VLA_ACCESS(2, i1, it, 0, m * n));
    libxs_bgemm_copyin_c(handlewx, f1gold, &ldz, &LIBXS_VLA_ACCESS(2, f1, it, 0, m * n));
    libxs_bgemm_copyin_c(handlewx, o1gold, &ldz, &LIBXS_VLA_ACCESS(2, o1, it, 0, m * n));
    libxs_bgemm_copyin_c(handlewx, c1gold, &ldz, &LIBXS_VLA_ACCESS(2, c1, it, 0, m * n));
  }
#else
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_c(handlewx, i1gold, &ldz, &LIBXS_VLA_ACCESS(3, i1, 0, it, 0, t, m * n));
  }
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_c(handlewx, f1gold, &ldz, &LIBXS_VLA_ACCESS(3, i1, 1, it, 0, t, m * n));
  }
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_c(handlewx, o1gold, &ldz, &LIBXS_VLA_ACCESS(3, i1, 2, it, 0, t, m * n));
  }
  for (it = 0; it < t; ++it) {
    libxs_bgemm_copyin_c(handlewx, c1gold, &ldz, &LIBXS_VLA_ACCESS(3, i1, 3, it, 0, t, m * n));
  }
#endif
  libxs_bgemm_copyin_c(handleuh, igold, &ldh, i);
  libxs_bgemm_copyin_c(handleuh, fgold, &ldh, f);
  libxs_bgemm_copyin_c(handleuh, ogold, &ldh, o);
  libxs_bgemm_copyin_c(handleuh, cgold, &ldh, c);
  libxs_bgemm_copyin_c(handleuh, d1gold, &ldh, d1);
  libxs_bgemm_copyin_c(handleuh, d2gold, &ldh, d2);
  libxs_bgemm_copyin_c(handleuh, d0gold, &ldh, d0);
  libxs_bgemm_copyin_c(handleuh, dgold, &ldh, d);
#if defined(MKL_ENABLE_AVX512)
  mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
  /* warmup OpenMP (populate thread pool) */
#if defined(NON_FUSED_INPUT_GEMM)
  libxs_bgemm_omp(handlewx, wi, x, &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), 1);
  libxs_bgemm_omp(handlewx, wf, x, &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), 1);
  libxs_bgemm_omp(handlewx, wo, x, &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), 1);
  libxs_bgemm_omp(handlewx, wc, x, &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), 1);
#else
  libxs_bgemm_omp(handlewx, wi, x, &LIBXS_VLA_ACCESS(3, i1, 0, 0, 0, t, m * n), 1);
#endif
#if defined(CHECK)
  if (!LIBXS_FEQ(0, check)) {
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wigold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, i1gold, &ldz);
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, f1gold, &ldz);
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wogold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, o1gold, &ldz);
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, c1gold, &ldz);
  }
#endif
  libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(REAL_TYPE),
    &transa, &transb, &m, &n, &k, &alpha, wi, &ldw, x, &ldx, &beta, &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), &ldz);
  fprintf(stdout, "\n\n");
  /* warmup OpenMP (populate thread pool) */
  libxs_bgemm_omp(handleuh, ri, h, i2, 1);
  libxs_bgemm_omp(handleuh, rf, h, f2, 1);
  libxs_bgemm_omp(handleuh, ro, h, o2, 1);
  libxs_bgemm_omp(handleuh, rc, h, c2, 1);
#if defined(CHECK)
  if (!LIBXS_FEQ(0, check)) {
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rigold, &ldu, hgold, &ldh, &beta, i2gold, &ldz);
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &ldu, hgold, &ldh, &beta, f2gold, &ldz);
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rogold, &ldu, hgold, &ldh, &beta, o2gold, &ldz);
    LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &ldu, hgold, &ldh, &beta, c2gold, &ldz);
  }
#endif
  libxs_gemm_print(stdout, LIBXS_GEMM_PRECISION(REAL_TYPE),
    &transa, &transb, &m, &n, &m, &alpha, ri, &ldu, h, &ldh, &beta, i2, &ldz);
  fprintf(stdout, "\n\n");
}


void lstm_execute(struct lstm_handle *lstm, const int nrepeat)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;
  libxs_blasint m = lstm->m;
  libxs_blasint n = lstm->n;
  libxs_blasint k = lstm->k;
  libxs_blasint t = lstm->t;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (4.0 * m * n)) * t * 1E-9;
  REAL_TYPE *wi = (REAL_TYPE*)lstm->wi->data;
  REAL_TYPE *wf = (REAL_TYPE*)lstm->wf->data;
  REAL_TYPE *wo = (REAL_TYPE*)lstm->wo->data;
  REAL_TYPE *wc = (REAL_TYPE*)lstm->wc->data;
  REAL_TYPE *xt = (REAL_TYPE*)lstm->xt->data;
  REAL_TYPE *ri = (REAL_TYPE*)lstm->ri->data;
  REAL_TYPE *rf = (REAL_TYPE*)lstm->rf->data;
  REAL_TYPE *ro = (REAL_TYPE*)lstm->ro->data;
  REAL_TYPE *rc = (REAL_TYPE*)lstm->rc->data;
  REAL_TYPE *h = (REAL_TYPE*)lstm->h->data;
  REAL_TYPE *i1t = (REAL_TYPE*)lstm->i1t->data;
  REAL_TYPE *i2 = (REAL_TYPE*)lstm->i2->data;
  REAL_TYPE *f1t = (REAL_TYPE*)lstm->f1t->data;
  REAL_TYPE *f2 = (REAL_TYPE*)lstm->f2->data;
  REAL_TYPE *o1t = (REAL_TYPE*)lstm->o1t->data;
  REAL_TYPE *o2 = (REAL_TYPE*)lstm->o2->data;
  REAL_TYPE *c1t = (REAL_TYPE*)lstm->c1t->data;
  REAL_TYPE *c2 = (REAL_TYPE*)lstm->c2->data;
  REAL_TYPE *i = (REAL_TYPE*)lstm->i->data;
  REAL_TYPE *f = (REAL_TYPE*)lstm->f->data;
  REAL_TYPE *o = (REAL_TYPE*)lstm->o->data;
  REAL_TYPE *c = (REAL_TYPE*)lstm->c->data;
  REAL_TYPE *d0 = (REAL_TYPE*)lstm->d0->data;
  REAL_TYPE *d1 = (REAL_TYPE*)lstm->d1->data;
  REAL_TYPE *d2 = (REAL_TYPE*)lstm->d2->data;
  REAL_TYPE *d = (REAL_TYPE*)lstm->d->data;
  LIBXS_VLA_DECL(2, REAL_TYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXS_VLA_DECL(2, REAL_TYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, c1, c1t, m * n);
#else
  LIBXS_VLA_DECL(3, REAL_TYPE, i4, i1t, t, m * n);
  i1t = &LIBXS_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n);
  f1t = &LIBXS_VLA_ACCESS(3, i4, 1, 0, 0, t, m * n);
  o1t = &LIBXS_VLA_ACCESS(3, i4, 2, 0, 0, t, m * n);
  c1t = &LIBXS_VLA_ACCESS(3, i4, 3, 0, 0, t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, i1, i1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, f1, f1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, o1, o1t, m * n);
  LIBXS_VLA_DECL(2, REAL_TYPE, c1, c1t, m * n);
#endif
  libxs_bgemm_handle *handlewx = lstm->handlewx;
  libxs_bgemm_handle *handleuh = lstm->handleuh;
  libxs_bgemm_handle *handlett = lstm->handlett;
  unsigned long long start;
  double duration;
#if defined(LSTM_TIMING)
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  int s;
  int j;
  libxs_blasint nt = n*t;
  start = libxs_timer_tick();
  for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
    Gbl_t_input = libxs_timer_tick();
#endif
    /* The following loop may be absorbed into libxs_lstm_omp */
#if defined(NON_FUSED_INPUT_GEMM)
    libxs_bgemm_omp(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), 1/*nrepeat*/);
    libxs_bgemm_omp(handlett, wf, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), 1/*nrepeat*/);
    libxs_bgemm_omp(handlett, wo, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), 1/*nrepeat*/);
    libxs_bgemm_omp(handlett, wc, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), 1/*nrepeat*/);
#else
    libxs_bgemm_omp(handlett, wi, &LIBXS_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXS_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), 1/*nrepeat*/);
#endif
#if defined(LSTM_TIMING)
    Gbl_duration_input = libxs_timer_duration(Gbl_t_input, libxs_timer_tick());
    Gbl_t_input_total += Gbl_duration_input;
#endif
    recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, 0, 0, m * n), i, i, 1, m * n); /*sigmoid*/
    recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, 0, 0, m * n), f, f, 1, m * n); /*sigmoid*/
    recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, 0, 0, m * n), o, o, 1, m * n); /*sigmoid*/
    recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, 0, 0, m * n), c, c, 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
    Gbl_t_eltwise = libxs_timer_tick();
#endif
    matrix_eltwise_mult(m*n, f, d0, d1);
    matrix_eltwise_mult(m*n, i, c, d2);
    matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
    Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
    Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    if (t > 1) {
#if defined(LSTM_TIMING)
      Gbl_t_nonlin = libxs_timer_tick();
#endif
      matrix_relu(m*n, d, d); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
      Gbl_t_nonlin_total += Gbl_duration_nonlin;
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      matrix_eltwise_mult(m*n, o, d, h);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
    for (j = 1; j < t-1; ++j) {
      recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      matrix_eltwise_mult(m*n, f, d, d1);
      matrix_eltwise_mult(m*n, i, c, d2);
      matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
      Gbl_t_nonlin = libxs_timer_tick();
#endif
      matrix_relu(m*n, d, d); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_duration_nonlin = libxs_timer_duration(Gbl_t_nonlin, libxs_timer_tick());
      Gbl_t_nonlin_total += Gbl_duration_nonlin;
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      matrix_eltwise_mult(m*n, o, d, h);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
    if (t > 1) {
      recursive_step(handleuh, ri, h, i2, &LIBXS_VLA_ACCESS(2, i1, t-1, 0, m * n), i, i, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rf, h, f2, &LIBXS_VLA_ACCESS(2, f1, t-1, 0, m * n), f, f, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, ro, h, o2, &LIBXS_VLA_ACCESS(2, o1, t-1, 0, m * n), o, o, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rc, h, c2, &LIBXS_VLA_ACCESS(2, c1, t-1, 0, m * n), c, c, 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxs_timer_tick();
#endif
      matrix_eltwise_mult(m*n, f, d, d1);
      matrix_eltwise_mult(m*n, i, c, d2);
      matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxs_timer_duration(Gbl_t_eltwise, libxs_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
  }
  duration = libxs_timer_duration(start, libxs_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#if defined(LSTM_TIMING)
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif
}


void lstm_destroy(struct lstm_handle *lstm)
{
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->wi ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->wf ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->wo ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->wc ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->xt ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->ri ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->rf ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->ro ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->rc ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->h ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->i1t ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->i2 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->f1t ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->f2 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->o1t ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->o2 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->c1t ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->c2 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->i ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->f ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->o ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->c ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->d0 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->d1 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->d2 ) );
  CHKERR_LIBXS_DNN( libxs_dnn_destroy_tensor( lstm->d ) );
}


int lstm (const libxs_blasint m, const libxs_blasint n, const libxs_blasint k, const libxs_blasint t,
          const libxs_blasint bm, const libxs_blasint bn, const libxs_blasint bk, const libxs_bgemm_order order, const int nrepeat,
          const libxs_blasint b_m1, const libxs_blasint b_n1, const libxs_blasint b_k1, const libxs_blasint b_k2, const libxs_blasint b_m2,
          const libxs_blasint ldw, const libxs_blasint ldx, const libxs_blasint ldz, const libxs_blasint ldu, const libxs_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXS_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (4.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXS_GEMM_FLAGS(transa, transb);
  const REAL_TYPE alpha = 1, beta = 1;

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload target(LIBXS_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE* wigold = (REAL_TYPE*)libxs_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* wfgold = (REAL_TYPE*)libxs_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* wogold = (REAL_TYPE*)libxs_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* wcgold = (REAL_TYPE*)libxs_malloc(ldw * k * sizeof(REAL_TYPE));
    REAL_TYPE* xgoldt = (REAL_TYPE*)libxs_malloc(ldx * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* rigold = (REAL_TYPE*)libxs_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* rfgold = (REAL_TYPE*)libxs_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* rogold = (REAL_TYPE*)libxs_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* rcgold = (REAL_TYPE*)libxs_malloc(ldu * m * sizeof(REAL_TYPE));
    REAL_TYPE* hgold = (REAL_TYPE*)libxs_malloc(ldh * n * sizeof(REAL_TYPE));
    REAL_TYPE* i1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* i2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* f1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* f2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* o1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* o2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* c1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* c2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* igold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* fgold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* ogold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* cgold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* d1gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* d2gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* d0gold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
    REAL_TYPE* dgold = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
#if defined(NON_FUSED_INPUT_GEMM)
    REAL_TYPE* wi = (REAL_TYPE*)libxs_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* wf = (REAL_TYPE*)libxs_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* wo = (REAL_TYPE*)libxs_malloc(m * k * sizeof(REAL_TYPE));
    REAL_TYPE* wc = (REAL_TYPE*)libxs_malloc(m * k * sizeof(REAL_TYPE));
#else
    REAL_TYPE* wi = (REAL_TYPE*)libxs_malloc(m * 4 * k * sizeof(REAL_TYPE));
    REAL_TYPE* wf = 0;
    REAL_TYPE* wo = 0;
    REAL_TYPE* wc = 0;
#endif
    REAL_TYPE* xt = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* ri = (REAL_TYPE*)libxs_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* rf = (REAL_TYPE*)libxs_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* ro = (REAL_TYPE*)libxs_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* rc = (REAL_TYPE*)libxs_malloc(m * m * sizeof(REAL_TYPE));
    REAL_TYPE* h = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
#if defined(NON_FUSED_INPUT_GEMM)
    REAL_TYPE* i1t = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* f1t = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* o1t = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* c1t = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE) * t);
#else
    REAL_TYPE* i1t = (REAL_TYPE*)libxs_malloc(m * 4 * n * sizeof(REAL_TYPE) * t);
    REAL_TYPE* f1t = 0;
    REAL_TYPE* o1t = 0;
    REAL_TYPE* c1t = 0;
#endif
    REAL_TYPE* i2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* f2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* o2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* c2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* i = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* f = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* o = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* c = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d1 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d2 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d0 = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    REAL_TYPE* d = (REAL_TYPE*)libxs_malloc(m * n * sizeof(REAL_TYPE));
    LIBXS_VLA_DECL(2, REAL_TYPE, xgold, xgoldt, ldx * n);
    libxs_bgemm_handle* handlewx = 0;
    libxs_bgemm_handle* handleuh = 0;
    libxs_bgemm_handle* handlett = 0;
    const libxs_gemm_prefetch_type strategy = LIBXS_PREFETCH_AUTO;
    handlewx = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
#if defined(NON_FUSED_INPUT_GEMM)
    handlett = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
#else
    handlett = libxs_bgemm_handle_create(LIBXS_GEMM_PRECISION(REAL_TYPE),
      m*4, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
#endif

    struct lstm_handle lstm;
    lstm.m = m;
    lstm.n = n;
    lstm.k = k;
    lstm.t = t;
    lstm.wi = libxs_create_dnn_tensor_rnn( wi );
    lstm.wf = libxs_create_dnn_tensor_rnn( wf );
    lstm.wo = libxs_create_dnn_tensor_rnn( wo );
    lstm.wc = libxs_create_dnn_tensor_rnn( wc );
    lstm.xt = libxs_create_dnn_tensor_rnn( xt );
    lstm.ri = libxs_create_dnn_tensor_rnn( ri );
    lstm.rf = libxs_create_dnn_tensor_rnn( rf );
    lstm.ro = libxs_create_dnn_tensor_rnn( ro );
    lstm.rc = libxs_create_dnn_tensor_rnn( rc );
    lstm.h = libxs_create_dnn_tensor_rnn( h );
    lstm.i1t = libxs_create_dnn_tensor_rnn( i1t );
    lstm.i2 = libxs_create_dnn_tensor_rnn( i2 );
    lstm.f1t = libxs_create_dnn_tensor_rnn( f1t );
    lstm.f2 = libxs_create_dnn_tensor_rnn( f2 );
    lstm.o1t = libxs_create_dnn_tensor_rnn( o1t );
    lstm.o2 = libxs_create_dnn_tensor_rnn( o2 );
    lstm.c1t = libxs_create_dnn_tensor_rnn( c1t );
    lstm.c2 = libxs_create_dnn_tensor_rnn( c2 );
    lstm.i = libxs_create_dnn_tensor_rnn( i );
    lstm.f = libxs_create_dnn_tensor_rnn( f );
    lstm.o = libxs_create_dnn_tensor_rnn( o );
    lstm.c = libxs_create_dnn_tensor_rnn( c );
    lstm.d0 = libxs_create_dnn_tensor_rnn( d0 );
    lstm.d1 = libxs_create_dnn_tensor_rnn( d1 );
    lstm.d2 = libxs_create_dnn_tensor_rnn( d2 );
    lstm.d = libxs_create_dnn_tensor_rnn( d );
    lstm.handlewx = handlewx;
    lstm.handleuh = handleuh;
    lstm.handlett = handlett;

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      lstm_init(&lstm, wigold, wfgold, wogold, wcgold, xgoldt, rigold, rfgold, rogold, rcgold, hgold,
        i1gold, i2gold, f1gold, f2gold, o1gold, o2gold, c1gold, c2gold, igold, fgold, ogold, cgold,
        d0gold, d1gold, d2gold, dgold, ldw, ldx, ldz, ldu, ldh);
      lstm_execute(&lstm, nrepeat);

#if defined(CHECK)
      if (!LIBXS_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        unsigned long long start;
        double duration;
        REAL_TYPE* dtest = 0;
        int j;
        int s;
        start = libxs_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (j = 0; j < t; ++j) {
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wigold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, i1gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, f1gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wogold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, o1gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &ldw, &LIBXS_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, c1gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rigold, &ldu, hgold, &ldh, &beta, i2gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &ldu, hgold, &ldh, &beta, f2gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rogold, &ldu, hgold, &ldh, &beta, o2gold, &ldz);
            LIBXS_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &ldu, hgold, &ldh, &beta, c2gold, &ldz);
            matrix_add(m*n, i1gold, i2gold, igold);
            matrix_add(m*n, f1gold, f2gold, fgold);
            matrix_add(m*n, o1gold, o2gold, ogold);
            matrix_add(m*n, c1gold, c2gold, cgold);
            matrix_relu(m*n, igold, igold); /*sigmoid*/
            matrix_relu(m*n, fgold, fgold); /*sigmoid*/
            matrix_relu(m*n, ogold, ogold); /*sigmoid*/
            matrix_relu(m*n, cgold, cgold); /*tanh*/
            if (j == 0) {
              matrix_eltwise_mult(m*n, fgold, d0gold, d1gold);
            } else {
              matrix_eltwise_mult(m*n, fgold, dgold, d1gold);
            }
            matrix_eltwise_mult(m*n, igold, cgold, d2gold);
            matrix_add(m*n, d1gold, d2gold, dgold);
            if (j < t-1) {
              matrix_relu(m*n, dgold, dgold); /*tanh*/
              matrix_eltwise_mult(m*n, ogold, dgold, hgold);
            }
          }
        }
        duration = libxs_timer_duration(start, libxs_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxs_free(wigold); wigold = 0;
        libxs_free(wfgold); wfgold = 0;
        libxs_free(wogold); wogold = 0;
        libxs_free(wcgold); wcgold = 0;
        libxs_free(xgoldt); xgoldt = 0;
        libxs_free(rigold); rigold = 0;
        libxs_free(rfgold); rfgold = 0;
        libxs_free(rogold); rogold = 0;
        libxs_free(rcgold); rcgold = 0;
        libxs_free(hgold); hgold = 0;
        libxs_free(i1gold); i1gold = 0;
        libxs_free(i2gold); i2gold = 0;
        libxs_free(f1gold); f1gold = 0;
        libxs_free(f2gold); f2gold = 0;
        libxs_free(o1gold); o1gold = 0;
        libxs_free(o2gold); o2gold = 0;
        libxs_free(c1gold); c1gold = 0;
        libxs_free(c2gold); c2gold = 0;
        libxs_free(igold); igold = 0;
        libxs_free(fgold); fgold = 0;
        libxs_free(ogold); ogold = 0;
        libxs_free(cgold); cgold = 0;
        libxs_free(d0gold); d0gold = 0;
        libxs_free(d1gold); d1gold = 0;
        libxs_free(d2gold); d2gold = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        dtest = (REAL_TYPE*)libxs_malloc(ldz * n * sizeof(REAL_TYPE));
        if (0 != dtest) {
          libxs_matdiff_info diff;
          libxs_bgemm_copyout_c(handleuh, d, &ldz, dtest);
          if (EXIT_SUCCESS == libxs_matdiff(LIBXS_DATATYPE(REAL_TYPE), m, n, dgold, dtest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxs_free(dtest);
        }
      }
#endif
      libxs_bgemm_handle_destroy(handlewx);
      libxs_bgemm_handle_destroy(handleuh);
      libxs_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXS_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxs_free(wigold);
    libxs_free(wfgold);
    libxs_free(wogold);
    libxs_free(wcgold);
    libxs_free(xgoldt);
    libxs_free(rigold);
    libxs_free(rfgold);
    libxs_free(rogold);
    libxs_free(rcgold);
    libxs_free(hgold);
    libxs_free(i1gold);
    libxs_free(i2gold);
    libxs_free(f1gold);
    libxs_free(f2gold);
    libxs_free(o1gold);
    libxs_free(o2gold);
    libxs_free(c1gold);
    libxs_free(c2gold);
    libxs_free(igold);
    libxs_free(fgold);
    libxs_free(ogold);
    libxs_free(cgold);
    libxs_free(d0gold);
    libxs_free(d1gold);
    libxs_free(d2gold);
    libxs_free(dgold);
    lstm_destroy(&lstm);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


int main(int argc, char* argv[])
{
  const int type = (1 < argc ? atoi(argv[1]) : 0);
  const libxs_blasint m = (2 < argc ? atoi(argv[2]) : 1024);
  const libxs_blasint k = (4 < argc ? atoi(argv[4]) : m);
  const libxs_blasint n = (3 < argc ? atoi(argv[3]) : k);
  const libxs_blasint t = (5 < argc ? atoi(argv[5]) : 3);
  const libxs_blasint bm = (6 < argc ? atoi(argv[6]) : 32);
  const libxs_blasint bk = (8 < argc ? atoi(argv[8]) : bm);
  const libxs_blasint bn = (7 < argc ? atoi(argv[7]) : bk);
  const libxs_bgemm_order order = (libxs_bgemm_order)(9 < argc ? atoi(argv[9]) : 0);
  const int nrepeat = (10 < argc ? atoi(argv[10]) : 100);
  const libxs_blasint b_m1 = (11 < argc ? atoi(argv[11]) : 1);
  const libxs_blasint b_n1  = (12 < argc ? atoi(argv[12]) : 1);
  const libxs_blasint b_k1 = (13 < argc ? atoi(argv[13]) : 1);
  const libxs_blasint b_k2 = (14 < argc ? atoi(argv[14]) : 1);
  const libxs_blasint b_m2 = (15 < argc ? atoi(argv[15]) : 1);
  const libxs_blasint ldw = (16 < argc ? atoi(argv[16]) : m);
  const libxs_blasint ldx = (17 < argc ? atoi(argv[17]) : k);
  const libxs_blasint ldz = (18 < argc ? atoi(argv[18]) : m);
  const libxs_blasint ldu = (19 < argc ? atoi(argv[19]) : m);
  const libxs_blasint ldh = (20 < argc ? atoi(argv[20]) : m);
  int result = EXIT_SUCCESS;
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./lstm [type: 0--RNN, 1--LSTM] [M] [N] [K] [time_steps] [bm] [bn] [bk] [order] [reps] [b_m1] [b_n1] [b_k1] [b_k2] [b_m2]\n\n");
    return result;
  }
  if (type == 0) {
    fprintf(stdout, "Running RNN ...\n");
    return rnn (m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh);
  } else if (type == 1) {
    fprintf(stdout, "Running LSTM ...\n");
    return lstm(m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh);
  } else {
    fprintf(stdout, "Type %d currently not implemented!\n", type);
    return result;
  }
}

