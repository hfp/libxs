/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_math.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  /* 1-D: constant array, all derivatives beyond order 0 should be zero */
  { double a[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4, 0, 0, 0)) {
      FPRINTF(stderr, "ERROR line #%i: constant fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (4 != fa.order || 5 != fa.n) {
      FPRINTF(stderr, "ERROR line #%i: order=%d n=%d\n", __LINE__, fa.order, fa.n);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.l2[0]) {
      FPRINTF(stderr, "ERROR line #%i: constant norm[0]=%.17g\n", __LINE__, fa.l2[0]);
      exit(EXIT_FAILURE);
    }
    { int k;
      for (k = 1; k <= fa.order; ++k) {
        if (1E-10 < fa.l2[k]) {
          FPRINTF(stderr, "ERROR line #%i: constant norm[%d]=%.17g\n", __LINE__, k, fa.l2[k]);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  /* 1-D: linear ramp, order-1 norm nonzero, order>=2 should be zero */
  { double a[] = {0.0, 1.0, 2.0, 3.0, 4.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4, 0, 0, 0)) {
      FPRINTF(stderr, "ERROR line #%i: linear fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.l2[0] || 1E-12 > fa.l2[1]) {
      FPRINTF(stderr, "ERROR line #%i: linear norm[0]=%.17g norm[1]=%.17g\n",
        __LINE__, fa.l2[0], fa.l2[1]);
      exit(EXIT_FAILURE);
    }
    { int k;
      for (k = 2; k <= fa.order; ++k) {
        if (1E-8 < fa.l2[k]) {
          FPRINTF(stderr, "ERROR line #%i: linear norm[%d]=%.17g\n", __LINE__, k, fa.l2[k]);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  /* 1-D: self-distance is zero */
  { double a[] = {1.0, 3.0, 2.0, 7.0, 5.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4, 0, 0, 0);
    d = libxs_fprint_diff(&fa, &fa, NULL);
    if (1E-15 < d) {
      FPRINTF(stderr, "ERROR line #%i: self-distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: identical data, different lengths */
  { double a5[] = {0.0, 1.0, 4.0, 9.0, 16.0};
    double a3[] = {0.0, 4.0, 16.0};
    const size_t s5[] = {5}, s3[] = {3};
    libxs_fprint_t f5, f3;
    libxs_fprint(&f5, LIBXS_DATATYPE_F64, a5, 1, s5, NULL, 2, 0, 0, 0);
    libxs_fprint(&f3, LIBXS_DATATYPE_F64, a3, 1, s3, NULL, 2, 0, 0, 0);
    FPRINTF(stderr, "INFO line #%i: quadratic 5vs3 dist=%.17g\n",
      __LINE__, libxs_fprint_diff(&f5, &f3, NULL));
  }

  /* 1-D: structurally different data yields nonzero distance */
  { double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {1.0, 4.0, 2.0, 8.0};
    const size_t shape[] = {4};
    libxs_fprint_t fa, fb;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 3, 0, 0, 0);
    d = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-12 > d) {
      FPRINTF(stderr, "ERROR line #%i: different distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: symmetry d(a,b) == d(b,a) */
  { double a[] = {1.0, 4.0, 2.0, 8.0};
    double b[] = {2.0, 3.0, 5.0, 7.0};
    const size_t shape[] = {4};
    libxs_fprint_t fa, fb;
    double dab, dba;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 3, 0, 0, 0);
    dab = libxs_fprint_diff(&fa, &fb, NULL);
    dba = libxs_fprint_diff(&fb, &fa, NULL);
    if (1E-15 < LIBXS_DELTA(dab, dba)) {
      FPRINTF(stderr, "ERROR line #%i: symmetry dab=%.17g dba=%.17g\n", __LINE__, dab, dba);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: custom weights */
  { double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {1.1, 2.1, 3.1, 4.1};
    double w[] = {1.0, 0.0, 0.0, 0.0};
    const size_t shape[] = {4};
    libxs_fprint_t fa, fb;
    double d_val, d_def;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 3, 0, 0, 0);
    d_val = libxs_fprint_diff(&fa, &fb, w);
    d_def = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-15 < LIBXS_DELTA(d_val, d_def) && d_val > d_def) {
      FPRINTF(stderr, "ERROR line #%i: value-only > default\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: F32 type */
  { float a[] = {1.0f, 2.0f, 4.0f, 8.0f};
    const size_t shape[] = {4};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F32, a, 1, shape, NULL, 3, 0, 0, 0)) {
      FPRINTF(stderr, "ERROR line #%i: F32 fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.l2[0]) {
      FPRINTF(stderr, "ERROR line #%i: F32 norm[0]=%.17g\n", __LINE__, fa.l2[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: I32 type */
  { int a[] = {10, 20, 30, 40};
    const size_t shape[] = {4};
    libxs_fprint_t fa;
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_I32, a, 1, shape, NULL, 2, 0, 0, 0)) {
      FPRINTF(stderr, "ERROR line #%i: I32 fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.l2[0]) {
      FPRINTF(stderr, "ERROR line #%i: I32 norm[0]=%.17g\n", __LINE__, fa.l2[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: order clamped to n-1 */
  { double a[] = {1.0, 2.0};
    const size_t shape[] = {2};
    libxs_fprint_t fa;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 10, 0, 0, 0);
    if (1 != fa.order) {
      FPRINTF(stderr, "ERROR line #%i: clamp order=%d\n", __LINE__, fa.order);
      exit(EXIT_FAILURE);
    }
  }

  /* 1-D: single element */
  { double a[] = {42.0};
    const size_t shape[] = {1};
    libxs_fprint_t fa;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 5, 0, 0, 0);
    if (0 != fa.order || 1 != fa.n) {
      FPRINTF(stderr, "ERROR line #%i: single order=%d n=%d\n", __LINE__, fa.order, fa.n);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.l2[0]) {
      FPRINTF(stderr, "ERROR line #%i: single norm[0]=%.17g\n", __LINE__, fa.l2[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: identical matrices yield zero self-distance */
  { double m[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0,
                  7.0, 8.0, 9.0,
                  10.0, 11.0, 12.0};
    const size_t shape[] = {3, 4};
    libxs_fprint_t fa, fb;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, m, 2, shape, NULL, 2, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, m, 2, shape, NULL, 2, 0, 0, 0);
    d = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-15 < d) {
      FPRINTF(stderr, "ERROR line #%i: 2D self-distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: different matrices yield nonzero distance */
  { double a[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    double b[] = {1.0, 9.0, 3.0,
                  4.0, 0.0, 6.0};
    const size_t shape[] = {3, 2};
    libxs_fprint_t fa, fb;
    double d;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 2, shape, NULL, 2, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 2, shape, NULL, 2, 0, 0, 0);
    d = libxs_fprint_diff(&fa, &fb, NULL);
    if (1E-12 > d) {
      FPRINTF(stderr, "ERROR line #%i: 2D different distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: symmetry */
  { double a[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double b[] = {6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    const size_t shape[] = {3, 2};
    libxs_fprint_t fa, fb;
    double dab, dba;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 2, shape, NULL, 2, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 2, shape, NULL, 2, 0, 0, 0);
    dab = libxs_fprint_diff(&fa, &fb, NULL);
    dba = libxs_fprint_diff(&fb, &fa, NULL);
    if (1E-14 < LIBXS_DELTA(dab, dba)) {
      FPRINTF(stderr, "ERROR line #%i: 2D symmetry dab=%.17g dba=%.17g\n", __LINE__, dab, dba);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D with stride: column-major 3x2 with leading dimension 4 */
  { double m[] = {1.0, 2.0, 3.0, 99.0,
                  4.0, 5.0, 6.0, 99.0};
    const size_t shape[] = {3, 2};
    const size_t stride[] = {1, 4};
    double ref[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const size_t rshape[] = {3, 2};
    libxs_fprint_t fm, fr;
    double d;
    libxs_fprint(&fm, LIBXS_DATATYPE_F64, m, 2, shape, stride, 2, 0, 0, 0);
    libxs_fprint(&fr, LIBXS_DATATYPE_F64, ref, 2, rshape, NULL, 2, 0, 0, 0);
    d = libxs_fprint_diff(&fm, &fr, NULL);
    if (1E-14 < d) {
      FPRINTF(stderr, "ERROR line #%i: 2D stride distance=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }

  /* 2-D: different shapes are comparable */
  { double a[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    double b[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                  7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    const size_t sa[] = {3, 2}, sb[] = {6, 2};
    libxs_fprint_t fa, fb;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 2, sa, NULL, 2, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 2, sb, NULL, 2, 0, 0, 0);
    FPRINTF(stderr, "INFO line #%i: 2D shape diff dist=%.17g\n",
      __LINE__, libxs_fprint_diff(&fa, &fb, NULL));
  }

  /* 3-D: smoke test */
  { double a[2*3*4];
    const size_t shape[] = {4, 3, 2};
    libxs_fprint_t fa;
    int i;
    for (i = 0; i < 24; ++i) a[i] = (double)(i * i);
    if (EXIT_SUCCESS != libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 3, shape, NULL, 2, 0, 0, 0)) {
      FPRINTF(stderr, "ERROR line #%i: 3D fprint failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (1E-12 > fa.l2[0]) {
      FPRINTF(stderr, "ERROR line #%i: 3D norm[0]=%.17g\n", __LINE__, fa.l2[0]);
      exit(EXIT_FAILURE);
    }
  }

  /**
   * N-D per-axis: 3D array, shape={3, 4, 2}, linear along axis 1 only.
   * m[i][j][k] = 1 + j  (j=0..3 is the only varying axis).
   * axis=0: constant along dim 0 -> all derivatives ~ 0
   * axis=1: linear along dim 1   -> d1 nonzero, d2+ ~ 0
   * axis=2: constant along dim 2 -> all derivatives ~ 0
   */
  { double m[3 * 4 * 2];
    const size_t shape[] = {3, 4, 2};
    libxs_fprint_t fp;
    int i0, i1, i2, k, ax;
    for (i2 = 0; i2 < 2; ++i2) {
      for (i1 = 0; i1 < 4; ++i1) {
        for (i0 = 0; i0 < 3; ++i0) {
          m[i2 * 12 + i1 * 3 + i0] = 1.0 + i1;
        }
      }
    }
    for (ax = 0; ax < 3; ++ax) {
      if (EXIT_SUCCESS != libxs_fprint(&fp, LIBXS_DATATYPE_F64, m, 3, shape, NULL, 3, ax, 0, LIBXS_FPRINT_PERAXIS)) {
        FPRINTF(stderr, "ERROR line #%i: 3D axis=%d fprint failed\n", __LINE__, ax);
        exit(EXIT_FAILURE);
      }
      if (1 == ax) {
        if (1E-12 > fp.linf[1]) {
          FPRINTF(stderr, "ERROR line #%i: 3D axis=1 linf[1]=%.17g\n", __LINE__, fp.linf[1]);
          exit(EXIT_FAILURE);
        }
        for (k = 2; k <= fp.order; ++k) {
          if (1E-8 < fp.linf[k]) {
            FPRINTF(stderr, "ERROR line #%i: 3D axis=1 linf[%d]=%.17g\n", __LINE__, k, fp.linf[k]);
            exit(EXIT_FAILURE);
          }
        }
      }
      else {
        for (k = 1; k <= fp.order; ++k) {
          if (1E-10 < fp.linf[k]) {
            FPRINTF(stderr, "ERROR line #%i: 3D axis=%d linf[%d]=%.17g\n", __LINE__, ax, k, fp.linf[k]);
            exit(EXIT_FAILURE);
          }
        }
      }
    }
  }

  /* per-axis: libxs_fprint_raw recovers raw forward difference */
  { double a[] = {0.0, 10.0, 20.0, 30.0, 40.0};
    const size_t shape[] = {5};
    libxs_fprint_t fa;
    double raw_d1;
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 2, 0, 0, 0);
    raw_d1 = libxs_fprint_raw(&fa, 1, fa.linf[1]);
    if (1.0 < LIBXS_DELTA(raw_d1, 10.0)) {
      FPRINTF(stderr, "ERROR line #%i: raw d1=%.17g expected 10\n", __LINE__, raw_d1);
      exit(EXIT_FAILURE);
    }
  }

  /* partial: single chunk matches batch fprint */
  { double a[] = {1.0, 3.0, 2.0, 7.0, 5.0, 4.0, 8.0, 6.0};
    const size_t shape[] = {8};
    libxs_fprint_t batch, partial;
    int k;
    libxs_fprint(&batch, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4, 0, 0, 0);
    memset(&partial, 0, sizeof(partial));
    if (EXIT_SUCCESS != libxs_fprint_partial(&partial, LIBXS_DATATYPE_F64, a, 8, 4)) {
      FPRINTF(stderr, "ERROR line #%i: partial single-chunk failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (batch.order != partial.order || batch.n != partial.n) {
      FPRINTF(stderr, "ERROR line #%i: partial order=%d/%d n=%d/%d\n",
        __LINE__, batch.order, partial.order, batch.n, partial.n);
      exit(EXIT_FAILURE);
    }
    for (k = 0; k <= batch.order; ++k) {
      if (1E-10 < LIBXS_DELTA(batch.l2[k], partial.l2[k])) {
        FPRINTF(stderr, "ERROR line #%i: partial l2[%d]=%.17g vs %.17g\n",
          __LINE__, k, batch.l2[k], partial.l2[k]);
        exit(EXIT_FAILURE);
      }
      if (1E-10 < LIBXS_DELTA(batch.mean[k], partial.mean[k])) {
        FPRINTF(stderr, "ERROR line #%i: partial mean[%d]=%.17g vs %.17g\n",
          __LINE__, k, batch.mean[k], partial.mean[k]);
        exit(EXIT_FAILURE);
      }
    }
  }

  /* partial: two chunks match batch on concatenation */
  { double a[] = {1.0, 3.0, 2.0, 7.0, 5.0, 4.0, 8.0, 6.0};
    const size_t shape[] = {8};
    libxs_fprint_t batch, partial;
    int k;
    libxs_fprint(&batch, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 3, 0, 0, 0);
    memset(&partial, 0, sizeof(partial));
    libxs_fprint_partial(&partial, LIBXS_DATATYPE_F64, a, 4, 3);
    if (EXIT_SUCCESS != libxs_fprint_partial(&partial, LIBXS_DATATYPE_F64, a + 4, 4, 3)) {
      FPRINTF(stderr, "ERROR line #%i: partial two-chunk failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (batch.n != partial.n) {
      FPRINTF(stderr, "ERROR line #%i: partial n=%d/%d\n",
        __LINE__, batch.n, partial.n);
      exit(EXIT_FAILURE);
    }
    for (k = 0; k <= batch.order; ++k) {
      if (1E-10 < LIBXS_DELTA(batch.l2[k], partial.l2[k])) {
        FPRINTF(stderr, "ERROR line #%i: partial 2-chunk l2[%d]=%.17g vs %.17g\n",
          __LINE__, k, batch.l2[k], partial.l2[k]);
        exit(EXIT_FAILURE);
      }
      if (1E-10 < LIBXS_DELTA(batch.mean[k], partial.mean[k])) {
        FPRINTF(stderr, "ERROR line #%i: partial 2-chunk mean[%d]=%.17g vs %.17g\n",
          __LINE__, k, batch.mean[k], partial.mean[k]);
        exit(EXIT_FAILURE);
      }
    }
  }

  /* join: merging two halves approximates the full fingerprint */
  { double a[] = {1.0, 3.0, 2.0, 7.0, 5.0, 4.0, 8.0, 6.0};
    const size_t s_full[] = {8}, s_half[] = {4};
    libxs_fprint_t full, fa, fb, joined;
    double d_full, d_joined;
    libxs_fprint(&full, LIBXS_DATATYPE_F64, a, 1, s_full, NULL, 3, 0, 0, 0);
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, s_half, NULL, 3, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, a + 4, 1, s_half, NULL, 3, 0, 0, 0);
    if (EXIT_SUCCESS != libxs_fprint_join(&joined, &fa, &fb)) {
      FPRINTF(stderr, "ERROR line #%i: join failed\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (8 != joined.n) {
      FPRINTF(stderr, "ERROR line #%i: join n=%d\n", __LINE__, joined.n);
      exit(EXIT_FAILURE);
    }
    d_full = libxs_fprint_diff(&full, &full, NULL);
    d_joined = libxs_fprint_diff(&full, &joined, NULL);
    FPRINTF(stderr, "INFO line #%i: join approx dist=%.6g\n", __LINE__, d_joined);
    if (d_joined > 1000.0) {
      FPRINTF(stderr, "ERROR line #%i: join too far dist=%.17g\n", __LINE__, d_joined);
      exit(EXIT_FAILURE);
    }
    LIBXS_UNUSED(d_full);
  }

  /* Bernstein norms: bounded by the fingerprint, and matching known values */
  { double sine[41], step[41], noise[41], ratio;
    double norms[LIBXS_FPRINT_MAXORDER + 1];
    const size_t shape[] = {41};
    unsigned int seed = 1;
    libxs_fprint_t fp;
    int i, k, c;
    for (i = 0; i < 41; ++i) {
      sine[i] = sin(2 * 3.14159265358979323846 * i / 40.0);
      step[i] = (0.37 <= i / 40.0 ? 1.0 : 0.0);
      seed = seed * 1103515245U + 12345U;
      noise[i] = (double)(seed >> 16 & 0x7FFF) / 0x7FFF - 0.5;
    }
    for (c = 0; c < 3; ++c) {
      const double *const data = (0 == c ? sine : (1 == c ? step : noise));
      libxs_fprint(&fp, LIBXS_DATATYPE_F64, data, 1, shape, NULL, 4, 0, 0, 0);
      if (EXIT_SUCCESS != libxs_fprint_bernstein(norms, LIBXS_DATATYPE_F64, data, 41, 4)) {
        FPRINTF(stderr, "ERROR line #%i: bernstein failed\n", __LINE__);
        exit(EXIT_FAILURE);
      }
      for (k = 0; k <= 4; ++k) {
        ratio = norms[k] / fp.l2[k];
        if (1 + 1E-12 < ratio) {
          FPRINTF(stderr, "ERROR line #%i: bernstein case %d order %d ratio=%.17g\n",
            __LINE__, c, k, ratio);
          exit(EXIT_FAILURE);
        }
      }
      if (2 > c) { /* independent quadrature: 0.9172 and 0.3071 at order 1 */
        ratio = norms[1] / fp.l2[1];
        if (1E-3 < LIBXS_DELTA(ratio, 0 == c ? 0.9172 : 0.3071)) {
          FPRINTF(stderr, "ERROR line #%i: bernstein case %d ratio=%.17g\n",
            __LINE__, c, ratio);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  /* Bernstein norms reproduce a linear function exactly, also when banded */
  { static double line[2001];
    double norms[LIBXS_FPRINT_MAXORDER + 1];
    const int sizes[] = {2, 41, 2001};
    int i, k, j;
    for (j = 0; j < 3; ++j) {
      const int n = sizes[j];
      for (i = 0; i < n; ++i) line[i] = (double)i / (n - 1);
      if (EXIT_SUCCESS != libxs_fprint_bernstein(norms, LIBXS_DATATYPE_F64, line, n, 3)) {
        FPRINTF(stderr, "ERROR line #%i: bernstein failed\n", __LINE__);
        exit(EXIT_FAILURE);
      }
      for (k = 0; k <= 3; ++k) {
        const double expect = (0 == k ? sqrt(1.0 / 3.0) : (1 == k ? 1.0 : 0.0));
        if (1E-9 + 1E-13 * pow(n - 1, k) < LIBXS_DELTA(norms[k], expect)) { /* rounding */
          FPRINTF(stderr, "ERROR line #%i: bernstein n=%d order %d norm=%.17g\n",
            __LINE__, n, k, norms[k]);
          exit(EXIT_FAILURE);
        }
      }
    }
    for (i = 0; i < 5; ++i) line[i] = 3.0;
    libxs_fprint_bernstein(norms, LIBXS_DATATYPE_F64, line, 5, 1);
    if (1E-12 < LIBXS_DELTA(norms[0], 3.0) || 1E-12 < norms[1]) {
      FPRINTF(stderr, "ERROR line #%i: bernstein constant %.17g %.17g\n",
        __LINE__, norms[0], norms[1]);
      exit(EXIT_FAILURE);
    }
  }

  /* Bernstein distance: a metric across lengths, exact for equal lengths */
  { static double fine[2001];
    double a[21], b[41], c[41], diff[41], norms[LIBXS_FPRINT_MAXORDER + 1];
    const double w0[] = {1, 0, 0, 0};
    double d, dab, dba, dac, dbc, ref = 0, expect = 0, wk = 1;
    int i, k, q;
    for (i = 0; i < 21; ++i) a[i] = sin(3.0 * i / 20.0);
    for (i = 0; i < 41; ++i) {
      b[i] = sin(3.0 * i / 40.0) + 0.01 * (i % 3);
      c[i] = cos(2.0 * i / 40.0);
      diff[i] = b[i] - c[i];
    }
    libxs_fprint_bernstein_diff(&d, LIBXS_DATATYPE_F64, b, 41, b, 41, 3, NULL);
    if (0 != d) {
      FPRINTF(stderr, "ERROR line #%i: bernstein_diff self=%.17g\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
    /* equal lengths: the distance is the norm of the difference (B is linear) */
    libxs_fprint_bernstein_diff(&d, LIBXS_DATATYPE_F64, b, 41, c, 41, 3, NULL);
    libxs_fprint_bernstein(norms, LIBXS_DATATYPE_F64, diff, 41, 3);
    for (k = 0; k <= 3; ++k) {
      if (0 < k) wk /= k;
      expect += wk * norms[k] * norms[k];
    }
    if (1E-12 < LIBXS_DELTA(d, sqrt(expect))) {
      FPRINTF(stderr, "ERROR line #%i: bernstein_diff=%.17g vs %.17g\n", __LINE__,
        d, sqrt(expect));
      exit(EXIT_FAILURE);
    }
    /* different lengths: order 0 against de Casteljau and Simpson's rule */
    for (q = 0; q < 2001; ++q) {
      const double x = q / 2000.0;
      double ta[21], tb[41];
      int r;
      for (i = 0; i < 21; ++i) ta[i] = a[i];
      for (i = 0; i < 41; ++i) tb[i] = b[i];
      for (r = 20; 0 < r; --r) {
        for (i = 0; i < r; ++i) ta[i] = (1 - x) * ta[i] + x * ta[i + 1];
      }
      for (r = 40; 0 < r; --r) {
        for (i = 0; i < r; ++i) tb[i] = (1 - x) * tb[i] + x * tb[i + 1];
      }
      fine[q] = (ta[0] - tb[0]) * (ta[0] - tb[0]);
    }
    for (q = 0; q < 2001; ++q) {
      ref += fine[q] * (0 == q || 2000 == q ? 1 : (1 == (q & 1) ? 4 : 2));
    }
    ref = sqrt(ref / (3.0 * 2000.0));
    libxs_fprint_bernstein_diff(&dab, LIBXS_DATATYPE_F64, a, 21, b, 41, 3, w0);
    if (1E-9 < LIBXS_DELTA(dab, ref)) {
      FPRINTF(stderr, "ERROR line #%i: bernstein_diff=%.17g vs quadrature %.17g\n",
        __LINE__, dab, ref);
      exit(EXIT_FAILURE);
    }
    /* symmetry and the triangle inequality */
    libxs_fprint_bernstein_diff(&dab, LIBXS_DATATYPE_F64, a, 21, b, 41, 3, NULL);
    libxs_fprint_bernstein_diff(&dba, LIBXS_DATATYPE_F64, b, 41, a, 21, 3, NULL);
    libxs_fprint_bernstein_diff(&dac, LIBXS_DATATYPE_F64, a, 21, c, 41, 3, NULL);
    libxs_fprint_bernstein_diff(&dbc, LIBXS_DATATYPE_F64, b, 41, c, 41, 3, NULL);
    if (1E-12 < LIBXS_DELTA(dab, dba) || dac > dab + dbc + 1E-12) {
      FPRINTF(stderr, "ERROR line #%i: bernstein_diff %.17g %.17g %.17g %.17g\n",
        __LINE__, dab, dba, dac, dbc);
      exit(EXIT_FAILURE);
    }
  }

  /* stored Bernstein norms (LIBXS_FPRINT_WITH_BERNSTEIN) and their term in the distance */
  { double a[33], b[33], norms[LIBXS_FPRINT_MAXORDER + 1], plain, both, mixed, extra = 0;
    const size_t shape[] = {33};
    libxs_fprint_t fa, fb, ga, gb;
    double wk = 1;
    int i, k;
    for (i = 0; i < 33; ++i) {
      a[i] = sin(4.0 * i / 32.0);
      b[i] = (i < 13 ? 0.0 : 1.0);
    }
    libxs_fprint(&fa, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4, 0, 0, 0);
    libxs_fprint(&fb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 4, 0, 0, 0);
    libxs_fprint(&ga, LIBXS_DATATYPE_F64, a, 1, shape, NULL, 4, 0, 0,
      LIBXS_FPRINT_WITH_BERNSTEIN);
    libxs_fprint(&gb, LIBXS_DATATYPE_F64, b, 1, shape, NULL, 4, 0, 0,
      LIBXS_FPRINT_WITH_BERNSTEIN);
    libxs_fprint_bernstein(norms, LIBXS_DATATYPE_F64, a, 33, 4);
    if (0 != fa.nbernstein || 5 != ga.nbernstein) {
      FPRINTF(stderr, "ERROR line #%i: nbernstein %d %d\n", __LINE__,
        fa.nbernstein, ga.nbernstein);
      exit(EXIT_FAILURE);
    }
    for (k = 0; k <= 4; ++k) {
      if (norms[k] != ga.bernstein[k] || ga.l2[k] != fa.l2[k]) {
        FPRINTF(stderr, "ERROR line #%i: stored bernstein[%d]=%.17g vs %.17g\n",
          __LINE__, k, ga.bernstein[k], norms[k]);
        exit(EXIT_FAILURE);
      }
      if (0 < k) wk /= k;
      extra += wk * (ga.bernstein[k] - gb.bernstein[k]) * (ga.bernstein[k] - gb.bernstein[k]);
    }
    plain = libxs_fprint_diff(&fa, &fb, NULL);
    both = libxs_fprint_diff(&ga, &gb, NULL);
    mixed = libxs_fprint_diff(&ga, &fb, NULL);
    if (1E-12 * both * both < LIBXS_DELTA(both * both, plain * plain + extra)
      || plain != mixed)
    {
      FPRINTF(stderr, "ERROR line #%i: diff plain=%.17g both=%.17g mixed=%.17g\n",
        __LINE__, plain, both, mixed);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
