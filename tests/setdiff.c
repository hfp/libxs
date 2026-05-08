/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_math.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  /* identical elements, permuted order */
  { double a[] = {3.0, 1.0, 2.0}, b[] = {1.0, 2.0, 3.0};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 permuted identical\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* single mismatch */
  { double a[] = {1.0, 2.0, 3.0}, b[] = {1.0, 2.0, 4.0};
    if (1 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 single mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* tolerance absorbs mismatch */
  { double a[] = {1.0, 2.0, 3.0}, b[] = {1.0, 2.0, 4.0};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 3, b, 3, 1.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 tolerance\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* completely disjoint */
  { double a[] = {1.0, 2.0}, b[] = {5.0, 6.0};
    if (2 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 2, b, 2, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 disjoint\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* symmetry: d(a,b) == d(b,a) */
  { double a[] = {1.0, 5.0}, b[] = {2.0, 6.0};
    const int dab = libxs_setdiff(LIBXS_DATATYPE_F64, a, 2, b, 2, 0.5);
    const int dba = libxs_setdiff(LIBXS_DATATYPE_F64, b, 2, a, 2, 0.5);
    if (dab != dba) {
      FPRINTF(stderr, "ERROR line #%i: F64 symmetry %d != %d\n", __LINE__, dab, dba);
      exit(EXIT_FAILURE);
    }
  }
  /* identity of indiscernibles */
  { double a[] = {7.0, -3.0, 0.5};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 3, a, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 self-identity\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* different lengths: distance >= |na-nb| */
  { double a[] = {1.0, 2.0, 3.0, 4.0}, b[] = {1.0, 2.0};
    const int d = libxs_setdiff(LIBXS_DATATYPE_F64, a, 4, b, 2, 0.0);
    if (d < 2) {
      FPRINTF(stderr, "ERROR line #%i: F64 length difference %d < 2\n", __LINE__, d);
      exit(EXIT_FAILURE);
    }
  }
  /* empty vectors */
  { double a[] = {1.0};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 0, a, 0, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 empty-empty\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (2 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 0, a, 2, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 empty vs non-empty\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* multiset: duplicate in a cannot match single in b twice */
  { double a[] = {1.0, 1.0}, b[] = {1.0, 2.0};
    if (1 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 2, b, 2, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 multiset double-count\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* multiset: greedy order matters -- sort ensures optimal 1-to-1 */
  { double a[] = {1.0, 2.0}, b[] = {1.5, 1.0};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 2, b, 2, 0.6)) {
      FPRINTF(stderr, "ERROR line #%i: F64 multiset greedy order\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  /* multiset: three duplicates vs two -- only two can match */
  { double a[] = {5.0, 5.0, 5.0}, b[] = {5.0, 5.0, 9.0};
    if (1 != libxs_setdiff(LIBXS_DATATYPE_F64, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F64 multiset dup count\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* F32 */
  { float a[] = {1.0f, 2.0f, 3.0f}, b[] = {3.0f, 1.0f, 2.0f};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F32, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F32 permuted identical\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  { float a[] = {1.0f, 2.0f}, b[] = {1.5f, 2.5f};
    if (2 != libxs_setdiff(LIBXS_DATATYPE_F32, a, 2, b, 2, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: F32 mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (0 != libxs_setdiff(LIBXS_DATATYPE_F32, a, 2, b, 2, 0.5)) {
      FPRINTF(stderr, "ERROR line #%i: F32 tolerance\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* C64: complex modulus matching */
  { double a[] = {1.0, 0.0,  0.0, 1.0};
    double b[] = {0.0, 1.0,  1.0, 0.0};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_C64, a, 2, b, 2, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: C64 permuted identical\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  { double a[] = {1.0, 0.0,  0.0, 1.0};
    double b[] = {2.0, 0.0,  0.0, 1.0};
    if (1 != libxs_setdiff(LIBXS_DATATYPE_C64, a, 2, b, 2, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: C64 mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (0 != libxs_setdiff(LIBXS_DATATYPE_C64, a, 2, b, 2, 1.0)) {
      FPRINTF(stderr, "ERROR line #%i: C64 tolerance\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* I32 */
  { int a[] = {10, 20, 30}, b[] = {30, 10, 20};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_I32, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: I32 permuted identical\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }
  { int a[] = {10, 20, 30}, b[] = {10, 20, 40};
    if (1 != libxs_setdiff(LIBXS_DATATYPE_I32, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: I32 mismatch\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (0 != libxs_setdiff(LIBXS_DATATYPE_I32, a, 3, b, 3, 10.0)) {
      FPRINTF(stderr, "ERROR line #%i: I32 tolerance\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* U8 */
  { unsigned char a[] = {10, 20, 30}, b[] = {11, 21, 31};
    if (3 != libxs_setdiff(LIBXS_DATATYPE_U8, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: U8 all differ\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (0 != libxs_setdiff(LIBXS_DATATYPE_U8, a, 3, b, 3, 1.0)) {
      FPRINTF(stderr, "ERROR line #%i: U8 tolerance\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* I16 */
  { short a[] = {-100, 0, 100}, b[] = {100, -100, 0};
    if (0 != libxs_setdiff(LIBXS_DATATYPE_I16, a, 3, b, 3, 0.0)) {
      FPRINTF(stderr, "ERROR line #%i: I16 permuted identical\n", __LINE__);
      exit(EXIT_FAILURE);
    }
  }

  /* GSS standalone: minimize (x-3)^2 on [0, 5] */
  { double xmin = -1;
    double fmin;
    double parabola(double, const void*);
    fmin = libxs_gss_min(parabola, NULL, 0.0, 5.0, &xmin, 10000);
    if (1E-10 < LIBXS_FABS(xmin - 3.0) || 1E-10 < fmin) {
      FPRINTF(stderr, "ERROR line #%i: gss parabola xmin=%.17g fmin=%.17g\n",
        __LINE__, xmin, fmin);
      exit(EXIT_FAILURE);
    }
  }

  /* setdiff_min: F64 */
  { double a[] = {1.0, 2.0, 3.0}, b[] = {1.05, 2.1, 3.0};
    double tol;
    const int dm = libxs_setdiff_min(LIBXS_DATATYPE_F64, a, 3, b, 3, &tol);
    if (0 != dm || 1E-10 < LIBXS_FABS(tol - 0.1)) {
      FPRINTF(stderr, "ERROR line #%i: F64 dmin=%d tol=%.17g\n", __LINE__, dm, tol);
      exit(EXIT_FAILURE);
    }
  }
  /* setdiff_min: F32 */
  { float a[] = {1.0f, 2.0f}, b[] = {1.5f, 2.5f};
    double tol;
    const int dm = libxs_setdiff_min(LIBXS_DATATYPE_F32, a, 2, b, 2, &tol);
    if (0 != dm || 1E-6 < LIBXS_FABS(tol - 0.5)) {
      FPRINTF(stderr, "ERROR line #%i: F32 dmin=%d tol=%.17g\n", __LINE__, dm, tol);
      exit(EXIT_FAILURE);
    }
  }
  /* setdiff_min: C64 */
  { double a[] = {1.0, 0.0,  0.0, 1.0};
    double b[] = {1.1, 0.0,  0.0, 1.1};
    double tol;
    const int dm = libxs_setdiff_min(LIBXS_DATATYPE_C64, a, 2, b, 2, &tol);
    if (0 != dm || 1E-10 < LIBXS_FABS(tol - 0.1)) {
      FPRINTF(stderr, "ERROR line #%i: C64 dmin=%d tol=%.17g\n", __LINE__, dm, tol);
      exit(EXIT_FAILURE);
    }
  }
  /* setdiff_min: identical vectors -> tol==0, dmin==0 */
  { double a[] = {1.0, 2.0, 3.0};
    double tol = -1;
    const int dm = libxs_setdiff_min(LIBXS_DATATYPE_F64, a, 3, a, 3, &tol);
    if (0 != dm || 0 != tol) {
      FPRINTF(stderr, "ERROR line #%i: F64 self dmin=%d tol=%.17g\n", __LINE__, dm, tol);
      exit(EXIT_FAILURE);
    }
  }
  /* setdiff_min: integer type -> not supported, returns max(na,nb) */
  { int a[] = {1, 2, 3}, b[] = {1, 2, 4};
    double tol = -1;
    const int dm = libxs_setdiff_min(LIBXS_DATATYPE_I32, a, 3, b, 3, &tol);
    if (3 != dm || 0 != tol) {
      FPRINTF(stderr, "ERROR line #%i: I32 dmin unsupported dm=%d tol=%.17g\n",
        __LINE__, dm, tol);
      exit(EXIT_FAILURE);
    }
  }
  /* setdiff_min: tol==NULL accepted */
  { double a[] = {1.0, 2.0}, b[] = {1.0, 3.0};
    const int dm = libxs_setdiff_min(LIBXS_DATATYPE_F64, a, 2, b, 2, NULL);
    if (0 != dm) {
      FPRINTF(stderr, "ERROR line #%i: F64 dmin tol=NULL dm=%d\n", __LINE__, dm);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}


double parabola(double x, const void* data)
{
  LIBXS_UNUSED(data);
  return (x - 3.0) * (x - 3.0);
}
