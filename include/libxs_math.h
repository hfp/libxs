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
#ifndef LIBXS_MATH_H
#define LIBXS_MATH_H

#include "libxs_typedefs.h"


/**
 * Structure of differences with matrix norms according
 * to http://www.netlib.org/lapack/lug/node75.html).
 */
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_matdiff_info {
  /** One-norm */         double norm1_abs, norm1_rel;
  /** Infinity-norm */    double normi_abs, normi_rel;
  /** Froebenius-norm */  double normf_rel;
  /** L1-norm and L2-norm of differences. */
  double l2_abs, l2_rel, l1_ref, l1_tst;
  /** Maximum absolute and relative error. */
  double linf_abs, linf_rel;
  /** Location of maximum error (m, n). */
  libxs_blasint linf_abs_m, linf_abs_n;
} libxs_matdiff_info;


/** Utility function to calculate the difference between two matrices. */
LIBXS_API int libxs_matdiff(libxs_datatype datatype, libxs_blasint m, libxs_blasint n,
  const void* ref, const void* tst, const libxs_blasint* ldref, const libxs_blasint* ldtst,
  libxs_matdiff_info* info);

LIBXS_API void libxs_matdiff_reduce(libxs_matdiff_info* output, const libxs_matdiff_info* input);


/* SQRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_isqrt_u64(unsigned long long x);
/* SQRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_isqrt_u32(unsigned int x);
/* SQRT with Newton's method using double-precision. */
LIBXS_API double libxs_dsqrt(double x);
/* SQRT with Newton's method using single-precision. */
LIBXS_API float libxs_ssqrt(float x);


/* CBRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_icbrt_u64(unsigned long long x);
/* CBRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_icbrt_u32(unsigned int x);


/**
 * Exponential function, which exposes the number of iterations taken in the main case (1...22). For example,
 * a value of maxiter=13 yields fast (but reasonable results), whereas maxiter=20 yields more accurate results.
 */
LIBXS_API float libxs_sexp2_fast(float x, int maxiter);

/* A wrapper around libxs_sexp2_fast (or powf), which aims for accuracy. */
LIBXS_API float libxs_sexp2(float x);

/**
 * Exponential function (base 2), which is limited to unsigned 8-bit input values (0...255).
 * This function produces bit-accurate results (single-precision).
 */
LIBXS_API float libxs_sexp2_u8(unsigned char x);

/**
 * Exponential function (base 2), which is limited to signed 8-bit input values (-128...127).
 * This function produces bit-accurate results (single-precision).
 */
LIBXS_API float libxs_sexp2_i8(signed char x);

/** Similar to libxs_sexp2_i8: checks a full integer to fit into a signed 8-bit value. */
LIBXS_API float libxs_sexp2_i8i(int x);

#endif /*LIBXS_MATH_H*/
