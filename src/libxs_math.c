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

#include <libxs_math.h>
#include "libxs_main.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#if !defined(LIBXS_NO_LIBM)
# include <math.h>
#endif
#include <string.h>
#include <stdio.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXS_API int libxs_matdiff(libxs_datatype datatype, libxs_blasint m, libxs_blasint n,
  const void* ref, const void* tst, const libxs_blasint* ldref, const libxs_blasint* ldtst,
  libxs_matdiff_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != ref && 0 != tst && 0 != info) {
    libxs_blasint mm = m, nn = n, ldr = (0 == ldref ? m : *ldref), ldt = (0 == ldtst ? m : *ldtst);
    if (1 == n) { mm = ldr = ldt = 1; nn = m; } /* ensure row-vector shape to standardize results */
    memset(info, 0, sizeof(*info)); /* nullify */
    switch (datatype) {
    case LIBXS_DATATYPE_F64: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE double
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
    } break;
    case LIBXS_DATATYPE_F32: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE float
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
    } break;
    case LIBXS_DATATYPE_I32: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE int
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
    } break;
    case LIBXS_DATATYPE_I16: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE short
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
    } break;
    case LIBXS_DATATYPE_I8: {
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE signed char
#       include "template/libxs_matdiff.tpl.c"
#       undef  LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
    } break;
    default: {
      static int error_once = 0;
      if (0 != libxs_verbosity /* library code is expected to be mute */
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: unsupported data-type requested for libxs_matdiff!\n");
      }
      result = EXIT_FAILURE;
    }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    info->normf_rel = libxs_dsqrt(info->normf_rel);
    info->l2_abs = libxs_dsqrt(info->l2_abs);
    info->l2_rel = libxs_dsqrt(info->l2_rel);
    if (1 == n) {
      const libxs_blasint tmp = info->linf_abs_m;
      info->linf_abs_m = info->linf_abs_n;
      info->linf_abs_n = tmp;
    }
  }
  return result;
}


LIBXS_API void libxs_matdiff_reduce(libxs_matdiff_info* output, const libxs_matdiff_info* input)
{
  LIBXS_ASSERT(0 != output && 0 != input);
  if (output->normf_rel < input->normf_rel) {
    output->linf_abs_m = input->linf_abs_m;
    output->linf_abs_n = input->linf_abs_n;
    output->norm1_abs = input->norm1_abs;
    output->norm1_rel = input->norm1_rel;
    output->normi_abs = input->normi_abs;
    output->normi_rel = input->normi_rel;
    output->normf_rel = input->normf_rel;
    output->linf_abs = input->linf_abs;
    output->linf_rel = input->linf_rel;
    output->l2_abs = input->l2_abs;
    output->l2_rel = input->l2_rel;
    output->l1_ref = input->l1_ref;
    output->l1_tst = input->l1_tst;
  }
}


LIBXS_API unsigned int libxs_isqrt_u64(unsigned long long x)
{
  unsigned long long b; unsigned int y = 0, s;
  for (s = 0x80000000/*2^31*/; 0 < s; s >>= 1) {
    b = y | s; y |= (b * b <= x ? s : 0);
  }
  return y;
}

LIBXS_API unsigned int libxs_isqrt_u32(unsigned int x)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 0x40000000/*2^30*/; 0 < s; s >>= 2) {
    b = y | s; y >>= 1;
    if (b <= x) { x -= b; y |= s; }
  }
  return y;
}


LIBXS_API LIBXS_INTRINSICS(LIBXS_X86_GENERIC) double libxs_dsqrt(double x)
{
#if defined(LIBXS_INTRINSICS_X86)
  const __m128d a = LIBXS_INTRINSICS_MM_UNDEFINED_PD();
  const double result = _mm_cvtsd_f64(_mm_sqrt_sd(a, _mm_set_sd(x)));
#else
  double result, y = x;
  if (LIBXS_NEQ(0, x)) {
    do {
      result = y;
      y = 0.5 * (y + x / y);
    } while (LIBXS_NEQ(result, y));
  }
  result = y;
#endif
  return result;
}


LIBXS_API LIBXS_INTRINSICS(LIBXS_X86_GENERIC) float libxs_ssqrt(float x)
{
#if defined(LIBXS_INTRINSICS_X86)
  const float result = _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
  float result, y = x;
  if (LIBXS_NEQ(0, x)) {
    do {
      result = y;
      y = 0.5f * (y + x / y);
    } while (LIBXS_NEQ(result, y));
  }
  result = y;
#endif
  return result;
}


LIBXS_API unsigned int libxs_icbrt_u64(unsigned long long x)
{
  unsigned long long b; unsigned int y = 0; int s;
  for (s = 63; 0 <= s; s -= 3) {
    y += y; b = 3 * y * ((unsigned long long)y + 1) + 1;
    if (b <= (x >> s)) { x -= b << s; ++y; }
  }
  return y;
}


LIBXS_API unsigned int libxs_icbrt_u32(unsigned int x)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 30; 0 <= s; s -= 3) {
    y += y; b = 3 * y * (y + 1) + 1;
    if (b <= (x >> s)) { x -= b << s; ++y; }
  }
  return y;
}

/* Implementation based on Claude Baumann's work (http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm). */
LIBXS_API float libxs_sexp2_fast(float x, int maxiter)
{
  static const float lut[] = { /* tabulated powf(2.f, powf(2.f, -index)) */
    2.00000000f, 1.41421354f, 1.18920708f, 1.09050775f, 1.04427373f, 1.02189720f, 1.01088929f, 1.00542986f,
    1.00271130f, 1.00135469f, 1.00067711f, 1.00033855f, 1.00016928f, 1.00008464f, 1.00004232f, 1.00002110f,
    1.00001061f, 1.00000525f, 1.00000262f, 1.00000131f, 1.00000072f, 1.00000036f, 1.00000012f
  };
  const int lut_size = sizeof(lut) / sizeof(*lut), lut_size1 = lut_size - 1;
  const int *const raw = (const int*)&x;
  const int sign = (0 == (*raw & 0x80000000) ? 0 : 1);
  const int temp = *raw & 0x7FFFFFFF; /* clear sign */
  const int unbiased = (temp >> 23) - 127; /* exponent */
  const int exponent = -unbiased;
  int mantissa = (temp << 8) | 0x80000000;
  union { int i; float s; } result;
  if (lut_size1 >= exponent) {
    if (lut_size1 != exponent) { /* multiple lookups needed */
      if (7 >= unbiased) { /* not a degenerated case */
        const int n = (0 >= maxiter || lut_size1 <= maxiter) ? lut_size1 : maxiter;
        int i = 1;
        if (0 > unbiased) { /* regular/main case */
          LIBXS_ASSERT(0 <= exponent && exponent < lut_size);
          result.s = lut[exponent]; /* initial value */
          i = exponent + 1; /* next LUT offset */
        }
        else {
          result.s = 2.f; /* lut[0] */
          i = 1; /* next LUT offset */
        }
        for (; i <= n && 0 != mantissa; ++i) {
          mantissa <<= 1;
          if (0 != (mantissa & 0x80000000)) { /* check MSB */
            LIBXS_ASSERT(0 <= i && i < lut_size);
            result.s *= lut[i]; /* TODO: normalized multiply */
          }
        }
        for (i = 0; i < unbiased; ++i) { /* compute squares */
          result.s *= result.s;
        }
        if (0 != sign) { /* negative value, so reciprocal */
          result.s = 1.f / result.s;
        }
      }
      else { /* out of range */
#if defined(INFINITY)
        result.s = (0 == sign ? (INFINITY) : 0.f);
#else
        result.i = (0 == sign ? 0x7F800000 : 0);
#endif
      }
    }
    else if (0 == sign) {
      result.s = lut[lut_size1];
    }
    else { /* reciprocal */
      result.s = 1.f / lut[lut_size1];
    }
  }
  else {
    result.s = 1.f; /* case 2^0 */
  }
  return result.s;
}


LIBXS_API float libxs_sexp2(float x)
{
#if defined(LIBXS_NO_LIBM)
  return libxs_sexp2_fast(x, 20/*compromise*/);
#else
  return powf(2.f, x);
#endif
}


LIBXS_API float libxs_sexp2_u8(unsigned char x)
{
  union { int i; float s; } result;
  if (128 > x) {
    if (31 < x) {
      const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      const int n = x >> 5;
      int i;
      result.s = r32;
      for (i = 1; i < n; ++i) result.s *= r32;
      result.s *= (1U << (x - (n << 5)));
    }
    else {
      result.s = (float)(1U << x);
    }
  }
  else {
#if defined(INFINITY)
    result.s = INFINITY;
#else
    result.i = 0x7F800000;
#endif
  }
  return result.s;
}


LIBXS_API float libxs_sexp2_i8(signed char x)
{
  union { int i; float s; } result;
  if (-128 != x) {
    const signed char ux = (signed char)LIBXS_ABS(x);
    if (31 < ux) {
      const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      const int n = ux >> 5;
      int i;
      result.s = r32;
      for (i = 1; i < n; ++i) result.s *= r32;
      result.s *= (1U << (ux - (n << 5)));
    }
    else {
      result.s = (float)(1U << ux);
    }
    if (ux != x) { /* signed */
      result.s = 1.f / result.s;
    }
  }
  else {
    result.i = 0x200000;
  }
  return result.s;
}

