/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_math.h>
#include "libxs_main.h"

#include <sys/types.h>
#include <sys/stat.h>

#if !defined(LIBXS_PRODUCT_LIMIT)
# define LIBXS_PRODUCT_LIMIT 1024
#endif

#if defined(LIBXS_DEFAULT_CONFIG) || (defined(LIBXS_SOURCE_H) && !defined(LIBXS_CONFIGURED))
# if !defined(LIBXS_MATHDIFF_MHD)
#   include <libxs_mhd.h>
#   define LIBXS_MATHDIFF_MHD
# endif
#endif
#if !defined(LIBXS_MATH_DELIMS)
# define LIBXS_MATH_DELIMS " \t;,:"
#endif
#if !defined(LIBXS_MATH_ISDIR)
# if defined(S_IFDIR)
#   define LIBXS_MATH_ISDIR(MODE) 0 != ((MODE) & (S_IFDIR))
# else
#   define LIBXS_MATH_ISDIR(MODE) S_ISDIR(MODE)
# endif
#endif

/**
 * LIBXS_MATDIFF_DIV divides the numerator by the reference-denominator
 * unless the latter is zero in which case the fallback is returned.
 */
#define LIBXS_MATDIFF_DIV_DEN(A) (0 < (A) ? (A) : 1)   /* Clang: WA for div-by-zero */
#define LIBXS_MATDIFF_DIV(NUMERATOR, DENREF, FALLBACK) /* Clang: >= instead of < */ \
  (0 >= (DENREF) ? (FALLBACK) : ((NUMERATOR) / LIBXS_MATDIFF_DIV_DEN(DENREF)))


LIBXS_API int libxs_matdiff(libxs_matdiff_info_t* info,
  libxs_datatype datatype, int m, int n, const void* ref, const void* tst,
  const int* ldref, const int* ldtst)
{
  int result = EXIT_SUCCESS, result_swap = 0, result_nan = 0;
  int ldr = (NULL == ldref ? m : *ldref), ldt = (NULL == ldtst ? m : *ldtst);
  if (NULL == ref && NULL != tst) { ref = tst; tst = NULL; result_swap = 1; }
  if (NULL != ref && NULL != info && m <= ldr && m <= ldt) {
    static int matdiff_shuffle = -1; /* cache getenv result across calls */
    const size_t ntotal = (size_t)m * (size_t)n;
    int mm = m, nn = n;
    double pos_inf;
    if (0 > matdiff_shuffle) {
      const char *const env = getenv("LIBXS_MATDIFF_SHUFFLE");
      matdiff_shuffle = (NULL == env ? 0 : ('\0' != *env ? atoi(env) : 1));
    }
    if (1 == n) { mm = ldr = ldt = 1; nn = m; } /* ensure row-vector shape to standardize results */
    libxs_matdiff_clear(info);
    pos_inf = info->min_ref;
    switch ((int)datatype) {
      case LIBXS_DATATYPE_I64: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE long long
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I32: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE int
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U32: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE unsigned int
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I16: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE short
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_U16: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE unsigned short
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_I8: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) ((double)(VALUE))
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE signed char
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_F64: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE double
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_F32: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE float
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      default: {
        static int error_once = 0;
        if (0 != libxs_verbosity /* library code is expected to be mute */
          && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXS ERROR: unsupported data-type requested!\n");
        }
        result = EXIT_FAILURE;
      }
    }
    LIBXS_ASSERT((0 <= info->m && 0 <= info->n) || (0 > info->m && 0 > info->n));
    LIBXS_ASSERT(info->m < mm && info->n < nn);
    if (EXIT_SUCCESS == result) {
      const char *const env = getenv("LIBXS_DUMP");
      /*LIBXS_INIT*/
      if (NULL != env && 0 != *env && '0' != *env) {
        if ('-' != *env || (0 <= info->m && 0 <= info->n)) {
#if defined(LIBXS_MATHDIFF_MHD)
          const char *const defaultname = ((('0' < *env && '9' >= *env) || '-' == *env) ? "libxs_dump" : env);
          const int envi = atoi(env), reshape = (1 < envi || -1 > envi);
          size_t shape[2] = { 0 }, size[2] = { 0 };
          char filename[256] = "";
          libxs_mhd_element_handler_info_t info_dst;
          libxs_mhd_info_t mhd_info = { 2, 1, datatype, 0 };
          LIBXS_MEMZERO(&info_dst);
          if (0 == reshape) {
            shape[0] = (size_t)mm; shape[1] = (size_t)nn;
            size[0] = (size_t)ldr; size[1] = (size_t)nn;
          }
          else { /* reshape */
            const size_t y = (size_t)libxs_isqrt2_u32((unsigned int)ntotal);
            shape[0] = ntotal / y; shape[1] = y;
            size[0] = shape[0];
            size[1] = shape[1];
          }
          info_dst.type = LIBXS_MIN(LIBXS_DATATYPE_F32, datatype);
          LIBXS_SNPRINTF(filename, sizeof(filename), "%s-%p-ref.mhd", defaultname, ref);
          libxs_mhd_write(filename, NULL/*offset*/, shape, size, &mhd_info, ref, &info_dst,
            NULL/*handler*/, NULL/*extension_header*/,
            NULL/*extension*/, 0/*extension_size*/);
#endif
          if (NULL != tst) {
#if defined(LIBXS_MATHDIFF_MHD)
            if (0 == reshape) {
              size[0] = (size_t)ldt;
              size[1] = (size_t)nn;
            }
            LIBXS_SNPRINTF(filename, sizeof(filename), "%s-%p-tst.mhd", defaultname, ref/*adopt ref-ptr*/);
            libxs_mhd_write(filename, NULL/*offset*/, shape, size, &mhd_info, tst, &info_dst,
              NULL/*handler*/, NULL/*extension_header*/, NULL/*extension*/, 0/*extension_size*/);
#endif
            if ('-' == *env && '1' < env[1]) {
              printf("LIBXS MATDIFF (%s): m=%" PRIuPTR " n=%" PRIuPTR " ldi=%" PRIuPTR " ldo=%" PRIuPTR " failed.\n",
                libxs_typename(datatype), (uintptr_t)m, (uintptr_t)n, (uintptr_t)ldr, (uintptr_t)ldt);
            }
          }
        }
        else if ('-' == *env && '1' < env[1] && NULL != tst) {
          printf("LIBXS MATDIFF (%s): m=%" PRIuPTR " n=%" PRIuPTR " ldi=%" PRIuPTR " ldo=%" PRIuPTR " passed.\n",
            libxs_typename(datatype), (uintptr_t)m, (uintptr_t)n, (uintptr_t)ldr, (uintptr_t)ldt);
        }
      }
      if (0 == result_nan) {
        /* R-squared: l2_abs = SS_res, var_ref = SS_tot (both pre-normalization) */
        const double resrel = LIBXS_MATDIFF_DIV(info->l2_abs, info->var_ref, info->l2_abs);
        info->rsq = LIBXS_MAX(0.0, 1.0 - resrel);
        if (0 != ntotal) { /* final variance */
          info->var_ref /= ntotal;
          info->var_tst /= ntotal;
        }
        info->normf_rel = sqrt(info->normf_rel); /* sqrt(SS_res/SS_ref) = ||E||_F / ||R||_F */
        info->l2_abs = sqrt(info->l2_abs);
        info->l2_rel = sqrt(info->l2_rel);
      }
      else {
        /* in case of NaN (in test-set), initialize statistics to either Infinity or NaN */
        info->norm1_abs = info->norm1_rel = info->normi_abs = info->normi_rel = info->normf_rel
                        = info->linf_abs = info->linf_rel = info->l2_abs = info->l2_rel
                        = pos_inf;
        if (1 == result_nan) {
          info->l1_tst = info->var_tst = pos_inf;
          info->avg_tst = /*NaN*/info->v_tst;
          info->min_tst = +pos_inf;
          info->max_tst = -pos_inf;
        }
        else {
          info->l1_ref = info->var_ref = pos_inf;
          info->avg_ref = /*NaN*/info->v_ref;
          info->min_ref = +pos_inf;
          info->max_ref = -pos_inf;
        }
      }
      if (1 == n) LIBXS_ISWAP(info->m, info->n);
      if (0 != result_swap) { /* ref was NULL: move ref-stats to tst, zero ref-side */
        info->min_tst = info->min_ref;
        info->min_ref = 0;
        info->max_tst = info->max_ref;
        info->max_ref = 0;
        info->avg_tst = info->avg_ref;
        info->avg_ref = 0;
        info->var_tst = info->var_ref;
        info->var_ref = 0;
        info->l1_tst = info->l1_ref;
        info->l1_ref = 0;
        info->v_tst = info->v_ref;
        info->v_ref = 0;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API double libxs_matdiff_epsilon(const libxs_matdiff_info_t* input)
{
  double result;
  if (NULL != input) {
    const char *const matdiff_env = getenv("LIBXS_MATDIFF");
    if (0 < input->rsq) {
      result = LIBXS_MIN(input->normf_rel, input->linf_abs) / input->rsq;
    }
    else {
      const double a = LIBXS_MIN(input->norm1_abs, input->normi_abs);
      const double b = LIBXS_MAX(input->linf_abs, input->l2_abs);
      result = LIBXS_MAX(a, b);
    }
    if (NULL != matdiff_env && '\0' != *matdiff_env) {
      char buffer[4096];
      struct stat stat_info;
      const size_t envlen = strlen(matdiff_env);
      size_t offset = LIBXS_MIN(envlen + 1, sizeof(buffer));
      char *const env = strncpy(buffer, matdiff_env, sizeof(buffer) - 1);
      const char *arg, *filename = NULL;
      buffer[sizeof(buffer) - 1] = '\0'; /* ensure NUL-termination */
      arg = strtok(env, LIBXS_MATH_DELIMS);
      if (NULL != arg && 0 == stat(arg, &stat_info) && LIBXS_MATH_ISDIR(stat_info.st_mode)) {
        const int nchars = LIBXS_SNPRINTF(buffer + offset, sizeof(buffer) - offset,
          "%s/libxs_matdiff.log", arg);
        if (0 < nchars && (offset + nchars + 1) < sizeof(buffer)) {
          filename = buffer + offset;
          offset += nchars + 1;
        }
      }
      else filename = arg; /* assume file */
      if (NULL != filename) { /* bufferize output before file I/O */
        const size_t begin = offset;
        int nchars = ((2 * offset) < sizeof(buffer)
          ? LIBXS_SNPRINTF(buffer + offset, sizeof(buffer) - offset, "%.17g", result)
          : 0);
        if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) {
          offset += nchars;
          arg = strtok(NULL, LIBXS_MATH_DELIMS);
          while (NULL != arg) {
            nchars = LIBXS_SNPRINTF(buffer + offset, sizeof(buffer) - offset, " %s", arg);
            if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) offset += nchars;
            else break;
            arg = strtok(NULL, LIBXS_MATH_DELIMS);
          }
          if (NULL == arg) { /* all args consumed */
            nchars = libxs_print_cmdline(buffer + offset, sizeof(buffer) - offset, " [", "]");
            if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) {
              FILE *const file = fopen(filename, "a");
              if (NULL != file) {
                buffer[offset + nchars] = '\n'; /* replace terminator */
                fwrite(buffer + begin, 1, offset + nchars - begin + 1, file);
                fclose(file);
#if defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE) || \
   (defined(_XOPEN_SOURCE) && (500 <= _XOPEN_SOURCE))
                sync(); /* attempt to flush FS */
#endif
              }

            }
          }
        }
      }
    }
  }
  else result = 0;
  return result;
}


LIBXS_API void libxs_matdiff_reduce(libxs_matdiff_info_t* output, const libxs_matdiff_info_t* input)
{
  if (NULL != output && NULL != input && input->min_ref <= input->max_ref) {
    const double eps_out = libxs_matdiff_epsilon(output);
    const double eps_in  = libxs_matdiff_epsilon(input);
    ++output->r; /* increment reduction counter */
    /* epsilon is determined before updating the output */
    if (eps_out <= eps_in) {
      output->linf_abs = input->linf_abs;
      output->linf_rel = input->linf_rel;
      output->v_ref = input->v_ref;
      output->v_tst = input->v_tst;
      output->rsq = input->rsq;
      output->m = input->m;
      output->n = input->n;
      output->i = output->r;
    }
    else if (output->linf_abs <= input->linf_abs
          || output->linf_rel <= input->linf_rel)
    {
      output->linf_abs = LIBXS_MAX(output->linf_abs, input->linf_abs);
      output->linf_rel = LIBXS_MAX(output->linf_rel, input->linf_rel);
      output->v_ref = input->v_ref;
      output->v_tst = input->v_tst;
      output->rsq = input->rsq;
      output->m = input->m;
      output->n = input->n;
      output->i = output->r;
    }
    if (output->norm1_abs <= input->norm1_abs) {
      output->norm1_abs = input->norm1_abs;
      output->norm1_rel = input->norm1_rel;
    }
    if (output->normi_abs <= input->normi_abs) {
      output->normi_abs = input->normi_abs;
      output->normi_rel = input->normi_rel;
    }
    if (output->l2_abs <= input->l2_abs) {
      output->l2_abs = input->l2_abs;
      output->l2_rel = input->l2_rel;
    }
    if (output->normf_rel <= input->normf_rel) output->normf_rel = input->normf_rel;
    if (output->var_ref <= input->var_ref) output->var_ref = input->var_ref;
    if (output->var_tst <= input->var_tst) output->var_tst = input->var_tst;
    if (output->max_ref <= input->max_ref) output->max_ref = input->max_ref;
    if (output->max_tst <= input->max_tst) output->max_tst = input->max_tst;
    if (output->min_ref >= input->min_ref) output->min_ref = input->min_ref;
    if (output->min_tst >= input->min_tst) output->min_tst = input->min_tst;
    output->avg_ref = 0.5 * (output->avg_ref + input->avg_ref);
    output->avg_tst = 0.5 * (output->avg_tst + input->avg_tst);
    output->l1_ref += input->l1_ref;
    output->l1_tst += input->l1_tst;
  }
  else if (NULL == input || NULL == output) {
    libxs_matdiff_clear(output);
  }
}


LIBXS_API void libxs_matdiff_clear(libxs_matdiff_info_t* info)
{
  if (NULL != info) {
    const union { uint32_t raw; float value; } inf = { 0x7F800000U };
    memset(info, 0, sizeof(*info)); /* nullify */
    /* no location discovered yet with a difference */
    info->m = info->n = info->i = -1;
    /* initial minimum/maximum of reference/test */
    info->min_ref = info->min_tst = +inf.value;
    info->max_ref = info->max_tst = -inf.value;
  }
}


LIBXS_API size_t libxs_gcd(size_t a, size_t b)
{
  while (0 != b) {
    const size_t r = a % b;
    a = b; b = r;
  }
  return 0 != a ? a : 1;
}


LIBXS_API size_t libxs_lcm(size_t a, size_t b)
{
  const size_t gcd = libxs_gcd(a, b);
  return 0 != gcd ? ((a / gcd) * b) : 0;
}


LIBXS_API unsigned int libxs_remainder(unsigned int a, unsigned int b,
  const unsigned int* limit, const unsigned int* remainder)
{
  /* normalize such that a <= b */
  unsigned int ci, c;
  if (0 == b) return 0; /* guard against division by zero and infinite loop */
  ci = (b < a ? LIBXS_UP(a, b) : b); c = a * ci;
  /* sanitize limit argument */
  if (NULL != limit && (0 == b || ((*limit / b) * b) < a)) limit = NULL;
  if (1 <= a) {
    unsigned int r = a - 1;
    for (; ((NULL != remainder ? *remainder : 0) < r)
        &&  (NULL == limit || ci <= *limit); ci += b)
    {
      const unsigned int ri = ci % a;
      if (ri < r) {
        c = ci;
        r = ri;
      }
    }
  }
  return c;
}


LIBXS_API int libxs_primes_u32(unsigned int num, unsigned int num_factors_n32[])
{
  unsigned int c = num, i;
  int n = 0;
  if (0 < c && 0 == (c & 1)) { /* non-zero even */
    unsigned int j = c / 2;
    while (c == (2 * j)) {
      num_factors_n32[n++] = 2;
      c = j; j /= 2;
    }
  }
  for (i = 3; i <= c; i += 2) {
    unsigned int j = c / i;
    while (c == (i * j)) {
      num_factors_n32[n++] = i;
      c = j; j /= i;
    }
    if ((i * i) > num) {
      break;
    }
  }
  if (1 < c && 0 != n) {
    num_factors_n32[n++] = c;
  }
  return n;
}


LIBXS_API_INLINE unsigned int internal_product_limit(unsigned int product, unsigned int limit)
{
  unsigned int fact[32], maxp = limit, result = 1;
  int i, n;
  /* attempt to lower the memory requirement for DP; can miss best solution */
  if (LIBXS_PRODUCT_LIMIT < limit) {
    const unsigned int minfct = (limit + limit - 1) / LIBXS_PRODUCT_LIMIT;
    const unsigned int maxfct = (unsigned int)libxs_gcd(product, limit);
    result = maxfct;
    if (minfct < maxfct) {
      n = libxs_primes_u32(result, fact);
      for (i = 0; i < n; ++i) {
        if (minfct < fact[i]) {
          result = fact[i];
          break;
        }
      }
    }
    maxp /= result;
  }
  if (LIBXS_PRODUCT_LIMIT >= maxp) {
    unsigned int k[2][LIBXS_PRODUCT_LIMIT] = { {0} }, *k0 = k[0], *k1 = k[1], *kt, p;
    n = libxs_primes_u32(product / result, fact);
    /* initialize table with trivial factor */
    for (p = 0; p <= maxp; ++p) k[0][p] = 1;
    k[0][0] = k[1][0] = 1;
    for (i = 1; i <= n; ++i) {
      for (p = 1; p <= maxp; ++p) {
        const unsigned int f = fact[i - 1], h = k0[p];
        if (p < f) {
          k1[p] = h;
        }
        else {
          const unsigned int g = f * k0[p / f];
          k1[p] = LIBXS_MAX(g, h);
        }
      }
      kt = k0; k0 = k1; k1 = kt;
    }
    result *= k0[maxp];
  }
  else { /* trivial approximation */
    n = libxs_primes_u32(product, fact);
    for (i = 0; i < n; ++i) {
      const unsigned int f = result * fact[i];
      if (f <= limit) {
        result = f;
      }
      else break;
    }
  }
  return result;
}


LIBXS_API unsigned int libxs_product_limit(unsigned int product, unsigned int limit, int is_lower)
{
  unsigned int result;
  if (1 < limit) { /* check for fast-path */
    result = internal_product_limit(product, limit);
  }
  else {
    result = limit;
  }
  if (0 != is_lower) {
    if (limit < product) {
      if (result < limit) {
        const unsigned int limit2 = (limit <= (unsigned int)-1 / 2)
          ? (2 * limit - 1) : (unsigned int)-1;
        result = internal_product_limit(product, limit2);
      }
      if (result < limit) {
        result = product;
      }
      LIBXS_ASSERT(limit <= result);
    }
    else if (0 != product) {
      result = LIBXS_UP(limit, product);
    }
    else result = 0;
  }
  else if (product < result) {
    result = product;
  }
  LIBXS_ASSERT(0 != is_lower || result <= product);
  return result;
}


LIBXS_API size_t libxs_coprime(size_t n, size_t minco)
{
  const size_t s = (0 != (n & 1) ? ((LIBXS_MAX(minco, 1) - 1) | 1) : (minco & ~1));
  const size_t j = (0 != (n & 1) ? 1 : 2);
  size_t result = (1 < n ? 1 : 0), g = 0, h = 1, i;
  for (i = (j < n ? (n - 1) : 0); j < i; i -= j) {
    const size_t d = LIBXS_DELTA(s, i);
    size_t a = n, b = d;
    assert(i != s);
    do { /* GCD of initial A and initial B (result is in A) */
      const size_t c = a % b;
      a = b; b = c;
    } while (0 != b);
    assert(0 != d);
    if (1 == a) {
      const size_t r = n % d;
      result = d;
      if (g < r) {
        g = r;
        h = d;
      }
      if (d <= minco) {
        i = j; /* break */
      }
    }
  }
  if (minco < result) result = h;
  assert((0 == result && 1 >= n) || (result < n && 1 == libxs_gcd(result, n)));
  return result;
}


LIBXS_API size_t libxs_coprime2(size_t n)
{
  return libxs_coprime(n, libxs_isqrt_u64(n));
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


LIBXS_API unsigned int libxs_isqrt2_u32(unsigned int x)
{
  return libxs_product_limit(x, libxs_isqrt_u32(x), 0/*is_lower*/);
}


LIBXS_API double libxs_kahan_sum(double value, double* accumulator, double* compensation)
{
  double r, c;
  LIBXS_ASSERT(NULL != accumulator && NULL != compensation);
  c = value - *compensation; r = *accumulator + c;
  *compensation = (r - *accumulator) - c;
  *accumulator = r;
  return r;
}


LIBXS_API double libxs_pow2(int n)
{
  union { uint64_t u; double d; } cvt;
  if (n < -1022) return 0.0;
  if (n > 1023) {
    cvt.u = LIBXS_CONCATENATE(0x7FF0000000000000, ULL); /* +Inf */
    return cvt.d;
  }
  cvt.u = (uint64_t)(n + 1023) << 52;
  return cvt.d;
}


LIBXS_API unsigned int libxs_mod_inverse_u32(unsigned int a, unsigned int m)
{
  int t = 0, newt = 1;
  unsigned int r = m, newr = a % m;
  LIBXS_ASSERT(0 != m && 0 != a);
  while (0 != newr) {
    const unsigned int q = r / newr;
    { const int tmp = t - (int)(q) * newt; t = newt; newt = tmp; }
    { const unsigned int tmp = r - q * newr; r = newr; newr = tmp; }
  }
  LIBXS_ASSERT(1 == r); /* gcd(a, m) must be 1 */
  return (unsigned int)(t < 0 ? t + (int)m : t);
}


LIBXS_API unsigned int libxs_barrett_rcp(unsigned int p)
{
  LIBXS_ASSERT(0 != p);
  return (unsigned int)(LIBXS_CONCATENATE(0x100000000, ULL) / p);
}


LIBXS_API unsigned int libxs_barrett_pow18(unsigned int p)
{
  LIBXS_ASSERT(0 != p);
  return (unsigned int)((1UL << 18) % p);
}


LIBXS_API unsigned int libxs_barrett_pow36(unsigned int p)
{
  LIBXS_ASSERT(0 != p);
  return (unsigned int)(LIBXS_CONCATENATE(0x1000000000, ULL) % p);
}
