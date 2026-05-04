/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_math.h>
#include <libxs_malloc.h>
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
/** Relative error: DI / RA with fallback to TA for near-zero reference. */
#define LIBXS_MATDIFF_REL(DI, RA, TA) \
  LIBXS_MATDIFF_DIV(DI, ((RA) < (DI) ? 0 : (RA)), TA)

/** O(na*nb) multiset matching loop for real (scalar) element types. */
#define LIBXS_SETDIFF_REAL(TYPE, CVT) { \
  const TYPE *const ra = (const TYPE*)a, *const rb = (const TYPE*)b; \
  int u = 0, v = 0; \
  for (j = 0; j < nb; ++j) { \
    for (i = 0; i < na && u < nmin; ++i) { \
      if (LIBXS_DELTA(CVT(ra[i]), CVT(rb[j])) <= tol) { ++u; break; } \
    } \
  } \
  for (i = 0; i < na; ++i) { \
    for (j = 0; j < nb && v < nmin; ++j) { \
      if (LIBXS_DELTA(CVT(ra[i]), CVT(rb[j])) <= tol) { ++v; break; } \
    } \
  } \
  { const int m = LIBXS_MIN(LIBXS_MAX(u, v), nmin); result = nmax - m; } \
}

/** O(na*nb) multiset matching loop for complex element types. */
#define LIBXS_SETDIFF_CMPLX(TYPE, CVT) { \
  const TYPE *const ra = (const TYPE*)a, *const rb = (const TYPE*)b; \
  int u = 0, v = 0; \
  for (j = 0; j < nb; ++j) { \
    for (i = 0; i < na && u < nmin; ++i) { \
      const double dre = CVT(ra[2*i]) - CVT(rb[2*j]); \
      const double dim = CVT(ra[2*i+1]) - CVT(rb[2*j+1]); \
      if (sqrt(dre * dre + dim * dim) <= tol) { ++u; break; } \
    } \
  } \
  for (i = 0; i < na; ++i) { \
    for (j = 0; j < nb && v < nmin; ++j) { \
      const double dre = CVT(ra[2*i]) - CVT(rb[2*j]); \
      const double dim = CVT(ra[2*i+1]) - CVT(rb[2*j+1]); \
      if (sqrt(dre * dre + dim * dim) <= tol) { ++v; break; } \
    } \
  } \
  { const int m = LIBXS_MIN(LIBXS_MAX(u, v), nmin); result = nmax - m; } \
}

/** Min/max scan over a real array. */
#define LIBXS_SETDIFF_RANGE(TYPE, CVT, SRC, N, LO, HI) { \
  const TYPE *const p = (const TYPE*)(SRC); \
  int ii; \
  (LO) = (HI) = CVT(p[0]); \
  for (ii = 1; ii < (N); ++ii) { \
    const double vi = CVT(p[ii]); \
    if (vi < (LO)) (LO) = vi; \
    if (vi > (HI)) (HI) = vi; \
  } \
}

/** Min/max scan over a complex array (component-wise bounding box). */
#define LIBXS_SETDIFF_RANGE_CMPLX(TYPE, CVT, SRC, N, LO_RE, HI_RE, LO_IM, HI_IM) { \
  const TYPE *const p = (const TYPE*)(SRC); \
  int ii; \
  (LO_RE) = (HI_RE) = CVT(p[0]); \
  (LO_IM) = (HI_IM) = CVT(p[1]); \
  for (ii = 1; ii < (N); ++ii) { \
    const double re = CVT(p[2*ii]), im = CVT(p[2*ii+1]); \
    if (re < (LO_RE)) (LO_RE) = re; \
    if (re > (HI_RE)) (HI_RE) = re; \
    if (im < (LO_IM)) (LO_IM) = im; \
    if (im > (HI_IM)) (HI_IM) = im; \
  } \
}

#define LIBXS_SETDIFF_CVT(VALUE) ((double)(VALUE))
#define LIBXS_SETDIFF_NOP(VALUE) (VALUE)

#define LIBXS_MATH_MALLOC(SIZE, POOL) internal_libxs_math_malloc(SIZE, &(POOL))
#define LIBXS_MATH_FREE(PTR, POOL) internal_libxs_math_free(PTR, POOL)


/** Context for the GSS callback used by libxs_setdiff_min. */
LIBXS_EXTERN_C typedef struct internal_libxs_setdiff_ctx_t {
  const void *a, *b;
  libxs_data_t datatype;
  int na, nb;
} internal_libxs_setdiff_ctx_t;


LIBXS_API_INLINE void* internal_libxs_math_malloc(size_t size, int* pool) {
  void* p = libxs_malloc(internal_libxs_default_pool, size, LIBXS_MALLOC_AUTO);
  if (NULL != p) { *pool = 1; return p; }
  *pool = 0; return malloc(size);
}

LIBXS_API_INLINE void internal_libxs_math_free(void* ptr, int pool) {
  if (0 != pool) libxs_free(ptr); else free(ptr);
}


LIBXS_API int libxs_matdiff(libxs_matdiff_t* info,
  libxs_data_t datatype, int m, int n, const void* ref, const void* tst,
  const int* ldref, const int* ldtst)
{
  int result = EXIT_SUCCESS, result_swap = 0, result_nan = 0;
  int ldr = (NULL == ldref ? m : *ldref), ldt = (NULL == ldtst ? m : *ldtst);
  if (NULL == ref && NULL != tst) { ref = tst; tst = NULL; result_swap = 1; }
  if (NULL != ref && NULL != info && m <= ldr && m <= ldt) {
    static LIBXS_TLS int matdiff_shuffle = -1; /* cache getenv result per-thread */
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
      case LIBXS_DATATYPE_C64: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE double
#       define LIBXS_MATDIFF_COMPLEX
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_COMPLEX
#       undef LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE
#       undef LIBXS_MATDIFF_TEMPLATE_TYPE2FP64
      } break;
      case LIBXS_DATATYPE_C32: {
#       define LIBXS_MATDIFF_TEMPLATE_TYPE2FP64(VALUE) (VALUE)
#       define LIBXS_MATDIFF_TEMPLATE_ELEM_TYPE float
#       define LIBXS_MATDIFF_COMPLEX
        if (0 == matdiff_shuffle) {
#         include "libxs_matdiff.h"
        }
        else {
#         define LIBXS_MATDIFF_SHUFFLE
#         include "libxs_matdiff.h"
#         undef LIBXS_MATDIFF_SHUFFLE
        }
#       undef LIBXS_MATDIFF_COMPLEX
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
          libxs_mhd_info_t mhd_info = { 2, 1, 0/*type*/, 0 };
          libxs_mhd_write_info_t mhd_winfo = { 0 };
          LIBXS_MEMZERO(&info_dst);
          mhd_info.type = datatype;
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
          mhd_winfo.handler_info = &info_dst;
          LIBXS_SNPRINTF(filename, sizeof(filename), "%s-%p-ref.mhd", defaultname, ref);
          libxs_mhd_write(filename, NULL/*offset*/, shape, size, &mhd_info, ref, &mhd_winfo);
#endif
          if (NULL != tst) {
#if defined(LIBXS_MATHDIFF_MHD)
            if (0 == reshape) {
              size[0] = (size_t)ldt;
              size[1] = (size_t)nn;
            }
            LIBXS_SNPRINTF(filename, sizeof(filename), "%s-%p-tst.mhd", defaultname, ref/*adopt ref-ptr*/);
            libxs_mhd_write(filename, NULL/*offset*/, shape, size, &mhd_info, tst, &mhd_winfo);
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
        info->linf_rel = LIBXS_MATDIFF_REL(info->linf_abs,
          fabs(info->v_ref), fabs(info->v_tst));
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
          info->min_tst = info->diag_min_tst = +pos_inf;
          info->max_tst = info->diag_max_tst = -pos_inf;
        }
        else {
          info->l1_ref = info->var_ref = pos_inf;
          info->avg_ref = /*NaN*/info->v_ref;
          info->min_ref = info->diag_min_ref = +pos_inf;
          info->max_ref = info->diag_max_ref = -pos_inf;
        }
      }
      if (1 == n) LIBXS_ISWAP(info->m, info->n);
      if (0 != result_swap) { /* ref was NULL: move ref-stats to tst, sentinel ref-side */
        info->min_tst = info->min_ref;
        info->min_ref = +pos_inf; /* sentinel: min > max marks one-sided */
        info->max_tst = info->max_ref;
        info->max_ref = -pos_inf;
        info->diag_min_tst = info->diag_min_ref;
        info->diag_min_ref = +pos_inf;
        info->diag_max_tst = info->diag_max_ref;
        info->diag_max_ref = -pos_inf;
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


LIBXS_API double libxs_matdiff_epsilon(const libxs_matdiff_t* input)
{
  double result;
  if (NULL != input) {
    static LIBXS_TLS const char* matdiff_env_cache = (const char*)(uintptr_t)-1;
    const char* matdiff_env;
    if ((const char*)(uintptr_t)-1 == matdiff_env_cache) {
      matdiff_env_cache = getenv("LIBXS_MATDIFF");
    }
    matdiff_env = matdiff_env_cache;
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
      { /* manual tokenization (thread-safe, replaces strtok) */
        const char *p = env;
        char s[2] = {'\0'};
        while (*s = *p, NULL != strpbrk(s, LIBXS_MATH_DELIMS)) ++p; /* skip leading delims */
        arg = ('\0' != *p ? p : NULL);
        while ('\0' != *p && (*s = *p, NULL == strpbrk(s, LIBXS_MATH_DELIMS))) ++p;
        if ('\0' != *p) *(char*)(uintptr_t)p++ = '\0'; /* NUL-terminate first token */
      }
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
          const char *p2 = env + envlen; /* scan remaining tokens from end of first token */
          char s2[2] = {'\0'};
          { /* advance p2 past first token's NUL to the rest of the env copy */ const char *t = arg;
            while ('\0' != *t) ++t;
            p2 = t + 1; /* points past NUL separator into remainder */
          }
          offset += nchars;
          for (;;) {
            while (p2 < env + envlen && (*s2 = *p2, NULL != strpbrk(s2, LIBXS_MATH_DELIMS))) ++p2;
            if (p2 >= env + envlen || '\0' == *p2) break;
            arg = p2;
            while (p2 < env + envlen && '\0' != *p2 && (*s2 = *p2, NULL == strpbrk(s2, LIBXS_MATH_DELIMS))) ++p2;
            { const size_t arglen = (size_t)(p2 - arg);
              nchars = LIBXS_SNPRINTF(buffer + offset, sizeof(buffer) - offset, " %.*s", (int)arglen, arg);
            }
            if (0 < nchars && (2 * (offset + nchars)) < sizeof(buffer)) offset += nchars;
            else { arg = NULL; break; }
          }
          arg = NULL; /* all args consumed */
          { /* append command line and write log */
            nchars = internal_libxs_print_cmdline(buffer + offset, sizeof(buffer) - offset, " [", "]");
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


LIBXS_API_INLINE int internal_libxs_matdiff_onesided(const libxs_matdiff_t* info)
{
  return (NULL != info && info->min_ref > info->max_ref
                       && info->min_tst <= info->max_tst);
}


LIBXS_API int libxs_matdiff_combine(libxs_matdiff_t* output, const libxs_matdiff_t* input)
{
  int result = EXIT_SUCCESS;
  if (NULL != output && NULL != input) {
    const int lo = internal_libxs_matdiff_onesided(output);
    const int li = internal_libxs_matdiff_onesided(input);
    if (0 != lo && 0 != li) {
      libxs_matdiff_t tmp;
      libxs_matdiff_clear(&tmp);
      /* reference side: adopt tst-stats from output (lhs) */
      tmp.l1_ref  = output->l1_tst;
      tmp.min_ref = output->min_tst;
      tmp.max_ref = output->max_tst;
      tmp.avg_ref = output->avg_tst;
      tmp.var_ref = output->var_tst;
      tmp.diag_min_ref = output->diag_min_tst;
      tmp.diag_max_ref = output->diag_max_tst;
      /* test side: adopt tst-stats from input (rhs) */
      tmp.l1_tst  = input->l1_tst;
      tmp.min_tst = input->min_tst;
      tmp.max_tst = input->max_tst;
      tmp.avg_tst = input->avg_tst;
      tmp.var_tst = input->var_tst;
      tmp.diag_min_tst = input->diag_min_tst;
      tmp.diag_max_tst = input->diag_max_tst;
      /* values behind the linf_abs estimate (mean shift) */
      tmp.v_ref = tmp.avg_ref;
      tmp.v_tst = tmp.avg_tst;
      tmp.linf_abs = fabs(tmp.v_ref - tmp.v_tst);
      tmp.linf_rel = LIBXS_MATDIFF_REL(tmp.linf_abs,
        fabs(tmp.v_ref), fabs(tmp.v_tst));
      /* statistical L2 bound from pooled variance */
      tmp.l2_abs = sqrt(tmp.var_ref + tmp.var_tst);
      /* maintain reduction counter */
      tmp.r = output->r + 1;
      tmp.i = tmp.r;
      *output = tmp;
    }
    else if (0 != li || 0 != lo) {
      result = EXIT_FAILURE; /* input must be single-matrix */
    }
    /* both non-single: no-op */
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXS_API void libxs_matdiff_reduce(libxs_matdiff_t* output, const libxs_matdiff_t* input)
{
  if (NULL != output && NULL != input && 0 != internal_libxs_matdiff_onesided(input)) {
    libxs_matdiff_combine(output, input);
  }
  else if (NULL != output && NULL != input && input->min_ref <= input->max_ref) {
    const double eps_out = libxs_matdiff_epsilon(output);
    const double eps_in  = libxs_matdiff_epsilon(input);
    ++output->r; /* increment reduction counter */
    /* epsilon is determined before updating the output */
    if (eps_out <= eps_in) {
      output->v_ref = input->v_ref;
      output->v_tst = input->v_tst;
      output->m = input->m;
      output->n = input->n;
      output->i = output->r;
    }
    else if (output->linf_abs <= input->linf_abs) {
      output->v_ref = input->v_ref;
      output->v_tst = input->v_tst;
      output->m = input->m;
      output->n = input->n;
      output->i = output->r;
    }
    if (output->rsq > input->rsq) output->rsq = input->rsq;
    { /* derive linf_abs/linf_rel from v_ref/v_tst */
      output->linf_abs = fabs(output->v_ref - output->v_tst);
      output->linf_rel = LIBXS_MATDIFF_REL(output->linf_abs,
        fabs(output->v_ref), fabs(output->v_tst));
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
    if (output->diag_max_ref <= input->diag_max_ref) output->diag_max_ref = input->diag_max_ref;
    if (output->diag_max_tst <= input->diag_max_tst) output->diag_max_tst = input->diag_max_tst;
    if (output->diag_min_ref >= input->diag_min_ref) output->diag_min_ref = input->diag_min_ref;
    if (output->diag_min_tst >= input->diag_min_tst) output->diag_min_tst = input->diag_min_tst;
    output->avg_ref = 0.5 * (output->avg_ref + input->avg_ref);
    output->avg_tst = 0.5 * (output->avg_tst + input->avg_tst);
    output->l1_ref += input->l1_ref;
    output->l1_tst += input->l1_tst;
  }
  else if (NULL == input || NULL == output) {
    libxs_matdiff_clear(output);
  }
}


LIBXS_API void libxs_matdiff_clear(libxs_matdiff_t* info)
{
  if (NULL != info) {
    const union { uint32_t raw; float value; } inf = { 0x7F800000U };
    memset(info, 0, sizeof(*info)); /* nullify */
    /* no location discovered yet with a difference */
    info->m = info->n = info->i = -1;
    /* initial minimum/maximum of reference/test */
    info->min_ref = info->min_tst = +inf.value;
    info->max_ref = info->max_tst = -inf.value;
    info->diag_min_ref = info->diag_min_tst = +inf.value;
    info->diag_max_ref = info->diag_max_tst = -inf.value;
    info->rsq = 1.0; /* identity for min-reduction */
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


LIBXS_API int libxs_primes_u32(unsigned int num, unsigned int num_factors_n32[], int num_factors_max)
{
  unsigned int c = num, i;
  int n = 0;
  if (0 < c && 0 == (c & 1)) { /* non-zero even */
    unsigned int j = c / 2;
    while (c == (2 * j)) {
      if (n < num_factors_max) num_factors_n32[n] = 2;
      ++n;
      c = j; j /= 2;
    }
  }
  for (i = 3; i <= c; i += 2) {
    unsigned int j = c / i;
    while (c == (i * j)) {
      if (n < num_factors_max) num_factors_n32[n] = i;
      ++n;
      c = j; j /= i;
    }
    if ((i * i) > num) {
      break;
    }
  }
  if (1 < c && 0 != n) {
    if (n < num_factors_max) num_factors_n32[n] = c;
    ++n;
  }
  return n;
}


LIBXS_API_INLINE unsigned int internal_libxs_product_limit(unsigned int product, unsigned int limit)
{
  unsigned int fact[32], maxp = limit, result = 1;
  int i, n;
  /* attempt to lower the memory requirement for DP; can miss best solution */
  if (LIBXS_PRODUCT_LIMIT < limit) {
    const unsigned int minfct = (limit + limit - 1) / LIBXS_PRODUCT_LIMIT;
    const unsigned int maxfct = (unsigned int)libxs_gcd(product, limit);
    result = maxfct;
    if (minfct < maxfct) {
      n = libxs_primes_u32(result, fact, 32);
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
    n = libxs_primes_u32(product / result, fact, 32);
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
    n = libxs_primes_u32(product, fact, 32);
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
    result = internal_libxs_product_limit(product, limit);
  }
  else {
    result = limit;
  }
  if (0 != is_lower) {
    if (limit < product) {
      if (result < limit) {
        const unsigned int limit2 = (limit <= (unsigned int)-1 / 2)
          ? (2 * limit - 1) : (unsigned int)-1;
        result = internal_libxs_product_limit(product, limit2);
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


LIBXS_API size_t libxs_coprime_bias(size_t n, double bias)
{
  const size_t sqrtn = libxs_isqrt_u64(n);
  const size_t half = n / 2;
  size_t target, d;
  if (n <= 4) return libxs_coprime(n, sqrtn);
  bias = LIBXS_CLMP(bias, -1.0, 1.0);
  if (bias < 0.0) target = (size_t)(pow((double)sqrtn, 1.0 + bias) + 0.5);
  else if (bias <= 0.0) target = sqrtn;
  else target = (size_t)(pow((double)n, 0.5 * (1.0 + bias)) + 0.5);
  target = LIBXS_CLMP(target, 2, half);
  for (d = target; d >= 2; --d) {
    if (1 == libxs_gcd(d, n)) return d;
  }
  for (d = target + 1; d <= half; ++d) {
    if (1 == libxs_gcd(d, n)) return d;
  }
  return libxs_coprime(n, sqrtn);
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


LIBXS_API double libxs_gss_min(
  double (*fn)(double x, const void* data), const void* data,
  double x0, double x1, double* xmin, int maxiter)
{
  const double phi = (sqrt(5.0) - 1.0) * 0.5;
  double b0 = x0, b1 = x1, d = b1 - b0;
  double c0 = b0 + (1.0 - phi) * d;
  double c1 = b0 + phi * d;
  double f0, f1;
  int n;
  LIBXS_ASSERT(NULL != fn && x0 <= x1 && 0 < maxiter);
  f0 = fn(c0, data); f1 = fn(c1, data);
  for (n = 0; n < maxiter && b0 != c0 && b1 != c1; ++n) {
    if (f0 <= f1) {
      b1 = c1; c1 = c0; f1 = f0;
      d = b1 - b0;
      c0 = b0 + (1.0 - phi) * d;
      f0 = fn(c0, data);
    }
    else {
      b0 = c0; c0 = c1; f0 = f1;
      d = b1 - b0;
      c1 = b0 + phi * d;
      f1 = fn(c1, data);
    }
  }
  if (f0 <= f1) {
    if (NULL != xmin) *xmin = c0;
    return f0;
  }
  else {
    if (NULL != xmin) *xmin = c1;
    return f1;
  }
}


LIBXS_API int libxs_setdiff(
  libxs_data_t datatype, const void* a, int na,
  const void* b, int nb, double tol)
{
  const int nmin = LIBXS_MIN(na, nb);
  const int nmax = LIBXS_MAX(na, nb);
  int result = -1, i, j;
  LIBXS_ASSERT(NULL != a && NULL != b && 0 <= na && 0 <= nb && 0 <= tol);
  switch ((int)datatype) {
    case LIBXS_DATATYPE_F64: LIBXS_SETDIFF_REAL(double, LIBXS_SETDIFF_NOP) break;
    case LIBXS_DATATYPE_F32: LIBXS_SETDIFF_REAL(float, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_C64: LIBXS_SETDIFF_CMPLX(double, LIBXS_SETDIFF_NOP) break;
    case LIBXS_DATATYPE_C32: LIBXS_SETDIFF_CMPLX(float, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_I64: LIBXS_SETDIFF_REAL(long long, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_I32: LIBXS_SETDIFF_REAL(int, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_U32: LIBXS_SETDIFF_REAL(unsigned int, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_I16: LIBXS_SETDIFF_REAL(short, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_U16: LIBXS_SETDIFF_REAL(unsigned short, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_I8:  LIBXS_SETDIFF_REAL(signed char, LIBXS_SETDIFF_CVT) break;
    case LIBXS_DATATYPE_U8:  LIBXS_SETDIFF_REAL(unsigned char, LIBXS_SETDIFF_CVT) break;
    default: {
      static int error_once = 0;
      if (0 != libxs_verbosity
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: libxs_setdiff unsupported data-type!\n");
      }
    }
  }
  LIBXS_ASSERT(0 <= result || LIBXS_DATATYPE_UNKNOWN == datatype);
  return result;
}


LIBXS_API_INTERN double internal_libxs_setdiff_fn(double tol, const void* data)
{
  const internal_libxs_setdiff_ctx_t *const ctx =
    (const internal_libxs_setdiff_ctx_t*)data;
  return (double)libxs_setdiff(ctx->datatype,
    ctx->a, ctx->na, ctx->b, ctx->nb, tol);
}


LIBXS_API_INLINE double internal_libxs_setdiff_range(
  libxs_data_t datatype, const void* a, int na,
  const void* b, int nb)
{
  double mina, maxa, minb, maxb;
  switch ((int)datatype) {
    case LIBXS_DATATYPE_F64: LIBXS_SETDIFF_RANGE(double, LIBXS_SETDIFF_NOP, a, na, mina, maxa)
                              LIBXS_SETDIFF_RANGE(double, LIBXS_SETDIFF_NOP, b, nb, minb, maxb) break;
    case LIBXS_DATATYPE_F32: LIBXS_SETDIFF_RANGE(float, LIBXS_SETDIFF_CVT, a, na, mina, maxa)
                              LIBXS_SETDIFF_RANGE(float, LIBXS_SETDIFF_CVT, b, nb, minb, maxb) break;
    case LIBXS_DATATYPE_C64: case LIBXS_DATATYPE_C32: {
      double a_rlo, a_rhi, a_ilo, a_ihi, b_rlo, b_rhi, b_ilo, b_ihi, dre, dim;
      if (LIBXS_DATATYPE_C64 == (int)datatype) {
        LIBXS_SETDIFF_RANGE_CMPLX(double, LIBXS_SETDIFF_NOP, a, na, a_rlo, a_rhi, a_ilo, a_ihi)
        LIBXS_SETDIFF_RANGE_CMPLX(double, LIBXS_SETDIFF_NOP, b, nb, b_rlo, b_rhi, b_ilo, b_ihi)
      }
      else {
        LIBXS_SETDIFF_RANGE_CMPLX(float, LIBXS_SETDIFF_CVT, a, na, a_rlo, a_rhi, a_ilo, a_ihi)
        LIBXS_SETDIFF_RANGE_CMPLX(float, LIBXS_SETDIFF_CVT, b, nb, b_rlo, b_rhi, b_ilo, b_ihi)
      }
      dre = LIBXS_MAX(LIBXS_DELTA(a_rlo, b_rhi), LIBXS_DELTA(b_rlo, a_rhi));
      dim = LIBXS_MAX(LIBXS_DELTA(a_ilo, b_ihi), LIBXS_DELTA(b_ilo, a_ihi));
      return sqrt(dre * dre + dim * dim);
    }
    default: return 0;
  }
  { const double d0 = LIBXS_DELTA(mina, maxb), d1 = LIBXS_DELTA(minb, maxa);
    return LIBXS_MAX(d0, d1);
  }
}


LIBXS_API int libxs_setdiff_min(
  libxs_data_t datatype, const void* a, int na,
  const void* b, int nb, double* tol)
{
  LIBXS_ASSERT(NULL != a && NULL != b && 0 <= na && 0 <= nb);
  if (0 < na && 0 < nb && LIBXS_ENUM_IS_FLOAT((int)datatype)) {
    internal_libxs_setdiff_ctx_t ctx;
    const double x1 = internal_libxs_setdiff_range(datatype, a, na, b, nb);
    ctx.datatype = datatype;
    ctx.a = a; ctx.b = b;
    ctx.na = na; ctx.nb = nb;
    return (int)libxs_gss_min(
      internal_libxs_setdiff_fn, &ctx, 0.0, x1, tol, 10000);
  }
  else {
    if (NULL != tol) *tol = 0;
    return LIBXS_MAX(na, nb);
  }
}

#undef LIBXS_SETDIFF_NOP
#undef LIBXS_SETDIFF_CVT
#undef LIBXS_SETDIFF_RANGE_CMPLX
#undef LIBXS_SETDIFF_RANGE
#undef LIBXS_SETDIFF_CMPLX
#undef LIBXS_SETDIFF_REAL


/** Compute fingerprint from double values already in cur[0..n-1].
 * buf must point to an allocation of at least 2*n doubles;
 * cur must equal buf or buf+n. */
LIBXS_API_INTERN void internal_libxs_fprint_core(
  libxs_fprint_t* info, double* buf, double* cur, int n, int kmax)
{
  const double h = 1 < n ? 1.0 / (n - 1) : 1.0;
  double *prv = (cur == buf) ? buf + n : buf;
  int k, i;
  { double l2acc = 0, l2comp = 0, l1acc = 0, l1comp = 0, amax = 0;
    for (i = 0; i < n; ++i) {
      const double a = cur[i] < 0 ? -cur[i] : cur[i];
      libxs_kahan_sum(cur[i] * cur[i], &l2acc, &l2comp);
      libxs_kahan_sum(a, &l1acc, &l1comp);
      if (a > amax) amax = a;
    }
    info->l2[0] = sqrt(l2acc * h);
    info->l1[0] = l1acc * h;
    info->linf[0] = amax;
  }
  for (k = 1; k <= kmax; ++k) {
    const int nk = n - k;
    double *tmp, l2acc = 0, l2comp = 0, l1acc = 0, l1comp = 0, amax = 0;
    tmp = prv; prv = cur; cur = tmp;
    for (i = 0; i < nk; ++i) cur[i] = (prv[i + 1] - prv[i]) / h;
    for (i = 0; i < nk; ++i) {
      const double a = cur[i] < 0 ? -cur[i] : cur[i];
      libxs_kahan_sum(cur[i] * cur[i], &l2acc, &l2comp);
      libxs_kahan_sum(a, &l1acc, &l1comp);
      if (a > amax) amax = a;
    }
    info->l2[k] = sqrt(l2acc * h);
    info->l1[k] = l1acc * h;
    info->linf[k] = amax;
  }
}


/** Convert typed data with stride to double array dst[0..n-1]. */
#define LIBXS_FPRINT_LOAD(TYPE, SRC, STRIDE, N, DST) { \
  const TYPE *const p = (const TYPE*)(SRC); \
  const size_t s = (STRIDE); int ii; \
  for (ii = 0; ii < (N); ++ii) (DST)[ii] = (double)p[ii * s]; \
}

LIBXS_API_INTERN int internal_libxs_fprint_load(
  double* dst, libxs_data_t datatype,
  const void* data, size_t stride, int n)
{
  switch ((int)datatype) {
    case LIBXS_DATATYPE_F64: LIBXS_FPRINT_LOAD(double, data, stride, n, dst) break;
    case LIBXS_DATATYPE_F32: LIBXS_FPRINT_LOAD(float, data, stride, n, dst) break;
    case LIBXS_DATATYPE_I64: LIBXS_FPRINT_LOAD(long long, data, stride, n, dst) break;
    case LIBXS_DATATYPE_I32: LIBXS_FPRINT_LOAD(int, data, stride, n, dst) break;
    case LIBXS_DATATYPE_U32: LIBXS_FPRINT_LOAD(unsigned int, data, stride, n, dst) break;
    case LIBXS_DATATYPE_I16: LIBXS_FPRINT_LOAD(short, data, stride, n, dst) break;
    case LIBXS_DATATYPE_U16: LIBXS_FPRINT_LOAD(unsigned short, data, stride, n, dst) break;
    case LIBXS_DATATYPE_I8:  LIBXS_FPRINT_LOAD(signed char, data, stride, n, dst) break;
    case LIBXS_DATATYPE_U8:  LIBXS_FPRINT_LOAD(unsigned char, data, stride, n, dst) break;
    default: return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

#undef LIBXS_FPRINT_LOAD


LIBXS_API int libxs_fprint(libxs_fprint_t* info,
  libxs_data_t datatype, const void* data,
  int ndims, const size_t shape[], const size_t stride[],
  int order, int axis)
{
  int result = EXIT_SUCCESS;
  LIBXS_ASSERT(NULL != info && NULL != data && 0 < ndims);
  LIBXS_ASSERT(NULL != shape);
  memset(info, 0, sizeof(*info));
  info->datatype = datatype;
  if (LIBXS_DATATYPE_UNKNOWN == (int)datatype) {
    static const libxs_data_t probes[] = {
      LIBXS_DATATYPE_F64, LIBXS_DATATYPE_F32,
      LIBXS_DATATYPE_I64, LIBXS_DATATYPE_U64,
      LIBXS_DATATYPE_I32, LIBXS_DATATYPE_U32,
      LIBXS_DATATYPE_I16, LIBXS_DATATYPE_U16,
      LIBXS_DATATYPE_I8,  LIBXS_DATATYPE_U8
    };
    const size_t nbytes = shape[0];
    double best_decay = 1e30;
    libxs_data_t best_type = LIBXS_DATATYPE_U8;
    int found = 0, p;
    for (p = 0; p < (int)(sizeof(probes) / sizeof(*probes)); ++p) {
      const size_t tw = LIBXS_TYPESIZE((int)probes[p]);
      if (0 != tw && 0 == nbytes % tw && nbytes / tw > 1) {
        libxs_fprint_t fp;
        const size_t ne = nbytes / tw;
        int r = libxs_fprint(&fp, probes[p], data, 1, &ne, NULL, order, axis);
        if (EXIT_SUCCESS == r && fp.l2[0] == fp.l2[0] && 0 < fp.l2[0]
          && fp.linf[0] == fp.linf[0])
        {
          if (LIBXS_ENUM_IS_FLOAT(probes[p]) && fp.linf[0] < 1e-37) continue;
          { const double decay = libxs_fprint_decay(&fp);
            if (decay == decay
              && (decay < best_decay
                || (decay == best_decay
                  && tw < (size_t)LIBXS_TYPESIZE((int)best_type))))
            {
              best_decay = decay;
              best_type = probes[p];
              *info = fp;
              found = 1;
            }
          }
        }
      }
    }
    if (0 != found && best_decay == best_decay) {
      info->datatype = best_type;
      return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
  }
  if (0 <= axis && axis < ndims && 1 < ndims) {
    /* Per-axis mode: fingerprint along 'axis', max-reduce over others. */
    const size_t typesize = LIBXS_TYPESIZE((int)datatype);
    const size_t n_axis = shape[axis];
    size_t s_axis;
    size_t ngrid = 1;
    int d;
    if (NULL != stride) {
      s_axis = stride[axis];
    }
    else {
      s_axis = 1;
      for (d = 0; d < axis; ++d) s_axis *= shape[d];
    }
    if (1 > (int)n_axis) return EXIT_SUCCESS;
    for (d = 0; d < ndims; ++d) {
      if (d != axis) ngrid *= shape[d];
    }
    { /* Iterate over all positions in the non-axis grid. */
      size_t gi;
      int first = 1;
      for (gi = 0; gi < ngrid && EXIT_SUCCESS == result; ++gi) {
        /* Compute byte offset for grid position gi (mixed-radix). */
        size_t offset = 0, rem = gi;
        for (d = ndims - 1; d >= 0; --d) {
          size_t sd, coord;
          if (d == axis) continue;
          coord = rem % shape[d];
          rem /= shape[d];
          if (NULL != stride) sd = stride[d];
          else { int dd; sd = 1; for (dd = 0; dd < d; ++dd) sd *= shape[dd]; }
          offset += coord * sd * typesize;
        }
        { /* 1D fprint along axis at this grid position. */
          libxs_fprint_t fp1;
          result = libxs_fprint(&fp1, datatype, (const char*)data + offset,
            1, &n_axis, &s_axis, order, -1);
          if (EXIT_SUCCESS == result) {
            int k;
            if (0 != first) { *info = fp1; first = 0; }
            else {
              for (k = 0; k <= fp1.order; ++k) {
                if (fp1.linf[k] > info->linf[k]) info->linf[k] = fp1.linf[k];
                if (fp1.l2[k] > info->l2[k]) info->l2[k] = fp1.l2[k];
                if (fp1.l1[k] > info->l1[k]) info->l1[k] = fp1.l1[k];
              }
              if (fp1.order > info->order) info->order = fp1.order;
              if (fp1.n > info->n) info->n = fp1.n;
            }
          }
        }
      }
    }
  }
  else if (1 == ndims) {
    const size_t s = (NULL != stride ? stride[0] : 1);
    const int n = (int)shape[0];
    int kmax, pool = 0;
    double *buf, *cur;
    if (1 > n) return EXIT_SUCCESS;
    kmax = LIBXS_MIN(order, n - 1);
    kmax = LIBXS_MIN(kmax, LIBXS_FPRINT_MAXORDER);
    if (0 > kmax) kmax = 0;
    info->order = kmax; info->n = n;
    buf = (double*)LIBXS_MATH_MALLOC(2 * (size_t)n * sizeof(double), pool);
    if (NULL == buf) return EXIT_FAILURE;
    cur = buf;
    result = internal_libxs_fprint_load(cur, datatype, data, s, n);
    if (EXIT_SUCCESS != result) {
      static int error_once = 0;
      if (0 != libxs_verbosity
        && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXS ERROR: libxs_fprint unsupported data-type!\n");
      }
    }
    else {
      internal_libxs_fprint_core(info, buf, cur, n, kmax);
    }
    LIBXS_MATH_FREE(buf, pool);
  }
  else {
    /* Hierarchical mode (axis < 0 or axis >= ndims). */
    const size_t nouter = shape[ndims - 1];
    size_t souter;
    const size_t typesize = LIBXS_TYPESIZE((int)datatype);
    size_t j;
    libxs_fprint_t child;
    double *scalars, *buf, *cur;
    int kmax, pool_s = 0, pool_b = 0;
    if (NULL != stride) {
      souter = stride[ndims - 1];
    }
    else {
      size_t p = 1;
      int dd;
      for (dd = 0; dd + 1 < ndims; ++dd) p *= shape[dd];
      souter = p;
    }
    if (1 > (int)nouter) return EXIT_SUCCESS;
    scalars = (double*)LIBXS_MATH_MALLOC(nouter * sizeof(double), pool_s);
    if (NULL == scalars) return EXIT_FAILURE;
    for (j = 0; j < nouter && EXIT_SUCCESS == result; ++j) {
      const void* slice = (const char*)data + j * souter * typesize;
      double snorm = 0, scomp = 0, wk = 1.0;
      int k;
      result = libxs_fprint(&child, datatype, slice,
        ndims - 1, shape, stride, order, axis);
      for (k = 0; k <= child.order; ++k) {
        if (0 < k) wk /= k;
        libxs_kahan_sum(wk * child.l2[k] * child.l2[k], &snorm, &scomp);
      }
      scalars[j] = sqrt(snorm);
    }
    if (EXIT_SUCCESS == result) {
      kmax = LIBXS_MIN(order, (int)nouter - 1);
      kmax = LIBXS_MIN(kmax, LIBXS_FPRINT_MAXORDER);
      if (0 > kmax) kmax = 0;
      info->order = kmax; info->n = (int)nouter;
      buf = (double*)LIBXS_MATH_MALLOC(2 * nouter * sizeof(double), pool_b);
      if (NULL != buf) {
        cur = buf;
        for (j = 0; j < nouter; ++j) cur[j] = scalars[j];
        internal_libxs_fprint_core(info, buf, cur, (int)nouter, kmax);
        LIBXS_MATH_FREE(buf, pool_b);
      }
      else result = EXIT_FAILURE;
    }
    LIBXS_MATH_FREE(scalars, pool_s);
  }
  return result;
}


LIBXS_API double libxs_fprint_diff(
  const libxs_fprint_t* a, const libxs_fprint_t* b,
  const double weights[])
{
  int k, kmax;
  double acc = 0, comp = 0, wk = 1.0;
  LIBXS_ASSERT(NULL != a && NULL != b);
  kmax = LIBXS_MIN(a->order, b->order);
  for (k = 0; k <= kmax; ++k) {
    const double d = a->l2[k] - b->l2[k];
    if (NULL != weights) wk = weights[k];
    else if (0 < k) wk /= k; /* 1/k! */
    libxs_kahan_sum(wk * d * d, &acc, &comp);
  }
  return sqrt(acc);
}


LIBXS_API double libxs_fprint_decay(const libxs_fprint_t* info)
{
  LIBXS_ASSERT(NULL != info);
  if (0 < info->order && 0 < info->l2[0] && 1 < info->n) {
    const int k = info->order;
    return pow(info->l2[k] / info->l2[0], 1.0 / k) / (info->n - 1);
  }
  return 1e30;
}
