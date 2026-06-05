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
      info->w = (double)ntotal;
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
      /* maintain reduction counter and cumulative weight */
      tmp.w = output->w + input->w;
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
    output->w += input->w;
    if (0 < output->w) {
      output->avg_ref += input->w * (input->avg_ref - output->avg_ref) / output->w;
      output->avg_tst += input->w * (input->avg_tst - output->avg_tst) / output->w;
    }
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
