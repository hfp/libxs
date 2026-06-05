LIBXS_API int libxs_setdiff(
  libxs_data_t datatype, const void* a, int na,
  const void* b, int nb, double tol)
{
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
  const int nmax = LIBXS_MAX(ctx->na, ctx->nb);
  if (NULL != ctx->sa) {
    return (double)(nmax - internal_libxs_setdiff_merge(
      ctx->sa, ctx->na, ctx->sb, ctx->nb, tol));
  }
  else if (NULL != ctx->pts) {
    return (double)(nmax - internal_libxs_setdiff_kd_match(
      ctx->pts, ctx->idx, ctx->nb, ctx->qa, ctx->na, tol));
  }
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
  int result;
  LIBXS_ASSERT(NULL != a && NULL != b && 0 <= na && 0 <= nb);
  if (0 < na && 0 < nb && LIBXS_ENUM_IS_FLOAT((int)datatype)) {
    internal_libxs_setdiff_ctx_t ctx;
    const double x1 = internal_libxs_setdiff_range(datatype, a, na, b, nb);
    void* buf = NULL;
    int pool = 0;
    ctx.datatype = datatype;
    ctx.a = a; ctx.b = b;
    ctx.sa = NULL; ctx.sb = NULL;
    ctx.pts = NULL; ctx.qa = NULL; ctx.idx = NULL;
    ctx.na = na; ctx.nb = nb;
    if (LIBXS_DATATYPE_F64 == (int)datatype
      || LIBXS_DATATYPE_F32 == (int)datatype)
    {
      const size_t bufsz = ((size_t)na + (size_t)nb) * sizeof(double);
      buf = LIBXS_MATH_MALLOC(bufsz, pool);
      if (NULL != buf) {
        double* sa = (double*)buf, *sb = sa + na;
        int i;
        if (LIBXS_DATATYPE_F64 == (int)datatype) {
          const double* ra = (const double*)a;
          const double* rb = (const double*)b;
          for (i = 0; i < na; ++i) sa[i] = ra[i];
          for (i = 0; i < nb; ++i) sb[i] = rb[i];
        }
        else {
          const float* ra = (const float*)a;
          const float* rb = (const float*)b;
          for (i = 0; i < na; ++i) sa[i] = (double)ra[i];
          for (i = 0; i < nb; ++i) sb[i] = (double)rb[i];
        }
        libxs_sort(sa, na, sizeof(double), libxs_cmp_f64, NULL);
        libxs_sort(sb, nb, sizeof(double), libxs_cmp_f64, NULL);
        ctx.sa = sa; ctx.sb = sb;
      }
    }
    else if (LIBXS_DATATYPE_C64 == (int)datatype
      || LIBXS_DATATYPE_C32 == (int)datatype)
    {
      const size_t bufsz = (size_t)nb * 2 * sizeof(double)
        + (size_t)nb * sizeof(int)
        + (size_t)na * 2 * sizeof(double);
      buf = LIBXS_MATH_MALLOC(bufsz, pool);
      if (NULL != buf) {
        double* pts = (double*)buf;
        int* idx = (int*)(pts + 2 * nb);
        double* qa = (double*)(idx + nb);
        int i;
        if (LIBXS_DATATYPE_C64 == (int)datatype) {
          const double* ra = (const double*)a;
          const double* rb = (const double*)b;
          for (i = 0; i < nb; ++i) {
            pts[2*i] = rb[2*i]; pts[2*i+1] = rb[2*i+1];
          }
          for (i = 0; i < na; ++i) {
            qa[2*i] = ra[2*i]; qa[2*i+1] = ra[2*i+1];
          }
        }
        else {
          const float* ra = (const float*)a;
          const float* rb = (const float*)b;
          for (i = 0; i < nb; ++i) {
            pts[2*i] = (double)rb[2*i]; pts[2*i+1] = (double)rb[2*i+1];
          }
          for (i = 0; i < na; ++i) {
            qa[2*i] = (double)ra[2*i]; qa[2*i+1] = (double)ra[2*i+1];
          }
        }
        for (i = 0; i < nb; ++i) idx[i] = i;
        libxs_kdtree2d_build(pts, idx, nb);
        ctx.pts = pts; ctx.idx = idx; ctx.qa = qa;
      }
    }
    { const double fmin = internal_libxs_setdiff_fn(x1, &ctx);
      result = (int)libxs_bisect_min(
        internal_libxs_setdiff_fn, &ctx, 0.0, x1, fmin, tol, 10000, 0.0, NULL);
    }
    if (NULL != buf) {
      LIBXS_MATH_FREE(buf, pool);
    }
  }
  else {
    if (NULL != tol) *tol = 0;
    result = LIBXS_MAX(na, nb);
  }
  return result;
}

#undef LIBXS_SETDIFF_NOP
#undef LIBXS_SETDIFF_CVT
#undef LIBXS_SETDIFF_RANGE_CMPLX
#undef LIBXS_SETDIFF_RANGE
#undef LIBXS_SETDIFF_CMPLX
#undef LIBXS_SETDIFF_REAL


