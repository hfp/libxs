/** Compute fingerprint from double values already in cur[0..n-1].
 * buf must point to an allocation of at least 2*n doubles;
 * cur must equal buf or buf+n. */
LIBXS_API_INTERN void internal_libxs_fprint_core(
  libxs_fprint_t* info, double* buf, double* cur, int n, int kmax)
{
  const double h = 1 < n ? 1.0 / (n - 1) : 1.0;
  double *prv = (cur == buf) ? buf + n : buf;
  int k, i;
  { double l2acc = 0, l2comp = 0, l1acc = 0, l1comp = 0;
    double macc = 0, mcomp = 0, amax = 0;
    for (i = 0; i < n; ++i) {
      const double a = cur[i] < 0 ? -cur[i] : cur[i];
      libxs_kahan_sum(cur[i] * cur[i], &l2acc, &l2comp);
      libxs_kahan_sum(cur[i], &macc, &mcomp);
      libxs_kahan_sum(a, &l1acc, &l1comp);
      if (a > amax) amax = a;
    }
    info->l2[0] = sqrt(l2acc * h);
    info->l1[0] = l1acc * h;
    info->linf[0] = amax;
    info->mean[0] = macc / n;
  }
  for (k = 1; k <= kmax; ++k) {
    const int nk = n - k;
    double *tmp, l2acc = 0, l2comp = 0, l1acc = 0, l1comp = 0;
    double macc = 0, mcomp = 0, amax = 0;
    tmp = prv; prv = cur; cur = tmp;
    for (i = 0; i < nk; ++i) cur[i] = (prv[i + 1] - prv[i]) / h;
    for (i = 0; i < nk; ++i) {
      const double a = cur[i] < 0 ? -cur[i] : cur[i];
      libxs_kahan_sum(cur[i] * cur[i], &l2acc, &l2comp);
      libxs_kahan_sum(cur[i], &macc, &mcomp);
      libxs_kahan_sum(a, &l1acc, &l1comp);
      if (a > amax) amax = a;
    }
    info->l2[k] = sqrt(l2acc * h);
    info->l1[k] = l1acc * h;
    info->linf[k] = amax;
    info->mean[k] = macc / nk;
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
                info->mean[k] += fp1.mean[k];
              }
              if (fp1.order > info->order) info->order = fp1.order;
              if (fp1.n > info->n) info->n = fp1.n;
            }
          }
        }
      }
      if (EXIT_SUCCESS == result && 1 < ngrid) {
        int k;
        for (k = 0; k <= info->order; ++k) info->mean[k] /= (double)ngrid;
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
    const double dl2 = a->l2[k] - b->l2[k];
    const double dm = a->mean[k] - b->mean[k];
    if (NULL != weights) wk = weights[k];
    else if (0 < k) wk /= k; /* 1/k! */
    libxs_kahan_sum(wk * (dl2 * dl2 + dm * dm), &acc, &comp);
  }
  return sqrt(acc);
}


LIBXS_API double libxs_fprint_raw(
  const libxs_fprint_t* info, int k, double value)
{
  if (0 < k && 1 < info->n) {
    const double h = 1.0 / (info->n - 1);
    int i;
    for (i = 0; i < k; ++i) value *= h;
  }
  return value;
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


LIBXS_API double libxs_binom(double t, int k)
{
  double result = 1.0;
  int i;
  for (i = 0; i < k; ++i) result *= (t - i) / (i + 1);
  return result;
}


LIBXS_API double libxs_dist2(const double* a, const double* b, int n)
{
  double d = 0;
  int i;
  for (i = 0; i < n; ++i) {
    const double di = a[i] - b[i];
    d += di * di;
  }
  return d;
}
