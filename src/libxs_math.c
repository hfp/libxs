/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_math.h>
#include <libxs/libxs_perm.h>
#include <libxs/libxs_malloc.h>
#include "libxs_main.h"

#include <sys/types.h>
#include <sys/stat.h>

#if !defined(LIBXS_PRODUCT_LIMIT)
# define LIBXS_PRODUCT_LIMIT 1024
#endif

#if defined(LIBXS_DEFAULT_CONFIG) || (defined(LIBXS_SOURCE_H) && !defined(LIBXS_CONFIGURED))
# if !defined(LIBXS_MATHDIFF_MHD)
#   include <libxs/libxs_mhd.h>
#   define LIBXS_MATHDIFF_MHD
# endif
#endif
#if !defined(LIBXS_MATH_DELIMS)
# define LIBXS_MATH_DELIMS " \t" LIBXS_DELIMS
#endif
#if !defined(LIBXS_MATH_ISDIR)
# if defined(S_IFDIR)
#   define LIBXS_MATH_ISDIR(MODE) 0 != ((MODE) & (S_IFDIR))
# else
#   define LIBXS_MATH_ISDIR(MODE) S_ISDIR(MODE)
# endif
#endif

#define LIBXS_MATH_MALLOC(SIZE, POOL) internal_libxs_scratch_malloc(SIZE, &(POOL))
#define LIBXS_MATH_FREE(PTR, POOL) internal_libxs_scratch_free(PTR, POOL)

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


/** Sort-based multiset matching for real (scalar) element types. */
#define LIBXS_SETDIFF_REAL(TYPE, CVT) { \
  const TYPE *const ra = (const TYPE*)a, *const rb = (const TYPE*)b; \
  const size_t bufsz = ((size_t)na + (size_t)nb) * sizeof(double); \
  int pool_sd = 0; \
  double* const sa = (double*)LIBXS_MATH_MALLOC(bufsz, pool_sd); \
  if (NULL != sa) { \
    double* const sb = sa + na; \
    int m = 0; \
    for (i = 0; i < na; ++i) sa[i] = CVT(ra[i]); \
    for (j = 0; j < nb; ++j) sb[j] = CVT(rb[j]); \
    libxs_sort(sa, na, sizeof(double), libxs_cmp_f64, NULL); \
    libxs_sort(sb, nb, sizeof(double), libxs_cmp_f64, NULL); \
    i = 0; j = 0; \
    while (i < na && j < nb) { \
      if (LIBXS_DELTA(sa[i], sb[j]) <= tol) { ++m; ++i; ++j; } \
      else if (sa[i] < sb[j]) ++i; \
      else ++j; \
    } \
    result = nmax - m; \
    LIBXS_MATH_FREE(sa, pool_sd); \
  } \
  else { result = nmax; } \
}

/** k-d tree nearest-neighbor matching for complex types. */
#define LIBXS_SETDIFF_CMPLX(TYPE, CVT) { \
  const TYPE *const ra = (const TYPE*)a, *const rb = (const TYPE*)b; \
  const size_t bufsz = (size_t)nb * 2 * sizeof(double) \
    + (size_t)nb * sizeof(int) + (size_t)nb; \
  int pool_sd = 0; \
  double* const pts = (double*)LIBXS_MATH_MALLOC(bufsz, pool_sd); \
  if (NULL != pts) { \
    int* const idx = (int*)(pts + 2 * nb); \
    unsigned char* const used = (unsigned char*)(idx + nb); \
    int m = 0; \
    for (j = 0; j < nb; ++j) { \
      pts[2*j] = CVT(rb[2*j]); pts[2*j+1] = CVT(rb[2*j+1]); \
      idx[j] = j; \
    } \
    libxs_kdtree2d_build(pts, idx, nb); \
    memset(used, 0, (size_t)nb); \
    for (i = 0; i < na; ++i) { \
      const double qre = CVT(ra[2*i]), qim = CVT(ra[2*i+1]); \
      const double r2 = tol * tol; \
      const int hit = libxs_kdtree2d_nearest( \
        pts, idx, used, nb, qre, qim, r2); \
      if (0 <= hit) { used[hit] = 1; ++m; } \
    } \
    result = nmax - m; \
    LIBXS_MATH_FREE(pts, pool_sd); \
  } \
  else { result = nmax; } \
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


/** Merge-only: count matches on pre-sorted double arrays. */
LIBXS_API_INLINE int internal_libxs_setdiff_merge(
  const double* sa, int na, const double* sb, int nb, double tol)
{
  int i = 0, j = 0, m = 0;
  while (i < na && j < nb) {
    if (LIBXS_DELTA(sa[i], sb[j]) <= tol) { ++m; ++i; ++j; }
    else if (sa[i] < sb[j]) ++i;
    else ++j;
  }
  return m;
}

/** k-d tree query-only: count matches on pre-built tree. */
LIBXS_API_INLINE int internal_libxs_setdiff_kd_match(
  const double* pts, const int* idx, int nb,
  const double* qa, int na, double tol)
{
  const size_t usz = (size_t)nb;
  unsigned char ubuf[256];
  unsigned char* used = (usz <= sizeof(ubuf)) ? ubuf : NULL;
  int pool_u = 0, i, m = 0;
  if (NULL == used) {
    used = (unsigned char*)LIBXS_MATH_MALLOC(usz, pool_u);
    if (NULL == used) return 0;
  }
  memset(used, 0, usz);
  { const double r2 = tol * tol;
    for (i = 0; i < na; ++i) {
      const int hit = libxs_kdtree2d_nearest(
        pts, idx, used, nb, qa[2*i], qa[2*i+1], r2);
      if (0 <= hit) { used[hit] = 1; ++m; }
    }
  }
  if (used != ubuf) LIBXS_MATH_FREE(used, pool_u);
  return m;
}

/** Context for the GSS callback used by libxs_setdiff_min. */
LIBXS_EXTERN_C typedef struct internal_libxs_setdiff_ctx_t {
  const void *a, *b;
  const double *sa, *sb;
  const double *pts, *qa;
  const int* idx;
  libxs_data_t datatype;
  int na, nb;
} internal_libxs_setdiff_ctx_t;


#include "libxs_math_matdiff.h"


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
    unsigned int k[2][LIBXS_PRODUCT_LIMIT] = { { 0 } }, *k0 = k[0], *k1 = k[1], *kt, p;
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


LIBXS_API size_t libxs_mod_inverse(size_t a, size_t m)
{
  long long t = 0, newt = 1;
  size_t r = m, newr = a % m;
  LIBXS_ASSERT(0 != m && 0 != a);
  while (0 != newr) {
    const size_t q = r / newr;
    { const long long tmp = t - (long long)(q) * newt; t = newt; newt = tmp; }
    { const size_t tmp = r - q * newr; r = newr; newr = tmp; }
  }
  LIBXS_ASSERT(1 == r);
  return (size_t)(t < 0 ? t + (long long)m : t);
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


LIBXS_API_INTERN int internal_libxs_gss_close(double lhs, double rhs, double ftol)
{
  int result;
  if (0 < ftol) result = (LIBXS_FABS(lhs - rhs) <= ftol ? 1 : 0);
  else result = (lhs == rhs ? 1 : 0);
  return result;
}


LIBXS_API double libxs_bisect_min(
  double (*fn)(double x, const void* data), const void* data,
  double x0, double x1, double fmin, double* xmin, int maxiter,
  double ftol, libxs_gss_info_t* info)
{
  double left_x = x0, right_x = x1, best_x = x1, best_f;
  double left_f, right_f, midpoint, midpoint_f;
  int iteration = 0, evaluations = 0, status = 0;
  LIBXS_ASSERT(NULL != fn && x0 <= x1 && 0 < maxiter);
  left_f = fn(left_x, data);
  right_f = fn(right_x, data);
  evaluations += 2;
  best_f = right_f;
  if (0 != internal_libxs_gss_close(left_f, fmin, ftol) || left_f < fmin) {
    best_x = left_x;
    best_f = left_f;
    status |= LIBXS_GSS_STATUS_ENDPOINT_MIN;
  }
  else if (0 != internal_libxs_gss_close(right_f, fmin, ftol) || right_f < fmin) {
    status |= LIBXS_GSS_STATUS_ENDPOINT_MIN;
    for (iteration = 0; iteration < maxiter && left_x != right_x; ++iteration) {
      midpoint = 0.5 * (left_x + right_x);
      if (left_x == midpoint || right_x == midpoint) break;
      midpoint_f = fn(midpoint, data);
      ++evaluations;
      if (0 != internal_libxs_gss_close(midpoint_f, best_f, ftol) || midpoint_f < best_f) {
        right_x = midpoint;
        best_x = midpoint;
        best_f = midpoint_f;
      }
      else left_x = midpoint;
    }
    status |= LIBXS_GSS_STATUS_LEFT_REFINED;
    if (maxiter <= iteration) status |= LIBXS_GSS_STATUS_MAXITER;
  }
  else {
    status |= LIBXS_GSS_STATUS_NO_BRACKET;
  }
  if (NULL != xmin) *xmin = best_x;
  if (NULL != info) {
    info->status = status;
    info->iterations = iteration;
    info->evaluations = evaluations;
    info->unimodality = 1.0;
    info->xmin = best_x;
    info->fmin = best_f;
    info->x0 = left_x;
    info->x1 = right_x;
  }
  return best_f;
}


LIBXS_API double libxs_gss_min(
  double (*fn)(double x, const void* data), const void* data,
  double x0, double x1, double* xmin, int maxiter,
  int flags, double ftol, libxs_gss_info_t* info)
{
  const double phi = (sqrt(5.0) - 1.0) * 0.5;
  const int need_endpoints = (0 != (flags & LIBXS_GSS_EVAL_ENDPOINTS) ? 1 : 0);
  double bracket0 = x0, bracket1 = x1, width = bracket1 - bracket0;
  double candidate0 = bracket0 + (1.0 - phi) * width;
  double candidate1 = bracket0 + phi * width;
  double value0, value1, best_x, best_f, unimodality;
  double samples_x[64], samples_f[64];
  int iteration, nsamples = 0, evaluations = 0, consistent = 0, status = 0;
  LIBXS_ASSERT(NULL != fn && x0 <= x1 && 0 < maxiter);
  if (0 != need_endpoints) {
    if (nsamples < 64) { samples_x[nsamples] = x0; samples_f[nsamples] = fn(x0, data); ++nsamples; }
    if (nsamples < 64) { samples_x[nsamples] = x1; samples_f[nsamples] = fn(x1, data); ++nsamples; }
    evaluations += 2;
  }
  value0 = fn(candidate0, data); value1 = fn(candidate1, data); evaluations += 2;
  if (nsamples < 64) { samples_x[nsamples] = candidate0; samples_f[nsamples] = value0; ++nsamples; }
  if (nsamples < 64) { samples_x[nsamples] = candidate1; samples_f[nsamples] = value1; ++nsamples; }
  for (iteration = 0; iteration < maxiter && bracket0 != candidate0 && bracket1 != candidate1; ++iteration) {
    if (value0 <= value1) {
      bracket1 = candidate1; candidate1 = candidate0; value1 = value0;
      width = bracket1 - bracket0;
      candidate0 = bracket0 + (1.0 - phi) * width;
      value0 = fn(candidate0, data); ++evaluations;
      if (nsamples < 64) { samples_x[nsamples] = candidate0; samples_f[nsamples] = value0; ++nsamples; }
    }
    else {
      bracket0 = candidate0; candidate0 = candidate1; value0 = value1;
      width = bracket1 - bracket0;
      candidate1 = bracket0 + phi * width;
      value1 = fn(candidate1, data); ++evaluations;
      if (nsamples < 64) { samples_x[nsamples] = candidate1; samples_f[nsamples] = value1; ++nsamples; }
    }
  }
  if (maxiter <= iteration) status |= LIBXS_GSS_STATUS_MAXITER;
  if (nsamples >= 3) {
    int i, j, min_index = 0;
    for (i = 0; i < nsamples - 1; ++i) {
      for (j = i + 1; j < nsamples; ++j) {
        if (samples_x[j] < samples_x[i]) {
          double tmp_x = samples_x[i], tmp_f = samples_f[i];
          samples_x[i] = samples_x[j]; samples_f[i] = samples_f[j];
          samples_x[j] = tmp_x; samples_f[j] = tmp_f;
        }
      }
    }
    for (i = 1; i < nsamples; ++i) {
      if (samples_f[i] < samples_f[min_index]) min_index = i;
    }
    best_x = samples_x[min_index]; best_f = samples_f[min_index];
    for (i = 0; i < nsamples; ++i) {
      if (samples_f[i] < best_f || (internal_libxs_gss_close(samples_f[i], best_f, ftol) && samples_x[i] < best_x)) {
        best_x = samples_x[i]; best_f = samples_f[i]; min_index = i;
      }
    }
    consistent = 0;
    for (i = 1; i < nsamples; ++i) {
      if (i <= min_index && samples_f[i] <= samples_f[i - 1]) ++consistent;
      else if (i > min_index && samples_f[i] >= samples_f[i - 1]) ++consistent;
    }
    unimodality = (double)consistent / (nsamples - 1);
    for (i = 0; i < nsamples; ++i) {
      if (i != min_index && internal_libxs_gss_close(samples_f[i], best_f, ftol)) {
        status |= LIBXS_GSS_STATUS_FLAT_MIN;
      }
    }
    if (0 != internal_libxs_gss_close(best_x, x0, 0.0)
      || 0 != internal_libxs_gss_close(best_x, x1, 0.0)) status |= LIBXS_GSS_STATUS_ENDPOINT_MIN;
  }
  else {
    unimodality = 1.0;
    if (value0 <= value1) { best_x = candidate0; best_f = value0; }
    else { best_x = candidate1; best_f = value1; }
  }
  if (NULL != xmin) *xmin = best_x;
  if (NULL != info) {
    info->status = status;
    info->iterations = iteration;
    info->evaluations = evaluations;
    info->unimodality = unimodality;
    info->xmin = best_x;
    info->fmin = best_f;
    info->x0 = bracket0;
    info->x1 = bracket1;
  }
  return best_f;
}


#include "libxs_math_setdiff.h"

#include "libxs_math_fprint.h"
