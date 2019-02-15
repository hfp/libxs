#include <libxs.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 1000000


LIBXS_INLINE unsigned int ref_isqrt_u32(unsigned int u32)
{
  const unsigned int r = (unsigned int)(sqrt((double)u32) + 0.5);
  return ((double)r * r) <= u32 ? r : (r - 1);
}


LIBXS_INLINE unsigned int ref_isqrt_u64(unsigned long long u64)
{
  const unsigned long long r = (unsigned long long)(sqrtl((long double)u64) + 0.5);
  return (unsigned int)(((long double)r * r) <= u64 ? r : (r - 1));
}


LIBXS_INLINE unsigned int ref_icbrt_u32(unsigned int u32)
{
  const unsigned int r = (unsigned int)(pow((double)u32, 1.0 / 3.0) + 0.5);
  return ((double)r * r * r) <= u32 ? r : (r - 1);
}


LIBXS_INLINE unsigned int ref_icbrt_u64(unsigned long long u64)
{
  const unsigned long long r = (unsigned long long)(powl((long double)u64, 1.0 / 3.0) + 0.5);
  return (unsigned int)(((long double)r * r * r) <= u64 ? r : (r - 1));
}


LIBXS_INLINE unsigned int ref_ilog2_u32(unsigned int u32)
{
  return (unsigned int)ceil(LIBXS_LOG2(u32));
}


int main(int argc, char* argv[])
{
  const int exp_maxiter = (1 < argc ? atoi(argv[1]) : 20);
  const unsigned long long scale64 = ((unsigned long long)-1) / (RAND_MAX) - 1;
  const unsigned int scale32 = ((unsigned int)-1) / (RAND_MAX) - 1;
  int warn_dsqrt = 0, warn_ssqrt = 0, i;

  for (i = 0; i < 256; ++i) {
    const float a = libxs_sexp2_u8((unsigned char)i);
    const float b = LIBXS_POWF(2, i);
    if (LIBXS_NEQ(a, b)) exit(EXIT_FAILURE);
  }

  for (i = -128; i < 127; ++i) {
    const float a = libxs_sexp2_i8((signed char)i);
    const float b = LIBXS_POWF(2, i);
    if (LIBXS_NEQ(a, b)) exit(EXIT_FAILURE);
  }

  for (i = 0; i < (N); ++i) {
    const int r1 = (0 != i ? rand() : 0), r2 = (1 < i ? rand() : 0);
    const double rd = 2.0 * (r1 * (r2 - RAND_MAX / 2)) / RAND_MAX;
    const unsigned long long r64 = scale64 * r1;
    const unsigned int r32 = scale32 * r1;
    double d1, d2, e1, e2, e3;
    unsigned int a, b;

    if (LIBXS_NEQ(LIBXS_ROUND((double)r1), LIBXS_ROUNDX(double, (double)r1))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUND((double)r2), LIBXS_ROUNDX(double, (double)r2))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUND(r1), LIBXS_ROUNDX(double, r1))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUND(r2), LIBXS_ROUNDX(double, r2))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUND(rd), LIBXS_ROUNDX(double, rd))) exit(EXIT_FAILURE);

    if (LIBXS_NEQ(LIBXS_ROUNDF((float)r1), LIBXS_ROUNDX(float, (float)r1))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUNDF((float)r2), LIBXS_ROUNDX(float, (float)r2))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUNDF(r1), LIBXS_ROUNDX(float, r1))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUNDF(r2), LIBXS_ROUNDX(float, r2))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUNDF(rd), LIBXS_ROUNDX(float, rd))) exit(EXIT_FAILURE);

    d1 = libxs_sexp2_fast((float)rd, exp_maxiter);
    d2 = LIBXS_POWF(2, rd);
    e1 = fabs(d1 - d2); e2 = fabs(d2);
    e3 = 0 < e2 ? (e1 / e2) : 0.0;
    if (1E-4 < fmin(e1, e3)) exit(EXIT_FAILURE);

    a = libxs_isqrt_u32(r32);
    b = ref_isqrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = libxs_isqrt_u64(r64);
    b = ref_isqrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);
    d1 = libxs_ssqrt((float)fabs(rd));
    e1 = fabs(d1 * d1 - fabs(rd));
    d2 = sqrtf((float)fabs(rd));
    e2 = fabs(d2 * d2 - fabs(rd));
    if (e2 < e1) {
      e3 = 0 < e2 ? (e1 / e2) : 0.f;
      if (1E-2 > fmin(fabs(e1 - e2), e3)) {
        ++warn_ssqrt;
      }
      else {
        exit(EXIT_FAILURE);
      }
    }
    d1 = libxs_dsqrt(fabs(rd));
    e1 = fabs(d1 * d1 - fabs(rd));
    d2 = sqrt(fabs(rd));
    e2 = fabs(d2 * d2 - fabs(rd));
    if (e2 < e1) {
      e3 = 0 < e2 ? (e1 / e2) : 0.f;
      if (1E-11 > fmin(fabs(e1 - e2), e3)) {
        ++warn_dsqrt;
      }
      else {
        exit(EXIT_FAILURE);
      }
    }

    a = libxs_icbrt_u32(r32);
    b = ref_icbrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = libxs_icbrt_u64(r64);
    b = ref_icbrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);

    a = LIBXS_INTRINSICS_BITSCANFWD32(r32);
    b = LIBXS_INTRINSICS_BITSCANFWD32_SW(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = LIBXS_INTRINSICS_BITSCANBWD32(r32);
    b = LIBXS_INTRINSICS_BITSCANBWD32_SW(r32);
    if (a != b) exit(EXIT_FAILURE);

    a = LIBXS_INTRINSICS_BITSCANFWD64(r64);
    b = LIBXS_INTRINSICS_BITSCANFWD64_SW(r64);
    if (a != b) exit(EXIT_FAILURE);
    a = LIBXS_INTRINSICS_BITSCANBWD64(r64);
    b = LIBXS_INTRINSICS_BITSCANBWD64_SW(r64);
    if (a != b) exit(EXIT_FAILURE);

    a = LIBXS_ILOG2(i);
    b = ref_ilog2_u32(i);
    if (0 != i && a != b) exit(EXIT_FAILURE);
    a = LIBXS_ILOG2(r32);
    b = ref_ilog2_u32(r32);
    if (0 != r32 && a != b) exit(EXIT_FAILURE);

    a = LIBXS_SQRT2(i);
    b = libxs_isqrt_u32(i);
    if (a < LIBXS_DIFF(a, b)) exit(EXIT_FAILURE);
    a = LIBXS_SQRT2(r32);
    b = libxs_isqrt_u32(r32);
    if (a < LIBXS_DIFF(a, b)) exit(EXIT_FAILURE);
    a = LIBXS_SQRT2(r64);
    b = libxs_isqrt_u64(r64);
    if (0 != a/*u32-overflow*/ && a < LIBXS_DIFF(a, b)) exit(EXIT_FAILURE);
  }

  if (0 < warn_ssqrt || 0 < warn_dsqrt) {
    fprintf(stderr, "missed bitwise exact result in %.0f%% of the cases!\n", 100.0 * LIBXS_MAX(warn_ssqrt, warn_dsqrt) / N);
  }

  { /* check GCD */
    const size_t a[] = { 0, 1, 0, 100, 10 };
    const size_t b[] = { 0, 0, 1, 10, 100 };
    const size_t c[] = { 1, 1, 1, 10,  10 };
    const int n = sizeof(a) / sizeof(*a);
    for (i = 0; i < n; ++i) {
      if (libxs_gcd(a[i], b[i]) != c[i]) exit(EXIT_FAILURE);
    }
  }

  { /* check prime factorization */
    const unsigned int test[] = { 0, 1, 2, 3, 5, 7, 12, 13, 24, 32, 2057, 120, 14, 997 };
    const int n = sizeof(test) / sizeof(*test);
    unsigned int fact[32];
    for (i = 0; i < n; ++i) {
      const int np = libxs_primes_u32(test[i], fact);
      int j; for (j = 1; j < np; ++j) fact[0] *= fact[j];
      if (0 < np && fact[0] != test[i]) {
        exit(EXIT_FAILURE);
      }
    }
  }

  { /* check shuffle routine */
    const unsigned int test[] = { 0, 1, 2, 3, 5, 7, 12, 13, 24, 32, 2057, 120, 14, 997 };
    const int n = sizeof(test) / sizeof(*test);
    for (i = 0; i < n; ++i) {
      const size_t coprime = libxs_shuffle(test[i]);
      const unsigned int gcd = (unsigned int)libxs_gcd(coprime, test[i]);
      if ((0 != coprime || 1 < test[i]) && (test[i] <= coprime || 1 != gcd)) {
        exit(EXIT_FAILURE);
      }
    }
    if (libxs_shuffle(65423) != 32711) exit(EXIT_FAILURE);
    if (libxs_shuffle(1000) != 499) exit(EXIT_FAILURE);
    if (libxs_shuffle(997) != 498) exit(EXIT_FAILURE);
    if (libxs_shuffle(24) != 11) exit(EXIT_FAILURE);
    if (libxs_shuffle(5) != 2) exit(EXIT_FAILURE);
  }

  /* find upper limited product */
  if (libxs_product_limit(12 * 5 * 7 * 11 * 13 * 17, 231, 0) != (3 * 7 * 11)) exit(EXIT_FAILURE);
  if (libxs_product_limit(12 * 5 * 7, 32, 0) != (2 * 3 * 5)) exit(EXIT_FAILURE);
  if (libxs_product_limit(12 * 13, 13, 0) != 13) exit(EXIT_FAILURE);
  if (libxs_product_limit(12, 6, 0) != 6) exit(EXIT_FAILURE);
  if (libxs_product_limit(0, 48, 0) != 0) exit(EXIT_FAILURE);
  if (libxs_product_limit(0, 1, 0) != 0) exit(EXIT_FAILURE);
  if (libxs_product_limit(0, 0, 0) != 0) exit(EXIT_FAILURE);
  if (libxs_product_limit(1, 0, 0) != 0) exit(EXIT_FAILURE);

  /* find lower limited product */
  if (libxs_product_limit(12 * 5 * 7 * 11 * 13 * 17, 231, 1) != (3 * 7 * 11)) exit(EXIT_FAILURE);
  if (libxs_product_limit(12 * 5 * 7, 36, 1) != (2 * 5 * 7)) exit(EXIT_FAILURE);
  if (libxs_product_limit(12 * 13, 13, 1) != 13) exit(EXIT_FAILURE);
  if (libxs_product_limit(320, 300, 1) != 320) exit(EXIT_FAILURE);
  if (libxs_product_limit(320, 65, 1) != 80) exit(EXIT_FAILURE);
  if (libxs_product_limit(320, 33, 1) != 64) exit(EXIT_FAILURE);
  if (libxs_product_limit(1000, 6, 1) != 10) exit(EXIT_FAILURE);
  if (libxs_product_limit(1000, 9, 1) != 10) exit(EXIT_FAILURE);
  if (libxs_product_limit(12, 7, 1) != 12) exit(EXIT_FAILURE);
  if (libxs_product_limit(5, 2, 1) != 5) exit(EXIT_FAILURE);
  if (libxs_product_limit(5, 2, 0) != 1) exit(EXIT_FAILURE);
  if (libxs_product_limit(0, 1, 1) != 0) exit(EXIT_FAILURE);
  if (libxs_product_limit(0, 0, 1) != 0) exit(EXIT_FAILURE);
  if (libxs_product_limit(1, 0, 1) != 0) exit(EXIT_FAILURE);

  return EXIT_SUCCESS;
}

