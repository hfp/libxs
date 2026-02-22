/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_utils.h>
#include <libxs_rng.h>

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#define N 1000000


LIBXS_INLINE unsigned int ref_isqrt_u32(unsigned int u32)
{
  const unsigned int r = (unsigned int)(sqrt((double)u32) + 0.5);
  return ((double)r * r) <= u32 ? r : (r - 1);
}


LIBXS_INLINE unsigned int ref_isqrt_u64(unsigned long long u64)
{
#if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
  const unsigned long long r = (unsigned long long)(sqrtl((long double)u64) + 0.5);
#else
  const unsigned long long r = (unsigned long long)(sqrt((double)u64) + 0.5);
#endif
  return (unsigned int)(((long double)r * r) <= u64 ? r : (r - 1));
}


LIBXS_INLINE unsigned int ref_ilog2_u32(unsigned int u32)
{
  return (unsigned int)ceil(LIBXS_LOG2(u32));
}


int main(int argc, char* argv[])
{
  const unsigned long long scale64 = ((unsigned long long)-1) / (RAND_MAX) - 1;
  const unsigned int scale32 = ((unsigned int)-1) / (RAND_MAX) - 1;
  int i, j;
  LIBXS_UNUSED(argc); LIBXS_UNUSED(argv);

  for (i = 0; i < (N); ++i) {
    const int r1 = (0 != i ? rand() : 0), r2 = (1 < i ? rand() : 0);
    const double rd = 2.0 * ((long long int)r1 * (r2 - RAND_MAX / 2)) / RAND_MAX;
    const unsigned long long r64 = scale64 * r1;
    const unsigned int r32 = scale32 * r1;
    double d1, d2, e1, e2, e3;
    unsigned int a, b;

    if (LIBXS_NEQ(LIBXS_ROUND((double)r1), LIBXS_ROUNDX(double, (double)r1))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUND((double)r2), LIBXS_ROUNDX(double, (double)r2))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUND((double)rd), LIBXS_ROUNDX(double, (double)rd))) exit(EXIT_FAILURE);

    if (LIBXS_NEQ(LIBXS_ROUNDF((float)r1), LIBXS_ROUNDX(float, (float)r1))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUNDF((float)r2), LIBXS_ROUNDX(float, (float)r2))) exit(EXIT_FAILURE);
    if (LIBXS_NEQ(LIBXS_ROUNDF((float)rd), LIBXS_ROUNDX(float, (float)rd))) exit(EXIT_FAILURE);

    a = libxs_isqrt_u32(r32);
    b = ref_isqrt_u32(r32);
    if (a != b) exit(EXIT_FAILURE);
    a = libxs_isqrt_u64(r64);
    b = ref_isqrt_u64(r64);
    if (a != b) exit(EXIT_FAILURE);

    d1 = 1.f / LIBXS_SQRTF(28.f);
    e1 = LIBXS_FABS(1.0 / (d1 * d1) - 28.0);
    d2 = 1.0 / sqrt(28.0);
    e2 = LIBXS_FABS(1.0 / (d2 * d2) - 28.0);
    if (e2 < e1) {
      e3 = 0 < e2 ? (e1 / e2) : 0.f;
      if (4E-06 < LIBXS_MIN(LIBXS_FABS(e1 - e2), e3)) {
        exit(EXIT_FAILURE);
      }
    }

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

    a = LIBXS_ISQRT2(i);
    b = libxs_isqrt_u32(i);
    if (a < LIBXS_DELTA(a, b)) exit(EXIT_FAILURE);
    a = LIBXS_ISQRT2(r32);
    b = libxs_isqrt_u32(r32);
    if (a < LIBXS_DELTA(a, b)) exit(EXIT_FAILURE);
    a = LIBXS_ISQRT2(r64);
    b = libxs_isqrt_u64(r64);
    if (0 != a/*u32-overflow*/ && a < LIBXS_DELTA(a, b)) exit(EXIT_FAILURE);
  }

  { /* further check LIBXS_INTRINSICS_BITSCANBWD32 */
    const int npot[] = { 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 65536 };
    const int n = sizeof(npot) / sizeof(*npot);
    for (i = 0; i < n; ++i) {
      const int numpot = npot[i];
      const int nbits = LIBXS_INTRINSICS_BITSCANBWD32(numpot);
      const int num = nbits < numpot ? (1 << nbits) : nbits;
      if (numpot != num) {
        exit(EXIT_FAILURE);
      }
    }
  }

  { /* bit operations: specific tests */
    unsigned int a, b;
    a = LIBXS_INTRINSICS_BITSCANFWD64(0x2aaaab69ede0);
    b = LIBXS_INTRINSICS_BITSCANFWD64_SW(0x2aaaab69ede0);
    if (a != b) exit(EXIT_FAILURE);
    /* values with no bits set in the low 32 bits (regression: was an infinite loop) */
    b = LIBXS_INTRINSICS_BITSCANFWD64_SW(0x100000000ULL);
    if (32 != b) exit(EXIT_FAILURE);
    b = LIBXS_INTRINSICS_BITSCANFWD64_SW(0x8000000000000000ULL);
    if (63 != b) exit(EXIT_FAILURE);
    b = LIBXS_INTRINSICS_BITSCANFWD64_SW(0x4000000000ULL);
    if (38 != b) exit(EXIT_FAILURE);
  }

  { /* check LIBXS_UP2POT and LIBXS_LO2POT */
    const size_t a[] = { 0, 1, 10, 100, 127, 128, 129 };
    const size_t b[] = { 0, 1, 16, 128, 128, 128, 256 };
    const size_t c[] = { 0, 1,  8,  64,  64, 128, 128 };
    const int n = sizeof(a) / sizeof(*a);
    for (i = 0; i < n; ++i) {
      if ((size_t)LIBXS_UP2POT(a[i]) != b[i]) exit(EXIT_FAILURE);
      if ((size_t)LIBXS_LO2POT(a[i]) != c[i]) exit(EXIT_FAILURE);
      if (LIBXS_ISPOT(a[i]) != (0 != a[i] && a[i] == (size_t)LIBXS_UP2POT(a[i]))) exit(EXIT_FAILURE);
      if (LIBXS_ISPOT(a[i]) != (0 != a[i] && a[i] == (size_t)LIBXS_LO2POT(a[i]))) exit(EXIT_FAILURE);
      if (LIBXS_ISPOT(b[i]) != (0 != b[i] && b[i] == (size_t)LIBXS_UP2POT(b[i]))) exit(EXIT_FAILURE);
      if (LIBXS_ISPOT(b[i]) != (0 != b[i] && b[i] == (size_t)LIBXS_LO2POT(b[i]))) exit(EXIT_FAILURE);
      if (LIBXS_ISPOT(c[i]) != (0 != c[i] && c[i] == (size_t)LIBXS_UP2POT(c[i]))) exit(EXIT_FAILURE);
      if (LIBXS_ISPOT(c[i]) != (0 != c[i] && c[i] == (size_t)LIBXS_LO2POT(c[i]))) exit(EXIT_FAILURE);
    }
  }

#if defined(INFINITY)
  { /* check infinity */
    const union { uint32_t raw; float value; } inf = { 0x7F800000U };
    if (inf.value != INFINITY) exit(EXIT_FAILURE);
  }
#endif

  { /* check LIBXS_UPDIV */
    const int inp[] = { 0, 1, 3, 5, 127, 3000 };
    const int out[] = { 0, 1, 1, 1,  19,  429 };
    const int n = sizeof(inp) / sizeof(*inp);
    for (i = 0; i < n; ++i) {
      if (LIBXS_UPDIV(inp[i], 7) != out[i]) exit(EXIT_FAILURE);
    }
  }

  { /* check LIBXS_UP */
    const int inp[] = { 0, 1, 3, 5, 127, 3000 };
    const int out[] = { 0, 7, 7, 7, 133, 3003 };
    const int n = sizeof(inp) / sizeof(*inp);
    for (i = 0; i < n; ++i) {
      if (LIBXS_UP(inp[i], 7) != out[i]) exit(EXIT_FAILURE);
    }
  }

  { /* check LIBXS_LO2 */
    const int inp[] = { 0, 7, 8, 9, 127, 3000 };
    const int out[] = { 0, 0, 8, 8, 120, 3000 };
    const int n = sizeof(inp) / sizeof(*inp);
    for (i = 0; i < n; ++i) {
      if (LIBXS_LO2(inp[i], 8) != out[i]) exit(EXIT_FAILURE);
    }
  }

  { /* check LIBXS_UP2 */
    const int inp[] = { 0, 1, 3, 5, 127, 3000 };
    const int out[] = { 0, 8, 8, 8, 128, 3000 };
    const int n = sizeof(inp) / sizeof(*inp);
    for (i = 0; i < n; ++i) {
      if (LIBXS_UP2(inp[i], 8) != out[i]) exit(EXIT_FAILURE);
    }
  }

  { /* check LIBXS_UPF */
    const int inp[] = { 0, 1, 3, 5, 127, 3000 };
    const int out[] = { 0, 1, 3, 5, 130, 3090 };
    const int n = sizeof(inp) / sizeof(*inp);
    for (i = 0; i < n; ++i) {
      if (LIBXS_UPF(inp[i], 3, 100) != out[i]) exit(EXIT_FAILURE);
      if (LIBXS_UPF(inp[i], -100, 100) != 0) exit(EXIT_FAILURE);
    }
  }

  { /* check LIBXS_MAX / LIBXS_MIN */
    if (LIBXS_MAX(3, 7) != 7) { FPRINTF(stderr, "ERROR line #%i: MAX(3,7)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MAX(7, 3) != 7) { FPRINTF(stderr, "ERROR line #%i: MAX(7,3)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MAX(-1, 0) != 0) { FPRINTF(stderr, "ERROR line #%i: MAX(-1,0)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MIN(3, 7) != 3) { FPRINTF(stderr, "ERROR line #%i: MIN(3,7)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MIN(-1, 0) != -1) { FPRINTF(stderr, "ERROR line #%i: MIN(-1,0)\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_SIGN */
    if (LIBXS_SIGN(42) != 1) { FPRINTF(stderr, "ERROR line #%i: SIGN(42)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_SIGN(0) != 0) { FPRINTF(stderr, "ERROR line #%i: SIGN(0)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_SIGN(-7) != -1) { FPRINTF(stderr, "ERROR line #%i: SIGN(-7)\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_ABS */
    if (LIBXS_ABS(42) != 42) { FPRINTF(stderr, "ERROR line #%i: ABS(42)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_ABS(0) != 0) { FPRINTF(stderr, "ERROR line #%i: ABS(0)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_ABS(-7) != 7) { FPRINTF(stderr, "ERROR line #%i: ABS(-7)\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_MOD / LIBXS_MOD2 */
    if (LIBXS_MOD(10, 3) != 1) { FPRINTF(stderr, "ERROR line #%i: MOD(10,3)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MOD(8, 4) != 0) { FPRINTF(stderr, "ERROR line #%i: MOD(8,4)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MOD2(10, 8) != 2) { FPRINTF(stderr, "ERROR line #%i: MOD2(10,8)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MOD2(16, 16) != 0) { FPRINTF(stderr, "ERROR line #%i: MOD2(16,16)\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_MOD2(127, 64) != 63) { FPRINTF(stderr, "ERROR line #%i: MOD2(127,64)\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_CLMP (clamp) */
    if (LIBXS_CLMP(5, 1, 10) != 5) { FPRINTF(stderr, "ERROR line #%i: CLMP in-range\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_CLMP(-3, 0, 10) != 0) { FPRINTF(stderr, "ERROR line #%i: CLMP below\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_CLMP(42, 0, 10) != 10) { FPRINTF(stderr, "ERROR line #%i: CLMP above\n", __LINE__); exit(EXIT_FAILURE); }
    if (LIBXS_CLMP(0, 0, 0) != 0) { FPRINTF(stderr, "ERROR line #%i: CLMP degenerate\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_ISWAP */
    int x = 11, y = 22;
    LIBXS_ISWAP(x, y);
    if (x != 22 || y != 11) { FPRINTF(stderr, "ERROR line #%i: ISWAP\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_FEQ / LIBXS_ISNAN / LIBXS_NOTNAN */
    const double zero = 0.0;
    if (!LIBXS_FEQ(1.0, 1.0)) { FPRINTF(stderr, "ERROR line #%i: FEQ equal\n", __LINE__); exit(EXIT_FAILURE); }
    if ( LIBXS_FEQ(1.0, 2.0)) { FPRINTF(stderr, "ERROR line #%i: FEQ unequal\n", __LINE__); exit(EXIT_FAILURE); }
#if defined(NAN)
    if (!LIBXS_ISNAN(NAN)) { FPRINTF(stderr, "ERROR line #%i: ISNAN(NAN)\n", __LINE__); exit(EXIT_FAILURE); }
    if ( LIBXS_NOTNAN(NAN)) { FPRINTF(stderr, "ERROR line #%i: NOTNAN(NAN)\n", __LINE__); exit(EXIT_FAILURE); }
#endif
    if ( LIBXS_ISNAN(zero)) { FPRINTF(stderr, "ERROR line #%i: ISNAN(0)\n", __LINE__); exit(EXIT_FAILURE); }
    if (!LIBXS_NOTNAN(1.0)) { FPRINTF(stderr, "ERROR line #%i: NOTNAN(1)\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check LIBXS_NEARBYINT / LIBXS_NEARBYINTF */
    if (LIBXS_NEARBYINT(2.5) != 2.0 && LIBXS_NEARBYINT(2.5) != 3.0) {
      FPRINTF(stderr, "ERROR line #%i: NEARBYINT(2.5)=%f\n", __LINE__, LIBXS_NEARBYINT(2.5));
      exit(EXIT_FAILURE);
    }
    if (LIBXS_NEARBYINT(-0.5) != 0.0 && LIBXS_NEARBYINT(-0.5) != -1.0) {
      FPRINTF(stderr, "ERROR line #%i: NEARBYINT(-0.5)=%f\n", __LINE__, LIBXS_NEARBYINT(-0.5));
      exit(EXIT_FAILURE);
    }
    if (LIBXS_NEARBYINTF(2.5f) != 2.0f && LIBXS_NEARBYINTF(2.5f) != 3.0f) {
      FPRINTF(stderr, "ERROR line #%i: NEARBYINTF(2.5)=%f\n", __LINE__, (double)LIBXS_NEARBYINTF(2.5f));
      exit(EXIT_FAILURE);
    }
    if (LIBXS_NEARBYINT(3.7) != 4.0) {
      FPRINTF(stderr, "ERROR line #%i: NEARBYINT(3.7)=%f\n", __LINE__, LIBXS_NEARBYINT(3.7));
      exit(EXIT_FAILURE);
    }
    if (LIBXS_NEARBYINT(-3.7) != -4.0) {
      FPRINTF(stderr, "ERROR line #%i: NEARBYINT(-3.7)=%f\n", __LINE__, LIBXS_NEARBYINT(-3.7));
      exit(EXIT_FAILURE);
    }
  }

  { /* check GCD */
    const size_t a[] = { 0, 1, 0, 100, 10 };
    const size_t b[] = { 0, 0, 2, 10, 100 };
    const size_t c[] = { 1, 1, 2, 10,  10 };
    const int n = sizeof(a) / sizeof(*a);
    for (i = 0; i < n; ++i) {
      if (libxs_gcd(a[i], b[i]) != c[i]) exit(EXIT_FAILURE);
      if (libxs_gcd(b[i], a[i]) != c[i]) exit(EXIT_FAILURE);
    }
  }

  { /* check LCM */
    if (libxs_lcm(4, 6) != 12) { FPRINTF(stderr, "ERROR line #%i: lcm(4,6)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(6, 4) != 12) { FPRINTF(stderr, "ERROR line #%i: lcm(6,4)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(0, 5) !=  0) { FPRINTF(stderr, "ERROR line #%i: lcm(0,5)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(5, 0) !=  0) { FPRINTF(stderr, "ERROR line #%i: lcm(5,0)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(0, 0) !=  0) { FPRINTF(stderr, "ERROR line #%i: lcm(0,0)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(7, 7) !=  7) { FPRINTF(stderr, "ERROR line #%i: lcm(7,7)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(1, 9) !=  9) { FPRINTF(stderr, "ERROR line #%i: lcm(1,9)\n", __LINE__); exit(EXIT_FAILURE); }
    if (libxs_lcm(12, 18) != 36) { FPRINTF(stderr, "ERROR line #%i: lcm(12,18)\n", __LINE__); exit(EXIT_FAILURE); }
    /* LCM of coprimes equals the product */
    if (libxs_lcm(7, 13) != 91) { FPRINTF(stderr, "ERROR line #%i: lcm(7,13)\n", __LINE__); exit(EXIT_FAILURE); }
  }

  { /* check prime factorization */
    const unsigned int test[] = { 0, 1, 2, 3, 5, 7, 12, 13, 24, 32, 2057, 120, 14, 997, 65519u * 65521u };
    const int n = sizeof(test) / sizeof(*test);
    unsigned int fact[32];
    for (i = 0; i < n; ++i) {
      const int np = libxs_primes_u32(test[i], fact);
      for (j = 1; j < np; ++j) fact[0] *= fact[j];
      if (0 < np && fact[0] != test[i]) {
        exit(EXIT_FAILURE);
      }
    }
  }

  { /* check coprime routine */
    const size_t test[] = { 0, 1, 2, 3, 5, 7, 12, 13, 24, 32, 2057, 120, 14, 997, 1024, 4096 };
    const int n = sizeof(test) / sizeof(*test);
    for (i = 0; i < n; ++i) {
      for (j = 0; j <= (int)test[i]; ++j) {
        const size_t coprime = libxs_coprime(test[i], j);
        const size_t gcd = libxs_gcd(coprime, test[i]);
        if ((0 != coprime || 1 < test[i]) && (test[i] <= coprime || 1 != gcd)) {
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  { /* check coprime2 (coprime <= sqrt(n)) */
    const size_t test[] = { 0, 1, 2, 3, 4, 7, 16, 25, 100, 1024, 4096 };
    const int n = sizeof(test) / sizeof(*test);
    for (i = 0; i < n; ++i) {
      const size_t c2 = libxs_coprime2(test[i]);
      if (1 < test[i]) {
        const size_t s = libxs_isqrt_u64(test[i]);
        if (0 == c2 || c2 > s || test[i] <= c2) {
          FPRINTF(stderr, "ERROR line #%i: coprime2(%i)=%i sqrt=%i\n", __LINE__,
            (int)test[i], (int)c2, (int)s);
          exit(EXIT_FAILURE);
        }
        if (1 != libxs_gcd(c2, test[i])) {
          FPRINTF(stderr, "ERROR line #%i: coprime2(%i) gcd\n", __LINE__, (int)test[i]);
          exit(EXIT_FAILURE);
        }
      }
      else {
        if (0 != c2) {
          FPRINTF(stderr, "ERROR line #%i: coprime2(%i) expected 0\n", __LINE__, (int)test[i]);
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  { /* check libxs_remainder minimizing the remainder */
    unsigned int lim, rem;
    lim = 512;  if (libxs_remainder(23, 32, &lim, NULL) != (32 * 13)) exit(EXIT_FAILURE);
    rem = 4;    if (libxs_remainder(23, 32, NULL, &rem) != (32 * 3)) exit(EXIT_FAILURE);
    rem = 1;    if (libxs_remainder(23, 32, &lim, &rem) != (32 * 13)) exit(EXIT_FAILURE);
    lim = 32;   if (libxs_remainder(23, 8, &lim, NULL) != (8 * 3)) exit(EXIT_FAILURE);
    lim = 23;   if (libxs_remainder(23, 8, &lim, NULL) != (8 * 23)) exit(EXIT_FAILURE);
    lim = 4;    if (libxs_remainder(23, 8, &lim, NULL) != (8 * 23)) exit(EXIT_FAILURE);
    rem = 1;    if (libxs_remainder(0, 0, NULL, &rem) != 0) exit(EXIT_FAILURE);
    if (libxs_remainder(23, 32, NULL, NULL) != (32 * 23)) exit(EXIT_FAILURE);
    if (libxs_remainder(23, 8, NULL, NULL) != (8 * 23)) exit(EXIT_FAILURE);
    if (libxs_remainder(23, 8, NULL, NULL) != (8 * 23)) exit(EXIT_FAILURE);
    if (libxs_remainder(0, 32, NULL, NULL) != 0) exit(EXIT_FAILURE);
    if (libxs_remainder(23, 0, NULL, NULL) != 0) exit(EXIT_FAILURE);
    if (libxs_remainder(0, 0, NULL, NULL) != 0) exit(EXIT_FAILURE);
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
  if (libxs_product_limit(23, 32, 1) != (2 * 23)) exit(EXIT_FAILURE);
  if (libxs_product_limit(12, 32, 1) != (3 * 12)) exit(EXIT_FAILURE);
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

  if (libxs_isqrt2_u32(1024) *  32 != 1024) exit(EXIT_FAILURE);
  if (libxs_isqrt2_u32(1981) * 283 != 1981) exit(EXIT_FAILURE);
  if (libxs_isqrt2_u32(2507) * 109 != 2507) exit(EXIT_FAILURE);
  if (libxs_isqrt2_u32(1975) *  79 != 1975) exit(EXIT_FAILURE);

  { /* exercise large-limit path (limit > LIBXS_PRODUCT_LIMIT=1024) */
    unsigned int r;
    r = libxs_product_limit(2u * 3 * 5 * 7 * 11 * 13, 1500, 0);
    if (r > 1500) {
      FPRINTF(stderr, "ERROR line #%i: product_limit(30030,1500,0)=%u > 1500\n", __LINE__, r);
      exit(EXIT_FAILURE);
    }
    if (0 != (2u * 3 * 5 * 7 * 11 * 13) % r) {
      FPRINTF(stderr, "ERROR line #%i: product_limit result %u not a factor\n", __LINE__, r);
      exit(EXIT_FAILURE);
    }
    r = libxs_product_limit(2u * 3 * 5 * 7 * 11 * 13, 1500, 1);
    if (r < 1500) {
      FPRINTF(stderr, "ERROR line #%i: product_limit(30030,1500,1)=%u < 1500\n", __LINE__, r);
      exit(EXIT_FAILURE);
    }
  }

  { /* check Kahan summation */
    double acc = 0.0, comp = 0.0;
    const int kn = 10000;
    double naive = 0.0;
    for (i = 0; i < kn; ++i) {
      libxs_kahan_sum(1.0e-8, &acc, &comp);
      naive += 1.0e-8;
    }
    /* Kahan result must be closer to the exact answer (1e-4) than naive sum */
    if (LIBXS_FABS(acc - 1.0e-4) > LIBXS_FABS(naive - 1.0e-4)) {
      FPRINTF(stderr, "ERROR line #%i: kahan less accurate than naive\n", __LINE__);
      exit(EXIT_FAILURE);
    }
    if (LIBXS_FABS(acc - 1.0e-4) > 1.0e-14) {
      FPRINTF(stderr, "ERROR line #%i: kahan acc=%.20e expected 1e-4\n", __LINE__, acc);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
