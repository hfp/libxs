/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MATH_H
#define LIBXS_MATH_H

#include "libxs.h"


/**
 * Structure of differences with matrix norms according
 * to http://www.netlib.org/lapack/lug/node75.html).
 */
LIBXS_EXTERN_C typedef struct libxs_matdiff_info_t {
  /** One-norm */         double norm1_abs, norm1_rel;
  /** Infinity-norm */    double normi_abs, normi_rel;
  /** Froebenius-norm */  double normf_rel;
  /** Maximum difference, L2-norm (absolute and relative), and R-squared. */
  double linf_abs, linf_rel, l2_abs, l2_rel, rsq;
  /** Statistics: sum/l1, min., max., arith. avg., and variance. */
  double l1_ref, min_ref, max_ref, avg_ref, var_ref;
  /** Statistics: sum/l1, min., max., arith. avg., and variance. */
  double l1_tst, min_tst, max_tst, avg_tst, var_tst;
  /* Values(v_ref, v_tst) and location(m, n) of largest linf_abs. */
  double v_ref, v_tst;
  /**
   * If r is non-zero (i is not negative), values (v_ref, v_tst),
   * and the location (m, n) stem from the i-th reduction
   * (r calls of libxs_matdiff_reduce) of the largest
   * difference (libxs_matdiff_epsilon).
   */
  int m, n, i, r;
} libxs_matdiff_info_t;

/**
 * Utility function to calculate a collection of scalar differences between two matrices (libxs_matdiff_info_t).
 * The location (m, n) of the largest difference (linf_abs) is recorded (also in case of NaN). In case of NaN,
 * differences are set to infinity. If no difference is discovered, the location (m, n) is negative (OOB).
 * The return value does not judge the difference (norm) between reference and test data, but is about
 * missing support for the requested data-type or otherwise invalid input.
 */
LIBXS_API int libxs_matdiff(libxs_matdiff_info_t* info,
  libxs_datatype datatype, int m, int n, const void* ref, const void* tst,
  const int* ldref, const int* ldtst);

/**
 * Combine absolute and relative norms into a value which can be used to check against a margin.
 * A file or directory path given per environment variable LIBXS_MATDIFF=/path/to/file stores
 * the epsilon (followed by a line-break), which can be used to calibrate margins of a test case.
 * LIBXS_MATDIFF can carry optional space-separated arguments used to amend the file entry.
 */
LIBXS_API double libxs_matdiff_epsilon(const libxs_matdiff_info_t* input);
/**
 * Reduces input into output such that the difference is maintained or increased (max function).
 * The very first (initial) output should be zeroed (libxs_matdiff_clear).
 */
LIBXS_API void libxs_matdiff_reduce(libxs_matdiff_info_t* output, const libxs_matdiff_info_t* input);
/** Clears the given info-structure, e.g., for the initial reduction-value (libxs_matdiff_reduce). */
LIBXS_API void libxs_matdiff_clear(libxs_matdiff_info_t* info);

/** Greatest common divisor (corner case: the GCD of 0 and 0 is 1). */
LIBXS_API size_t libxs_gcd(size_t a, size_t b);
/** Least common multiple. */
LIBXS_API size_t libxs_lcm(size_t a, size_t b);

/**
 * This function finds prime-factors (up to 32) of an unsigned integer in ascending order, and
 * returns the number of factors found (zero if the given number is prime and unequal to two).
 */
LIBXS_API int libxs_primes_u32(unsigned int num, unsigned int num_factors_n32[]);

/** Co-prime R of N such that R <= MinCo (libxs_coprime2(0|1) == 0). */
LIBXS_API size_t libxs_coprime(size_t n, size_t minco);
/** Co-prime R of N such that R <= SQRT(N) (libxs_coprime2(0|1) == 0). */
LIBXS_API size_t libxs_coprime2(size_t n);

/**
 * Minimizes the waste, if "a" can only be processed in multiples of "b".
 * The remainder r is such that ((i * b) % a) <= r with i := {1, ..., a}.
 * Return value of this function is (i * b) with i := {1, ..., a}.
 * Remainder and limit are considered for early-exit and relaxation.
 * If the remainder is not given (NULL), it is assumed to be zero.
 * For example: libxs_remainder(23, 8, NULL, NULL) => 184.
 */
LIBXS_API unsigned int libxs_remainder(unsigned int a, unsigned int b,
  /** Optional limit such that (i * b) <= limit or ((i * b) % a) <= r. */
  const unsigned int* limit,
  /** Optional remainder limiting ((i * b) % a) <= r. */
  const unsigned int* remainder);

/**
 * Divides the product into prime factors and selects factors such that the new product is within
 * the given limit (0/1-Knapsack problem), e.g., product=12=2*2*3 and limit=6 then result=2*3=6.
 * The limit is at least reached or exceeded with the minimal possible product (is_lower=true).
 */
LIBXS_API unsigned int libxs_product_limit(unsigned int product, unsigned int limit, int is_lower);

/* Kahan's summation returns accumulator += value and updates compensation. */
LIBXS_API double libxs_kahan_sum(double value, double* accumulator, double* compensation);

/** SQRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_isqrt_u64(unsigned long long x);
/** SQRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_isqrt_u32(unsigned int x);
/** Based on libxs_isqrt_u32; result is factor of x. */
LIBXS_API unsigned int libxs_isqrt2_u32(unsigned int x);

/**
 * Construct a double with value 2^n by manipulating the IEEE-754 exponent
 * field directly. Valid for n in [-1022, 1023]; returns 0.0 for underflow
 * and +Inf for overflow. Subnormals (n < -1022) are flushed to zero.
 */
LIBXS_API double libxs_pow2(int n);

/**
 * Modular inverse via extended Euclidean algorithm: a^{-1} mod m.
 * Requires gcd(a, m) = 1 and 0 < a, 1 < m.
 */
LIBXS_API unsigned int libxs_mod_inverse_u32(unsigned int a, unsigned int m);

/**
 * Barrett reciprocal for a 32-bit modulus: floor(2^32 / p).
 * Used by libxs_mod_u32 and libxs_mod_u64 for fast reduction.
 */
LIBXS_API unsigned int libxs_barrett_rcp(unsigned int p);
/**
 * Radix-split power table entry: (1 << 18) mod p.
 * Used by libxs_mod_u64 for 64-bit reduction via radix-2^18 split.
 */
LIBXS_API unsigned int libxs_barrett_pow18(unsigned int p);
/**
 * Radix-split power table entry: (1 << 36) mod p.
 * Used by libxs_mod_u64 for 64-bit reduction via radix-2^18 split.
 */
LIBXS_API unsigned int libxs_barrett_pow36(unsigned int p);

/**
 * Fast 32-bit modular reduction via Barrett's method.
 * Returns x mod p using a precomputed reciprocal rcp = floor(2^32/p).
 * Valid for x < 2^32.
 */
LIBXS_API_INLINE unsigned int libxs_mod_u32(uint32_t x, unsigned int p,
  unsigned int rcp)
{
  const uint32_t q = (uint32_t)(((uint64_t)x * rcp) >> 32);
  uint32_t r = x - q * (uint32_t)p;
  if (r >= (uint32_t)p) r -= (uint32_t)p;
  return (unsigned int)r;
}

/**
 * Fast 64-bit modular reduction via radix-2^18 split and Barrett.
 * Decomposes x into three 18-bit chunks and recombines mod p:
 *   (a2*pow36 + a1*pow18 + a0) mod p
 * using the 32-bit Barrett libxs_mod_u32.
 * Valid for x < 2^54 and p < 8192 (ensures intermediate sum < 2^32).
 */
LIBXS_API_INLINE unsigned int libxs_mod_u64(uint64_t x, unsigned int p,
  unsigned int rcp, unsigned int pow18, unsigned int pow36)
{
  const uint32_t a0 = (uint32_t)(x & 0x3FFFFU);
  const uint32_t a1 = (uint32_t)((x >> 18) & 0x3FFFFU);
  const uint32_t a2 = (uint32_t)(x >> 36);
  return libxs_mod_u32(a2 * pow36 + a1 * pow18 + a0, p, rcp);
}

#endif /*LIBXS_MATH_H*/
