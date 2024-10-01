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

#include "libxs_typedefs.h"

/** Helper macro to setup a matrix with some initial values. */
#define LIBXS_MATRNG_AUX(OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) do { \
  /*const*/ double libxs_matrng_seed_ = SEED; /* avoid constant conditional */ \
  const double libxs_matrng_scale_ = libxs_matrng_seed_ * (SCALE) + (SCALE); \
  const libxs_blasint libxs_matrng_nrows_ = (libxs_blasint)(NROWS); \
  const libxs_blasint libxs_matrng_ncols_ = (libxs_blasint)(NCOLS); \
  const libxs_blasint libxs_matrng_ld_ = (libxs_blasint)(LD); \
  libxs_blasint libxs_matrng_i_ = 0, libxs_matrng_j_ = 0; \
  LIBXS_OMP_VAR(libxs_matrng_i_); LIBXS_OMP_VAR(libxs_matrng_j_); \
  if (0 != libxs_matrng_seed_) { \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_nrows_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((TYPE*)(DST))[libxs_matrng_k_] = (TYPE)(libxs_matrng_scale_ * (1.0 + \
          (double)libxs_matrng_i_ * libxs_matrng_nrows_ + libxs_matrng_j_)); \
      } \
      for (; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((TYPE*)(DST))[libxs_matrng_k_] = (TYPE)libxs_matrng_seed_; \
      } \
    } \
  } \
  else { /* shuffle based initialization */ \
    const libxs_blasint libxs_matrng_maxval_ = libxs_matrng_ncols_ * libxs_matrng_ld_; \
    const TYPE libxs_matrng_maxval2_ = (TYPE)((libxs_blasint)LIBXS_UPDIV(libxs_matrng_maxval_, 2)); /* non-zero */ \
    const TYPE libxs_matrng_inv_ = ((TYPE)(SCALE)) / libxs_matrng_maxval2_; \
    const size_t libxs_matrng_shuffle_ = libxs_coprime2((size_t)libxs_matrng_maxval_); \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const libxs_blasint libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((TYPE*)(DST))[libxs_matrng_k_] = libxs_matrng_inv_ * /* normalize values to an interval of [-1, +1] */ \
          ((TYPE)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) - libxs_matrng_maxval2_); \
      } \
    } \
  } \
} while(0)

#define LIBXS_MATRNG(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_ELIDE, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_SEQ(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_OMP(TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_PRAGMA_OMP, TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)

/** GEMM exercising the compiler's code generation. TODO: only NN is supported and SP/DP matrices. */
#define LIBXS_INLINE_XGEMM2(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxs_inline_xgemm_transa_ = (char)(NULL != ((const void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_A & LIBXS_FLAGS) ? 'n' : 't')); \
  const char libxs_inline_xgemm_transb_ = (char)(NULL != ((const void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXS_GEMM_FLAG_TRANS_B & LIBXS_FLAGS) ? 'n' : 't')); \
  const libxs_blasint libxs_inline_xgemm_m_ = *(const libxs_blasint*)(M); /* must be specified */ \
  const libxs_blasint libxs_inline_xgemm_k_ = (NULL != ((const void*)(K)) ? (*(const libxs_blasint*)(K)) : libxs_inline_xgemm_m_); \
  const libxs_blasint libxs_inline_xgemm_n_ = (NULL != ((const void*)(N)) ? (*(const libxs_blasint*)(N)) : libxs_inline_xgemm_k_); \
  const libxs_blasint libxs_inline_xgemm_lda_ = (NULL != ((const void*)(LDA)) ? (*(const libxs_blasint*)(LDA)) : \
    (('n' == libxs_inline_xgemm_transa_ || *"N" == libxs_inline_xgemm_transa_) ? libxs_inline_xgemm_m_ : libxs_inline_xgemm_k_)); \
  const libxs_blasint libxs_inline_xgemm_ldb_ = (NULL != ((const void*)(LDB)) ? (*(const libxs_blasint*)(LDB)) : \
    (('n' == libxs_inline_xgemm_transb_ || *"N" == libxs_inline_xgemm_transb_) ? libxs_inline_xgemm_k_ : libxs_inline_xgemm_n_)); \
  const libxs_blasint libxs_inline_xgemm_ldc_ = (NULL != ((const void*)(LDC)) ? (*(const libxs_blasint*)(LDC)) : libxs_inline_xgemm_m_); \
  const OTYPE libxs_inline_xgemm_alpha_ = (NULL != ((const void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXS_ALPHA)); \
  const OTYPE libxs_inline_xgemm_beta_  = (NULL != ((const void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXS_BETA)); \
  libxs_blasint libxs_inline_xgemm_ni_, libxs_inline_xgemm_mi_ = 0, libxs_inline_xgemm_ki_; /* loop induction variables */ \
  LIBXS_ASSERT('n' == libxs_inline_xgemm_transa_ || *"N" == libxs_inline_xgemm_transa_); \
  LIBXS_ASSERT('n' == libxs_inline_xgemm_transb_ || *"N" == libxs_inline_xgemm_transb_); \
  LIBXS_PRAGMA_SIMD \
  for (libxs_inline_xgemm_mi_ = 0; libxs_inline_xgemm_mi_ < libxs_inline_xgemm_m_; ++libxs_inline_xgemm_mi_) { \
    LIBXS_PRAGMA_LOOP_COUNT(1, LIBXS_CONFIG_MAX_DIM, LIBXS_CONFIG_AVG_DIM) \
    for (libxs_inline_xgemm_ki_ = 0; libxs_inline_xgemm_ki_ < libxs_inline_xgemm_k_; ++libxs_inline_xgemm_ki_) { \
      LIBXS_PRAGMA_UNROLL \
      for (libxs_inline_xgemm_ni_ = 0; libxs_inline_xgemm_ni_ < libxs_inline_xgemm_n_; ++libxs_inline_xgemm_ni_) { \
        ((OTYPE*)(C))[libxs_inline_xgemm_ni_*libxs_inline_xgemm_ldc_+libxs_inline_xgemm_mi_] \
          = ((const ITYPE*)(B))[libxs_inline_xgemm_ni_*libxs_inline_xgemm_ldb_+libxs_inline_xgemm_ki_] * \
           (((const ITYPE*)(A))[libxs_inline_xgemm_ki_*libxs_inline_xgemm_lda_+libxs_inline_xgemm_mi_] * libxs_inline_xgemm_alpha_) \
          + ((const OTYPE*)(C))[libxs_inline_xgemm_ni_*libxs_inline_xgemm_ldc_+libxs_inline_xgemm_mi_] * libxs_inline_xgemm_beta_; \
      } \
    } \
  } \
} while(0)

/** GEMM exercising the compiler's code generation. TODO: only NN is supported and SP/DP matrices. */
#define  LIBXS_INLINE_XGEMM(TYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXS_INLINE_XGEMM2(TYPE, TYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/**
 * Structure of differences with matrix norms according
 * to http://www.netlib.org/lapack/lug/node75.html).
 */
LIBXS_EXTERN_C typedef struct libxs_matdiff_info {
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
  libxs_blasint m, n, i, r;
} libxs_matdiff_info;

/**
 * Utility function to calculate a collection of scalar differences between two matrices (libxs_matdiff_info).
 * The location (m, n) of the largest difference (linf_abs) is recorded (also in case of NaN). In case of NaN,
 * differences are set to infinity. If no difference is discovered, the location (m, n) is negative (OOB).
 * The return value does not judge the difference (norm) between reference and test data, but is about
 * missing support for the requested data-type or otherwise invalid input.
 */
LIBXS_API int libxs_matdiff(libxs_matdiff_info* info,
  libxs_datatype datatype, libxs_blasint m, libxs_blasint n, const void* ref, const void* tst,
  const libxs_blasint* ldref, const libxs_blasint* ldtst);

/**
 * Combine absolute and relative norms into a value which can be used to check against a margin.
 * A file or directory path given per environment variable LIBXS_MATDIFF=/path/to/file stores
 * the epsilon (followed by a line-break), which can be used to calibrate margins of a test case.
 * LIBXS_MATDIFF can carry optional space-separated arguments used to amend the file entry.
 */
LIBXS_API double libxs_matdiff_epsilon(const libxs_matdiff_info* input);
/**
 * Reduces input into output such that the difference is maintained or increased (max function).
 * The very first (initial) output should be zeroed (libxs_matdiff_clear).
 */
LIBXS_API void libxs_matdiff_reduce(libxs_matdiff_info* output, const libxs_matdiff_info* input);
/** Clears the given info-structure, e.g., for the initial reduction-value (libxs_matdiff_reduce). */
LIBXS_API void libxs_matdiff_clear(libxs_matdiff_info* info);

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

/** Convert BF8 to FP32 (scalar). */
LIBXS_API float libxs_convert_bf8_to_f32(libxs_bfloat8 in);
/** Convert HF8 to FP32 (scalar). */
LIBXS_API float libxs_convert_hf8_to_f32(libxs_hfloat8 in);
/** Convert BF16 to FP32 (scalar). */
LIBXS_API float libxs_convert_bf16_to_f32(libxs_bfloat16 in);
/** Convert FP16 to FP32 (scalar). */
LIBXS_API float libxs_convert_f16_to_f32(libxs_float16 in);

/** Convert FP32 to BF16 (scalar). */
LIBXS_API libxs_bfloat16 libxs_convert_f32_to_bf16_truncate(float in);
/** Convert FP32 to BF16 (scalar). */
LIBXS_API libxs_bfloat16 libxs_convert_f32_to_bf16_rnaz(float in);
/** Convert FP32 to BF16 (scalar). */
LIBXS_API libxs_bfloat16 libxs_convert_f32_to_bf16_rne(float in);
/* Convert FP32 to BF8 (scalar). */
LIBXS_API libxs_bfloat8 libxs_convert_f32_to_bf8_stochastic(float in,
  /** Random number may decide rounding direction (boolean/odd/even). */
  unsigned int seed);
/** Convert FP32 to BF8 (scalar). */
LIBXS_API libxs_bfloat8 libxs_convert_f32_to_bf8_rne(float in);
/** Convert FP16 to HF8 (scalar). */
LIBXS_API libxs_hfloat8 libxs_convert_f16_to_hf8_rne(libxs_float16 in);
/** Convert FP32 to HF8 (scalar). */
LIBXS_API libxs_hfloat8 libxs_convert_f32_to_hf8_rne(float in);
/** Convert FP32 to FP16 (scalar). */
LIBXS_API libxs_float16 libxs_convert_f32_to_f16(float in);

/**
 * create a new external state for thread-save execution managed
 * by the user. We do not provide a function for drawing the random numbers
 * the user is supposed to call the LIBXS_INTRINSICS_MM512_RNG_EXTSTATE_PS
 * or LIBXS_INTRINSICS_MM512_RNG_XOSHIRO128P_EXTSTATE_EPI32 intrinsic.
 * */
LIBXS_API unsigned int* libxs_rng_create_extstate(unsigned int/*uint32_t*/ seed);

/**
 * return the size of the state such that users can save it
 * and recreate the same sequence of PRNG numbers.
 */
LIBXS_API unsigned int libxs_rng_get_extstate_size(void);

/** free a previously created rng_avx512_extstate */
LIBXS_API void libxs_rng_destroy_extstate(unsigned int* stateptr);

/** Set the seed of libxs_rng_* (similar to srand). */
LIBXS_API void libxs_rng_set_seed(unsigned int/*uint32_t*/ seed);

/**
 * This SP-RNG is using xoshiro128+ 1.0, work done by
 * David Blackman and Sebastiano Vigna (vigna @ acm.org).
 * It is their best and fastest 32-bit generator for
 * 32-bit floating-point numbers. They suggest to use
 * its upper bits for floating-point generation, what
 * we do here and generate numbers in [0,1(.
 */
LIBXS_API void libxs_rng_f32_seq(float* rngs, libxs_blasint count);

/**
 * Returns a (pseudo-)random value based on rand/rand48 in the interval [0, n).
 * This function compensates for an n, which is not a factor of RAND_MAX.
 * Note: libxs_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXS_API unsigned int libxs_rng_u32(unsigned int n);

/** SQRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_isqrt_u64(unsigned long long x);
/** SQRT with Newton's method using integer arithmetic. */
LIBXS_API unsigned int libxs_isqrt_u32(unsigned int x);
/** Based on libxs_isqrt_u32; result is factor of x. */
LIBXS_API unsigned int libxs_isqrt2_u32(unsigned int x);
/** SQRT with Newton's method using double-precision. */
LIBXS_API double libxs_dsqrt(double x);
/** SQRT with Newton's method using single-precision. */
LIBXS_API float libxs_ssqrt(float x);

#endif /*LIBXS_MATH_H*/
