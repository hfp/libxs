/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_RNG_H
#define LIBXS_RNG_H

#include "libxs_math.h"

/**
 * Generator version, bumped whenever the produced values change. A campaign records it so
 * rows from different generators are never mixed: they are otherwise indistinguishable.
 * 2: significands are drawn modulo an odd number in both branches, and the ESPAN!=0 ramp
 *    spans -|ESPAN|..+|ESPAN| (rounded) instead of 0..|ESPAN| (truncated).
 */
#define LIBXS_MATRNG_VERSION 2

/**
 * Initialize a column-major matrix with deterministic values.
 * ESPAN==0: shuffle-based init covering the full LD*NCOLS range, values in [-|SCALE|,+|SCALE|].
 * ESPAN!=0: adversarial exponent span for emulation stress-testing, patterned after the
 *           graded BLAS accuracy tests (Demmel et al., BLIS Retreat 2024): base values in
 *           [1,2) scaled by a diagonal of powers of two, column j taking the exponent
 *           -|ESPAN| + round(2*|ESPAN|*j/(NCOLS-1)), i.e. 2*|ESPAN| binades in total.
 *           ESPAN is that construction's b, so it is comparable with published sweeps.
 *           Use +ESPAN for A and -ESPAN for B so that A*B is well-conditioned
 *           but each operand has wide exponent range.
 *           Padding rows [NROWS,LD) are zero-filled.
 * Both branches divide by an ODD modulus, which is what keeps significands full. Dividing by
 * LD*NCOLS instead makes every value dyadic at power-of-two shapes, carrying only
 * log2(LD*NCOLS) significant bits, and an emulator that adapts to spare mantissa bits then
 * measures the generator rather than the data.
 */
#define LIBXS_MATRNG_AUX(OMP, INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE) do { \
  const double libxs_matrng_espan_ = (double)(ESPAN); \
  const INT_TYPE libxs_matrng_nrows_ = (INT_TYPE)(NROWS); \
  const INT_TYPE libxs_matrng_ncols_ = (INT_TYPE)(NCOLS); \
  const INT_TYPE libxs_matrng_ld_ = (INT_TYPE)(LD); \
  const INT_TYPE libxs_matrng_maxval_ = libxs_matrng_ncols_ * libxs_matrng_ld_; \
  const REAL_TYPE libxs_matrng_maxval2_ = (REAL_TYPE)((INT_TYPE)LIBXS_UPDIV(libxs_matrng_maxval_, 2) | 1); \
  const REAL_TYPE libxs_matrng_inv_ = ((REAL_TYPE)(SCALE)) / libxs_matrng_maxval2_; \
  const size_t libxs_matrng_shuffle_ = libxs_coprime2((size_t)libxs_matrng_maxval_); \
  INT_TYPE libxs_matrng_i_ = 0, libxs_matrng_j_ = 0; \
  LIBXS_OMP_VAR(libxs_matrng_i_); LIBXS_OMP_VAR(libxs_matrng_j_); \
  if (0 == libxs_matrng_espan_) { \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_nrows_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = libxs_matrng_inv_ * \
          ((REAL_TYPE)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) - libxs_matrng_maxval2_); \
      } \
      for (; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = 0; \
      } \
    } \
  } \
  else { \
    const double libxs_matrng_sign_ = (0 < libxs_matrng_espan_) ? 1.0 : -1.0; \
    const double libxs_matrng_abspan_ = libxs_matrng_sign_ * libxs_matrng_espan_; \
    const double libxs_matrng_denom_ = (1 < libxs_matrng_ncols_) ? (double)(libxs_matrng_ncols_ - 1) : 1.0; \
    const size_t libxs_matrng_maxodd_ = (size_t)libxs_matrng_maxval_ | 1; \
    const size_t libxs_matrng_shodd_ = libxs_coprime2(libxs_matrng_maxodd_); \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      const double libxs_matrng_exp_ = libxs_matrng_sign_ * (LIBXS_ROUND( \
        2.0 * libxs_matrng_abspan_ * libxs_matrng_i_ / libxs_matrng_denom_) - libxs_matrng_abspan_); \
      const REAL_TYPE libxs_matrng_colscale_ = (REAL_TYPE)ldexp(1.0, (int)libxs_matrng_exp_); \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_nrows_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        const REAL_TYPE libxs_matrng_base_ = (REAL_TYPE)(1.0 + (double)(libxs_matrng_shodd_ \
          * (size_t)libxs_matrng_k_ % libxs_matrng_maxodd_) / (double)libxs_matrng_maxodd_); \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = libxs_matrng_colscale_ * libxs_matrng_base_; \
      } \
      for (; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = 0; \
      } \
    } \
  } \
} while(0)

/** Sequential matrix initialization (see LIBXS_MATRNG_AUX). */
#define LIBXS_MATRNG(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_ELIDE, INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE)
/** Alias for LIBXS_MATRNG (sequential). */
#define LIBXS_MATRNG_SEQ(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE)
/** OpenMP-parallel matrix initialization (see LIBXS_MATRNG_AUX). */
#define LIBXS_MATRNG_OMP(INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_PRAGMA_OMP, INT_TYPE, REAL_TYPE, ESPAN, DST, NROWS, NCOLS, LD, SCALE)


/**
 * Set the seed of the calling thread's PRNG state.
 * Each thread maintains independent state via TLS; calling
 * libxs_rng_set_seed only affects the calling thread.
 * Unseeded threads start with a deterministic default (seed = 1).
 */
LIBXS_API void libxs_rng_set_seed(unsigned int/*uint32_t*/ seed);

/**
 * Returns a (pseudo-)random value in the interval [0, n) with
 * uniform distribution (Lemire's nearly-divisionless method).
 * Thread-safe: each thread has independent PRNG state.
 */
LIBXS_API unsigned int libxs_rng_u32(unsigned int n);

/**
 * Returns a double-precision value in the interval [0, 1) with
 * full 53-bit mantissa resolution.
 * Thread-safe: each thread has independent PRNG state.
 */
LIBXS_API double libxs_rng_f64(void);

/**
 * Fill a buffer with pseudo-random bytes.
 * Thread-safe: each thread has independent PRNG state.
 */
LIBXS_API void libxs_rng_seq(void* data, size_t nbytes);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_RNG_H*/
