/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
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
 * Initialize a column-major matrix with deterministic values.
 * ESPAN==0: shuffle-based init covering the full LD*NCOLS range, values in [-|SCALE|,+|SCALE|].
 * ESPAN!=0: adversarial exponent span for emulation stress-testing.
 *           Base values are shuffled in [1,2), then column j is scaled by
 *           2^(|ESPAN|*j/(NCOLS-1)) when ESPAN>0, or by
 *           2^(-|ESPAN|*j/(NCOLS-1)) when ESPAN<0.
 *           Use +ESPAN for A and -ESPAN for B so that A*B is well-conditioned
 *           but each operand has wide exponent range.
 *           Padding rows [NROWS,LD) are zero-filled.
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
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = libxs_matrng_inv_ * \
          ((REAL_TYPE)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) - libxs_matrng_maxval2_); \
      } \
    } \
  } \
  else { \
    const double libxs_matrng_sign_ = (0 < libxs_matrng_espan_) ? 1.0 : -1.0; \
    const double libxs_matrng_abspan_ = libxs_matrng_sign_ * libxs_matrng_espan_; \
    const double libxs_matrng_denom_ = (1 < libxs_matrng_ncols_) ? (double)(libxs_matrng_ncols_ - 1) : 1.0; \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      const double libxs_matrng_exp_ = libxs_matrng_sign_ * \
        floor(libxs_matrng_abspan_ * libxs_matrng_i_ / libxs_matrng_denom_); \
      const REAL_TYPE libxs_matrng_colscale_ = (REAL_TYPE)ldexp(1.0, (int)libxs_matrng_exp_); \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_nrows_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        const REAL_TYPE libxs_matrng_base_ = (REAL_TYPE)(1.0 + \
          (double)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) / libxs_matrng_maxval_); \
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
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_RNG_H*/
