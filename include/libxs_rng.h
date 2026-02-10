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

#include <libxs_math.h>

/** Helper macro to setup a matrix with some initial values. */
#define LIBXS_MATRNG_AUX(OMP, INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) do { \
  /*const*/ double libxs_matrng_seed_ = SEED; /* avoid constant conditional */ \
  const double libxs_matrng_scale_ = libxs_matrng_seed_ * (SCALE) + (SCALE); \
  const INT_TYPE libxs_matrng_nrows_ = (INT_TYPE)(NROWS); \
  const INT_TYPE libxs_matrng_ncols_ = (INT_TYPE)(NCOLS); \
  const INT_TYPE libxs_matrng_ld_ = (INT_TYPE)(LD); \
  INT_TYPE libxs_matrng_i_ = 0, libxs_matrng_j_ = 0; \
  LIBXS_OMP_VAR(libxs_matrng_i_); LIBXS_OMP_VAR(libxs_matrng_j_); \
  if (0 != libxs_matrng_seed_) { \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_nrows_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = (REAL_TYPE)(libxs_matrng_scale_ * (1.0 + \
          (double)libxs_matrng_i_ * libxs_matrng_nrows_ + libxs_matrng_j_)); \
      } \
      for (; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = (REAL_TYPE)libxs_matrng_seed_; \
      } \
    } \
  } \
  else { /* shuffle based initialization */ \
    const INT_TYPE libxs_matrng_maxval_ = libxs_matrng_ncols_ * libxs_matrng_ld_; \
    const REAL_TYPE libxs_matrng_maxval2_ = (REAL_TYPE)((INT_TYPE)LIBXS_UPDIV(libxs_matrng_maxval_, 2)); /* non-zero */ \
    const REAL_TYPE libxs_matrng_inv_ = ((REAL_TYPE)(SCALE)) / libxs_matrng_maxval2_; \
    const size_t libxs_matrng_shuffle_ = libxs_coprime2((size_t)libxs_matrng_maxval_); \
    OMP(parallel for private(libxs_matrng_i_, libxs_matrng_j_)) \
    for (libxs_matrng_i_ = 0; libxs_matrng_i_ < libxs_matrng_ncols_; ++libxs_matrng_i_) { \
      for (libxs_matrng_j_ = 0; libxs_matrng_j_ < libxs_matrng_ld_; ++libxs_matrng_j_) { \
        const INT_TYPE libxs_matrng_k_ = libxs_matrng_i_ * libxs_matrng_ld_ + libxs_matrng_j_; \
        ((REAL_TYPE*)(DST))[libxs_matrng_k_] = libxs_matrng_inv_ * /* normalize values to an interval of [-1, +1] */ \
          ((REAL_TYPE)(libxs_matrng_shuffle_ * libxs_matrng_k_ % libxs_matrng_maxval_) - libxs_matrng_maxval2_); \
      } \
    } \
  } \
} while(0)

#define LIBXS_MATRNG(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_ELIDE, INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_SEQ(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)
#define LIBXS_MATRNG_OMP(INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE) \
  LIBXS_MATRNG_AUX(LIBXS_PRAGMA_OMP, INT_TYPE, REAL_TYPE, SEED, DST, NROWS, NCOLS, LD, SCALE)


/** Set the seed of libxs_rng_* (similar to srand). */
LIBXS_API void libxs_rng_set_seed(unsigned int/*uint32_t*/ seed);

/**
 * Returns a (pseudo-)random value based on rand/rand48 in the interval [0, n).
 * This function compensates for an n, which is not a factor of RAND_MAX.
 * Note: libxs_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXS_API unsigned int libxs_rng_u32(unsigned int n);

/**
 * Similar to libxs_rng_u32, but returns a DP-value in the interval [0, 1).
 * Note: libxs_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXS_API double libxs_rng_f64(void);

/** Sequence of random data based on libxs_rng_u32. */
LIBXS_API void libxs_rng_seq(void* data, size_t nbytes);

#endif /*LIBXS_RNG_H*/
