/******************************************************************************
** Copyright (c) 2019, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
#ifndef LIBXS_RNG_H
#define LIBXS_RNG_H

#include "libxs_typedefs.h"

#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Set the seed of libxs_rng_* (similar to srand). */
LIBXS_API void libxs_rng_set_seed(unsigned int/*uint32_t*/ seed);

/**
 * This SP-RNG is using xoshiro128+ 1.0, work done by
 * David Blackman and Sebastiano Vigna (vigna@acm.org).
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

/**
 * Similar to libxs_rng_u32, but returns a DP-value in the interval [0, 1).
 * Note: libxs_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXS_API double libxs_rng_f64(void);

#endif /* LIBXS_RNG_H */
