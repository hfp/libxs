/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_rng.h>
#include <libxs_mem.h>
#include <libxs_sync.h>


/**
 * SplitMix64 PRNG (Vigna, 2015). Period: 2^64.
 * Self-contained, no libc dependency, excellent statistical quality.
 */
LIBXS_API_INLINE unsigned long long internal_rng_splitmix64(
  unsigned long long* state)
{
  unsigned long long z = (*state += 0x9E3779B97F4A7C15uLL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9uLL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBuLL;
  return z ^ (z >> 31);
}

/** Per-thread PRNG state (TLS when available, otherwise single global). */
static LIBXS_TLS unsigned long long internal_rng_state = 1;


LIBXS_API void libxs_rng_set_seed(unsigned int/*uint32_t*/ seed)
{
  internal_rng_state = (unsigned long long)seed;
}


LIBXS_API unsigned int libxs_rng_u32(unsigned int n)
{
  if (1 < n) {
    /* 64-bit output covers any 32-bit range; use Lemire's fast method */
    const unsigned long long r = internal_rng_splitmix64(&internal_rng_state);
    unsigned long long m = (unsigned long long)(unsigned int)r * n;
    unsigned int leftover = (unsigned int)m;
    if (leftover < n) { /* rejection branch (rare for small n) */
      const unsigned int threshold = (0U - n) % n; /* = (2^32 - n) mod n */
      while (leftover < threshold) {
        m = (unsigned long long)(unsigned int)
          internal_rng_splitmix64(&internal_rng_state) * n;
        leftover = (unsigned int)m;
      }
    }
    return (unsigned int)(m >> 32);
  }
  return 0;
}


LIBXS_API void libxs_rng_seq(void* data, size_t nbytes)
{
  unsigned char* dst = (unsigned char*)data;
  unsigned char* end;
  if (NULL == data) return;
  end = dst + (nbytes & ~(size_t)7);
  for (; dst < end; dst += 8) {
    unsigned long long r = internal_rng_splitmix64(&internal_rng_state);
    LIBXS_MEMCPY(dst, &r, 8);
  }
  end = (unsigned char*)data + nbytes;
  if (dst < end) {
    unsigned long long r = internal_rng_splitmix64(&internal_rng_state);
    const size_t tail = (size_t)(end - dst);
    LIBXS_ASSERT(tail < sizeof(r));
    LIBXS_MEMCPY(dst, &r, tail);
  }
}


LIBXS_API double libxs_rng_f64(void)
{
  /* Use top 53 bits of a 64-bit value for full double mantissa precision.
   * Result is in [0, 1). */
  return (double)(internal_rng_splitmix64(&internal_rng_state) >> 11)
    * (1.0 / 9007199254740992.0); /* 1 / 2^53 */
}
