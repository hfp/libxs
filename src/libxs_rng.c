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

/** Denote quality of scalar random number generator. */
#if !defined(LIBXS_RNG_DRAND48) && !defined(_WIN32) && !defined(__CYGWIN__) && \
    (defined(_DEFAULT_SOURCE) || defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
# define LIBXS_RNG_DRAND48
#endif


LIBXS_API void libxs_rng_set_seed(unsigned int/*uint32_t*/ seed)
{
  /* for consistency, other RNGs are seeded as well */
#if defined(LIBXS_RNG_DRAND48)
  srand48(seed);
#endif
  srand(seed);
}


LIBXS_API unsigned int libxs_rng_u32(unsigned int n)
{
  unsigned int result;
  if (1 < n) {
#if defined(LIBXS_RNG_DRAND48)
    const unsigned int rmax = (1U << 31); /* lrand48 returns [0, 2^31) */
    unsigned int r = (unsigned int)lrand48();
#else
    /* rand() returns [0, RAND_MAX]; guard against RAND_MAX+1 overflow to zero */
    const unsigned int rmax = ((unsigned int)RAND_MAX < ~0U)
      ? ((unsigned int)RAND_MAX + 1U) : ~0U;
    unsigned int r = (unsigned int)rand();
#endif
    const unsigned int nmax = LIBXS_MIN(n, rmax);
    const unsigned int q = (rmax / nmax) * nmax;
#if defined(LIBXS_RNG_DRAND48)
    /* coverity[dont_call] */
    while (q <= r) r = (unsigned int)lrand48();
#else
    while (q <= r) r = (unsigned int)rand();
#endif
    if (n <= nmax) result = r % nmax;
    else { /* input range exhausts RNG-state (precision): combine two calls */
      const unsigned int r2 =
#if defined(LIBXS_RNG_DRAND48)
        (unsigned int)lrand48();
#else
        (unsigned int)rand();
#endif
      /* use wide multiply: (r * n + r2) mod n, with r already in [0, nmax) */
      result = (unsigned int)(((unsigned long long)r * n / nmax + r2) % n);
    }
  }
  else result = 0;
  return result;
}


LIBXS_API void libxs_rng_seq(void* data, size_t nbytes)
{
  unsigned char* dst = (unsigned char*)data;
  unsigned char* end;
  unsigned int r;
  if (NULL == data) return;
  end = dst + (nbytes & ~(size_t)3);
  for (; dst < end; dst += 4) {
#if defined(LIBXS_RNG_DRAND48)
    /* coverity[dont_call] */
    r = (unsigned int)lrand48();
#else
    r = (unsigned int)rand();
#endif
    LIBXS_MEMCPY(dst, &r, 4);
  }
  end = (unsigned char*)data + nbytes;
  if (dst < end) {
    const size_t size = end - dst;
#if defined(LIBXS_RNG_DRAND48)
    r = (unsigned int)lrand48();
#else
    r = (unsigned int)rand();
#endif
    LIBXS_ASSERT(size < sizeof(r));
    LIBXS_MEMCPY(dst, &r, size);
  }
}


LIBXS_API double libxs_rng_f64(void)
{
#if defined(LIBXS_RNG_DRAND48)
  /* coverity[dont_call] */
  return drand48(); /* drand48 returns [0, 1) */
#else
  /* RAND_MAX + 1.0 avoids integer overflow and guarantees [0, 1) */
  return (double)rand() / ((double)RAND_MAX + 1.0);
#endif
}
