/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_source.h>

#if !defined(ATOMIC_KIND)
# define ATOMIC_KIND LIBXS_ATOMIC_LOCKORDER
#endif

#if !defined(PRINT) && (defined(_DEBUG) || 0)
# define PRINT
#endif
#if defined(PRINT)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  LIBXS_ALIGNED(LIBXS_ATOMIC_LOCKTYPE lock = 0/*unlocked*/, LIBXS_ALIGNMENT);
  int result = EXIT_SUCCESS;
  int mh = 1051981, hp, tmp;

  LIBXS_NONATOMIC_STORE(&hp, 25071975, ATOMIC_KIND);
  tmp = LIBXS_NONATOMIC_LOAD(&hp, ATOMIC_KIND);
  if (tmp != LIBXS_ATOMIC_LOAD(&hp, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (mh != LIBXS_NONATOMIC_SUB_FETCH(&hp, 24019994, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (mh != LIBXS_ATOMIC_FETCH_ADD(&hp, 24019994, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  LIBXS_ATOMIC_STORE(&tmp, mh, ATOMIC_KIND);
  if (25071975 != LIBXS_NONATOMIC_FETCH_OR(&hp, tmp, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if ((25071975 | mh) != hp) {
    result = EXIT_FAILURE;
  }
  /* check if non-atomic and atomic are compatible */
  if (LIBXS_NONATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
    if (LIBXS_ATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
      result = EXIT_FAILURE;
    }
    LIBXS_NONATOMIC_RELEASE(&lock, ATOMIC_KIND);
    if (0 != lock) result = EXIT_FAILURE;
  }
  else {
    result = EXIT_FAILURE;
  }

  LIBXS_ATOMIC_ACQUIRE(&lock, LIBXS_SYNC_NPAUSE, ATOMIC_KIND);
  if (0 == lock) result = EXIT_FAILURE;
  if (LIBXS_ATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (LIBXS_ATOMIC_TRYLOCK(&lock, ATOMIC_KIND)) {
    result = EXIT_FAILURE;
  }
  if (0 == lock) result = EXIT_FAILURE;
  LIBXS_ATOMIC_RELEASE(&lock, ATOMIC_KIND);
  if (0 != lock) result = EXIT_FAILURE;

  /* check LIBXS_ATOMIC_ADD_FETCH */
  if (EXIT_SUCCESS == result) {
    int val = 10;
    int r = LIBXS_ATOMIC_ADD_FETCH(&val, 5, ATOMIC_KIND);
    if (15 != r || 15 != val) {
      FPRINTF(stderr, "ERROR line #%i: ATOMIC_ADD_FETCH r=%i val=%i\n", __LINE__, r, val);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ATOMIC_SUB_FETCH */
  if (EXIT_SUCCESS == result) {
    int val = 20;
    int r = LIBXS_ATOMIC_SUB_FETCH(&val, 7, ATOMIC_KIND);
    if (13 != r || 13 != val) {
      FPRINTF(stderr, "ERROR line #%i: ATOMIC_SUB_FETCH r=%i val=%i\n", __LINE__, r, val);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ATOMIC_FETCH_SUB (returns old value) */
  if (EXIT_SUCCESS == result) {
    int val = 100;
    int old = LIBXS_ATOMIC_FETCH_SUB(&val, 30, ATOMIC_KIND);
    if (100 != old || 70 != val) {
      FPRINTF(stderr, "ERROR line #%i: ATOMIC_FETCH_SUB old=%i val=%i\n", __LINE__, old, val);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ATOMIC_CMPSWP - successful swap */
  if (EXIT_SUCCESS == result) {
    int val = 42;
    if (!LIBXS_ATOMIC_CMPSWP(&val, 42, 99, ATOMIC_KIND)) {
      FPRINTF(stderr, "ERROR line #%i: CMPSWP should succeed\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 99 != val) {
      FPRINTF(stderr, "ERROR line #%i: CMPSWP val=%i expected 99\n", __LINE__, val);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ATOMIC_CMPSWP - failed swap (value mismatch) */
  if (EXIT_SUCCESS == result) {
    int val = 10;
    if (LIBXS_ATOMIC_CMPSWP(&val, 20, 30, ATOMIC_KIND)) {
      FPRINTF(stderr, "ERROR line #%i: CMPSWP should fail\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 10 != val) {
      FPRINTF(stderr, "ERROR line #%i: CMPSWP should not modify val=%i\n", __LINE__, val);
      result = EXIT_FAILURE;
    }
  }

  /* CMPSWP regression: must fail even when OLDVAL == NEWVAL and *dst != OLDVAL */
  if (EXIT_SUCCESS == result) {
    int val = 5;
    if (LIBXS_ATOMIC_CMPSWP(&val, 7, 7, ATOMIC_KIND)) {
      FPRINTF(stderr, "ERROR line #%i: CMPSWP(5,7,7) should fail\n", __LINE__);
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result && 5 != val) {
      FPRINTF(stderr, "ERROR line #%i: CMPSWP should not modify val=%i\n", __LINE__, val);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ATOMIC_STORE_ZERO */
  if (EXIT_SUCCESS == result) {
    int val = 12345;
    LIBXS_ATOMIC_STORE_ZERO(&val, ATOMIC_KIND);
    if (0 != val) {
      FPRINTF(stderr, "ERROR line #%i: STORE_ZERO val=%i\n", __LINE__, val);
      result = EXIT_FAILURE;
    }
  }

  /* check LIBXS_ATOMIC_SYNC (just verify it compiles and doesn't crash) */
  LIBXS_ATOMIC_SYNC(ATOMIC_KIND);

  return result;
}
