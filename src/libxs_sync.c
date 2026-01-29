/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs_sync.h>
#include "libxs_main.h"

#if !defined(LIBXS_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
# define LIBXS_SYNC_FUTEX
#endif

#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# if defined(LIBXS_SYNC_FUTEX) && defined(__linux__) && defined(__USE_GNU)
#   include <linux/futex.h>
# endif
# include <time.h>
#endif


LIBXS_API unsigned int libxs_get_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXS_API_INTERN unsigned int internal_get_tid(void);
LIBXS_API_INTERN unsigned int internal_get_tid(void)
{
  const unsigned int nthreads = LIBXS_ATOMIC_ADD_FETCH(&libxs_thread_count, 1, LIBXS_ATOMIC_RELAXED);
#if !defined(NDEBUG)
  static int error_once = 0;
  if (LIBXS_NTHREADS_MAX < nthreads
    && 0 != libxs_verbosity /* library code is expected to be mute */
    && 1 == LIBXS_ATOMIC_ADD_FETCH(&error_once, 1, LIBXS_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXS ERROR: maximum number of threads is exhausted!\n");
  }
#endif
  LIBXS_ASSERT(LIBXS_ISPOT(LIBXS_NTHREADS_MAX));
  return LIBXS_MOD2(nthreads - 1, LIBXS_NTHREADS_MAX);
}


LIBXS_API unsigned int libxs_get_tid(void)
{
#if (0 != LIBXS_SYNC)
# if defined(_OPENMP) && defined(LIBXS_SYNC_OMP)
  return (unsigned int)omp_get_thread_num();
# else
  static LIBXS_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = internal_get_tid();
  return tid;
# endif
#else
  return 0;
#endif
}


LIBXS_API void libxs_stdio_acquire(void)
{
#if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    flock(libxs_stdio_handle - 1, LOCK_EX);
  }
  else
#endif
  {
    LIBXS_FLOCK(stdout);
    LIBXS_FLOCK(stderr);
  }
}


LIBXS_API void libxs_stdio_release(void)
{
#if !defined(_WIN32)
  if (0 < libxs_stdio_handle) {
    flock(libxs_stdio_handle - 1, LOCK_UN);
  }
  else
#endif
  {
    LIBXS_FUNLOCK(stderr);
    LIBXS_FUNLOCK(stdout);
  }
}
