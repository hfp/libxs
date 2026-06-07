/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_sync.h>
#include "libxs_main.h"

#include <stdint.h>
#if defined(_WIN32)
# include <process.h>
#else
# include <sys/file.h>
# include <time.h>
#endif


LIBXS_API unsigned int libxs_nranks(void)
{
  const char *const env_nranks = getenv("MPI_LOCALNRANKS"); /* TODO */
  return LIBXS_MAX(NULL == env_nranks ? 1 : atoi(env_nranks), 1);
}


LIBXS_API unsigned int libxs_nrank(void)
{
  const char *const env_rank = (NULL != getenv("PMI_RANK")
    ? getenv("PMI_RANK") : getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  return (NULL == env_rank ? 0 : atoi(env_rank)) % libxs_nranks();
}


LIBXS_API unsigned int libxs_rid(void)
{
  return 1 < libxs_nranks() ? libxs_nrank() : libxs_pid();
}


LIBXS_API unsigned int libxs_pid(void)
{
#if defined(_WIN32)
  return (unsigned int)_getpid();
#else
  return (unsigned int)getpid();
#endif
}


LIBXS_API unsigned int libxs_tid(void)
{
#if (0 != LIBXS_SYNC)
# if defined(_OPENMP) && defined(LIBXS_SYNC_OMP)
  return (unsigned int)omp_get_thread_num();
# else
  static LIBXS_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = LIBXS_ATOMIC_ADD_FETCH(&libxs_thread_count, 1, LIBXS_ATOMIC_RELAXED) - 1;
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
