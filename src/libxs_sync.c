/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
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
  /* Intel MPI and MPICH, Open MPI, MVAPICH2 */
  const char* env_nranks = getenv("MPI_LOCALNRANKS");
  int nranks;
  if (NULL == env_nranks) env_nranks = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (NULL == env_nranks) env_nranks = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  nranks = (NULL == env_nranks ? 1 : atoi(env_nranks));
  return (unsigned int)LIBXS_MAX(nranks, 1);
}


LIBXS_API unsigned int libxs_nrank(void)
{
  /* the global PMI_RANK is the last resort: it is local only if ranks are placed in blocks */
  const char* env_rank = getenv("MPI_LOCALRANKID");
  int nrank;
  if (NULL == env_rank) env_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (NULL == env_rank) env_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (NULL == env_rank) env_rank = getenv("SLURM_LOCALID");
  if (NULL == env_rank) env_rank = getenv("PMI_RANK");
  nrank = (NULL == env_rank ? 0 : atoi(env_rank));
  return (unsigned int)LIBXS_MAX(nrank, 0) % libxs_nranks();
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
  static LIBXS_TLS unsigned int tid = 0xFFFFFFFF;
  if (0xFFFFFFFF == tid) tid = LIBXS_ATOMIC_ADD_FETCH(&libxs_thread_count, 1, LIBXS_ATOMIC_RELAXED) - 1;
  return tid;
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
