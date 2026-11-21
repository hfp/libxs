/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if !defined(INTERNAL_LIBXS_MPI_CHUNK)
# define INTERNAL_LIBXS_MPI_CHUNK 4096 /* small messages: exercise the splitting */
#endif
#include <libxs/libxs_source.h>
#if defined(__MPI)
# include <mpi.h>
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#define NBYTES 10007


#if defined(__MPI)
static unsigned char pattern(size_t i, int rank)
{
  return (unsigned char)((31 * i + 7 * (size_t)rank + 1) & 0xFF);
}


/* Ring of bytes with sizes that differ per rank (and per direction) around the message size. */
static int ring(const internal_libxs_mpi_comm_t* comm, unsigned char* sbuf, unsigned char* rbuf)
{
  const size_t base[] = { 0, 1, 4095, 4096, NBYTES - 64 };
  const int dst = (comm->rank + 1) % comm->size, src = (comm->rank + comm->size - 1) % comm->size;
  int result = EXIT_SUCCESS, n;
  for (n = 0; n < (int)(sizeof(base) / sizeof(*base)) && EXIT_SUCCESS == result; ++n) {
    const size_t nsend = base[n] + (0 != base[n] ? (size_t)comm->rank : 0);
    const size_t nrecv = base[n] + (0 != base[n] ? (size_t)src : 0);
    size_t i;
    for (i = 0; i < nsend; ++i) sbuf[i] = pattern(i, comm->rank);
    memset(rbuf, 0, NBYTES);
    result = internal_libxs_mpi_sendrecv(comm, sbuf, nsend, dst, rbuf, nrecv, src);
    for (i = 0; i < nrecv && EXIT_SUCCESS == result; ++i) {
      if (pattern(i, src) != rbuf[i]) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result) {
      FPRINTF(stderr, "ERROR line #%i: ring of %u bytes\n", __LINE__, (unsigned int)nrecv);
    }
  }
  return result;
}
#endif


/**
 * This test case is NOT an example of how to use LIBXS
 * since INTERNAL functions are tested which are not part
 * of the LIBXS API.
 */
int main(int argc, char* argv[])
{
  const char *const env = getenv("LIBXS_MPI");
  const int disabled = (NULL != env && '0' == env[0] && '\0' == env[1]);
  internal_libxs_mpi_comm_t comm;
  int result = EXIT_SUCCESS, family;
  libxs_init();

  if (INTERNAL_LIBXS_MPI_MPICH != internal_libxs_mpi_classify("Intel(R) MPI Library 2021.19 for Linux* OS")
    || INTERNAL_LIBXS_MPI_MPICH != internal_libxs_mpi_classify("MPICH Version:      4.2.1")
    || INTERNAL_LIBXS_MPI_MPICH != internal_libxs_mpi_classify("MVAPICH2 Version      :\t2.3.7")
    || INTERNAL_LIBXS_MPI_MPICH != internal_libxs_mpi_classify("MPI VERSION    : CRAY MPICH version 8.1.28")
    || INTERNAL_LIBXS_MPI_OMPI != internal_libxs_mpi_classify("Open MPI v5.0.3, package: Open MPI")
    || INTERNAL_LIBXS_MPI_NONE != internal_libxs_mpi_classify("Some MPI 1.0")
    || INTERNAL_LIBXS_MPI_NONE != internal_libxs_mpi_classify(NULL))
  {
    FPRINTF(stderr, "ERROR line #%i: classification of version strings\n", __LINE__);
    result = EXIT_FAILURE;
  }

#if defined(__MPI)
  if (EXIT_SUCCESS == result && MPI_SUCCESS != MPI_Init(&argc, &argv)) result = EXIT_FAILURE;
#endif
  family = internal_libxs_mpi_family();
  FPRINTF(stdout, "MPI family: %i\n", family);

#if defined(__MPI)
  if (EXIT_SUCCESS == result) {
    const int fcomm = (int)MPI_Comm_c2f(MPI_COMM_WORLD);
    const int expected = (0 == disabled ? EXIT_SUCCESS : EXIT_FAILURE);
    if (expected != internal_libxs_mpi_comm(&comm, fcomm)) {
      FPRINTF(stderr, "ERROR line #%i: communicator (family %i)\n", __LINE__, family);
      result = EXIT_FAILURE;
    }
    else if (0 == disabled) {
      int rank = -1, size = -1;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      if (rank != comm.rank || size != comm.size) {
        FPRINTF(stderr, "ERROR line #%i: rank %i of %i instead of %i of %i\n", __LINE__,
          comm.rank, comm.size, rank, size);
        result = EXIT_FAILURE;
      }
      else {
        unsigned char *const sbuf = (unsigned char*)malloc(NBYTES), *const rbuf = (unsigned char*)malloc(NBYTES);
        result = ((NULL != sbuf && NULL != rbuf) ? ring(&comm, sbuf, rbuf) : EXIT_FAILURE);
        free(sbuf);
        free(rbuf);
      }
    }
    if (MPI_SUCCESS != MPI_Finalize()) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result && EXIT_SUCCESS == internal_libxs_mpi_comm(&comm, fcomm)) {
      FPRINTF(stderr, "ERROR line #%i: communicator after MPI_Finalize\n", __LINE__);
      result = EXIT_FAILURE;
    }
  }
#else
  LIBXS_UNUSED(argc);
  LIBXS_UNUSED(argv);
  LIBXS_UNUSED(disabled);
  if (EXIT_SUCCESS == result
    && (INTERNAL_LIBXS_MPI_NONE != family || EXIT_SUCCESS == internal_libxs_mpi_comm(&comm, 0)))
  {
    FPRINTF(stderr, "ERROR line #%i: MPI is not loaded (family %i)\n", __LINE__, family);
    result = EXIT_FAILURE;
  }
#endif

  return result;
}
