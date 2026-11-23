/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MPI_H
#define LIBXS_MPI_H

#include "libxs_main.h"

#define INTERNAL_LIBXS_MPI_NONE 0
#define INTERNAL_LIBXS_MPI_MPICH 1
#define INTERNAL_LIBXS_MPI_OMPI 2


/* An MPI handle without mpi.h: an integer (MPICH family) or a pointer (Open MPI) */
typedef union internal_libxs_mpi_handle_t {
  int i;
  void* p;
} internal_libxs_mpi_handle_t;

typedef struct internal_libxs_mpi_comm_t {
  internal_libxs_mpi_handle_t comm;
  int rank, size;
} internal_libxs_mpi_comm_t;


/**
 * Family of the MPI library found in the process (INTERNAL_LIBXS_MPI_*), resolved once at the first call:
 * symbols and the library's version string decide, the environment resolves an unknown version string.
 * LIBXS_MPI=0 disables the layer, LIBXS_MPI=mpich or LIBXS_MPI=ompi forces a family.
 */
LIBXS_API_INTERN int internal_libxs_mpi_family(void);
#if defined(LIBXS_INTERCEPT_DYNAMIC)
/** Symbol in global scope (MPI is loaded by the application, possibly before this library), or NULL. */
LIBXS_API_INTERN void* internal_libxs_mpi_dlsym(const char name[]);
#endif
/** Family named by a version string (MPI_Get_library_version), or INTERNAL_LIBXS_MPI_NONE if unknown. */
LIBXS_API_INTERN int internal_libxs_mpi_classify(const char version[]);
/**
 * The communicator given by its Fortran handle (MPI_Comm_c2f), the same integer in every MPI.
 * EXIT_SUCCESS, or an error without (known) MPI, before MPI_Init, or after MPI_Finalize.
 */
LIBXS_API_INTERN int internal_libxs_mpi_comm(internal_libxs_mpi_comm_t* comm, int fcomm);
/** Replaces comm by a duplicate, which keeps the traffic apart (collective, see internal_libxs_mpi_free). */
LIBXS_API_INTERN int internal_libxs_mpi_dup(internal_libxs_mpi_comm_t* comm);
/** Frees a duplicate (collective), unless MPI is finalized. */
LIBXS_API_INTERN void internal_libxs_mpi_free(internal_libxs_mpi_comm_t* comm);
/** Non-zero if any thread may call MPI, one at a time (MPI_THREAD_SERIALIZED or higher). */
LIBXS_API_INTERN int internal_libxs_mpi_serialized(void);
/** Non-zero if the caller is the thread that initialized MPI. */
LIBXS_API_INTERN int internal_libxs_mpi_main_thread(void);
/**
 * Blocking exchange of bytes (a ring step): send to dst and receive from src, either of which may be
 * the caller. The traffic has a reserved tag, and a communicator of its own if that is a duplicate
 * (internal_libxs_mpi_dup). Both sides must agree on the sizes. Beyond INT_MAX bytes, an exchange is
 * split into messages of equal size.
 */
LIBXS_API_INTERN int internal_libxs_mpi_sendrecv(const internal_libxs_mpi_comm_t* comm,
  const void* sbuf, size_t nsend, int dst, void* rbuf, size_t nrecv, int src);
/** Collective logical AND of flag over the ranks (a ring), e.g., to agree on a decision. */
LIBXS_API_INTERN int internal_libxs_mpi_and(const internal_libxs_mpi_comm_t* comm, int* flag);

#endif /*LIBXS_MPI_H*/
