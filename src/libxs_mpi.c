/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <libxs/libxs_str.h>
#include "libxs_mpi.h"

#if !defined(INTERNAL_LIBXS_MPI_TAG)
# define INTERNAL_LIBXS_MPI_TAG 22611 /* reserved for the layer (MPI_TAG_UB is at least 32767) */
#endif
#if !defined(INTERNAL_LIBXS_MPI_CHUNK)
# define INTERNAL_LIBXS_MPI_CHUNK INT_MAX /* most bytes per message: MPI counts are int */
#endif
#if (INTERNAL_LIBXS_MPI_CHUNK < LIBXS_CACHELINE)
# error "INTERNAL_LIBXS_MPI_CHUNK must hold a cache line"
#endif
/* a larger exchange is split into equal pieces (cache-line multiples but the last) */
#define INTERNAL_LIBXS_MPI_PIECE_MAX LIBXS_LO2(INTERNAL_LIBXS_MPI_CHUNK, LIBXS_CACHELINE)
/* MPI_MAX_LIBRARY_VERSION_STRING: 8192 (MPICH), 256 (Open MPI) */
#define INTERNAL_LIBXS_MPI_VERSION_MAX 8192
/* MPICH ABI (MPICH, Intel MPI, MVAPICH, Cray MPICH): mpi.h of MPICH 4.2 */
#define INTERNAL_LIBXS_MPICH_BYTE 0x4c00010d
#define INTERNAL_LIBXS_MPICH_STATUS_IGNORE ((void*)1)
/* MPI_THREAD_SERIALIZED, the same value in either family */
#define INTERNAL_LIBXS_MPI_SERIALIZED 2


/* Prototypes of the MPI functions used: handles are int (MPICH family) or pointers (Open MPI) */
typedef int (*internal_libxs_mpi_query_t)(int*);
typedef int (*internal_libxs_mpi_version_t)(char*, int*);
typedef int (*internal_libxs_mpi_abi_t)(int*, int*);
typedef void* (*internal_libxs_mpi_f2c_t)(int);
typedef int (*internal_libxs_mpi_info_i_t)(int, int*);
typedef int (*internal_libxs_mpi_info_p_t)(void*, int*);
typedef int (*internal_libxs_mpi_isend_i_t)(const void*, int, int, int, int, int, int*);
typedef int (*internal_libxs_mpi_isend_p_t)(const void*, int, void*, int, int, void*, void**);
typedef int (*internal_libxs_mpi_irecv_i_t)(void*, int, int, int, int, int, int*);
typedef int (*internal_libxs_mpi_irecv_p_t)(void*, int, void*, int, int, void*, void**);
typedef int (*internal_libxs_mpi_wait_i_t)(int*, void*);
typedef int (*internal_libxs_mpi_wait_p_t)(void**, void*);
typedef int (*internal_libxs_mpi_dup_i_t)(int, int*);
typedef int (*internal_libxs_mpi_dup_p_t)(void*, void**);
typedef int (*internal_libxs_mpi_free_i_t)(int*);
typedef int (*internal_libxs_mpi_free_p_t)(void**);

typedef struct internal_libxs_mpi_t {
  internal_libxs_mpi_query_t initialized, finalized, query_thread, is_main_thread;
  internal_libxs_mpi_f2c_t f2c;
  union { internal_libxs_mpi_info_i_t i; internal_libxs_mpi_info_p_t p; } rank, size;
  union { internal_libxs_mpi_isend_i_t i; internal_libxs_mpi_isend_p_t p; } isend;
  union { internal_libxs_mpi_irecv_i_t i; internal_libxs_mpi_irecv_p_t p; } irecv;
  union { internal_libxs_mpi_wait_i_t i; internal_libxs_mpi_wait_p_t p; } wait;
  union { internal_libxs_mpi_dup_i_t i; internal_libxs_mpi_dup_p_t p; } dup;
  union { internal_libxs_mpi_free_i_t i; internal_libxs_mpi_free_p_t p; } release;
  internal_libxs_mpi_handle_t byte;
  void* status_ignore;
  int family, resolved;
} internal_libxs_mpi_t;


LIBXS_APIVAR_DEFINE(internal_libxs_mpi_t internal_libxs_mpi);


#if defined(LIBXS_INTERCEPT_DYNAMIC)
LIBXS_API_INTERN void* internal_libxs_mpi_dlsym(const char name[])
{
  void* result;
  dlerror();
  result = dlsym(LIBXS_RTLD_DEFAULT, name);
  if (NULL != dlerror()) result = NULL;
  return result;
}


/* Launcher variables: a hint only, since they describe the launcher rather than the library. */
LIBXS_API_INLINE int internal_libxs_mpi_hint(void)
{
  int result = INTERNAL_LIBXS_MPI_NONE;
  if (NULL != getenv("OMPI_COMM_WORLD_SIZE")) result = INTERNAL_LIBXS_MPI_OMPI;
  else if (NULL != getenv("I_MPI_ROOT") || NULL != getenv("MV2_COMM_WORLD_SIZE")
    || NULL != getenv("CRAY_MPICH_DIR") || NULL != getenv("PMI_SIZE"))
  {
    result = INTERNAL_LIBXS_MPI_MPICH;
  }
  return result;
}


/* Family by vote: the symbols and the version string decide, the environment resolves an unknown string. */
LIBXS_API_INLINE int internal_libxs_mpi_vote(const char* env, internal_libxs_mpi_t* mpi, const char** reason)
{
  const int forced = (NULL == env ? INTERNAL_LIBXS_MPI_NONE
    : ((NULL != libxs_stristr(env, "ompi") || NULL != libxs_stristr(env, "open")) ? INTERNAL_LIBXS_MPI_OMPI
    : (NULL != libxs_stristr(env, "mpich") ? INTERNAL_LIBXS_MPI_MPICH : INTERNAL_LIBXS_MPI_NONE)));
  void *const vinit = internal_libxs_mpi_dlsym("MPI_Initialized");
  void *const vfini = internal_libxs_mpi_dlsym("MPI_Finalized");
  void *const vversion = internal_libxs_mpi_dlsym("MPI_Get_library_version");
  void *const vrank = internal_libxs_mpi_dlsym("MPI_Comm_rank");
  void *const vsize = internal_libxs_mpi_dlsym("MPI_Comm_size");
  void *const visend = internal_libxs_mpi_dlsym("MPI_Isend");
  void *const virecv = internal_libxs_mpi_dlsym("MPI_Irecv");
  void *const vwait = internal_libxs_mpi_dlsym("MPI_Wait");
  void *const vdup = internal_libxs_mpi_dlsym("MPI_Comm_dup");
  void *const vfree = internal_libxs_mpi_dlsym("MPI_Comm_free");
  void *const vquery = internal_libxs_mpi_dlsym("MPI_Query_thread");
  void *const vmain = internal_libxs_mpi_dlsym("MPI_Is_thread_main");
  int result = INTERNAL_LIBXS_MPI_NONE;
  if (NULL != vinit && NULL != vfini && NULL != vversion && NULL != vrank && NULL != vsize
    && NULL != visend && NULL != virecv && NULL != vwait
    && NULL != vdup && NULL != vfree && NULL != vquery && NULL != vmain)
  {
    void *const vabi = internal_libxs_mpi_dlsym("MPI_Abi_get_version");
    void *const vbyte = internal_libxs_mpi_dlsym("ompi_mpi_byte");
    void *const vf2c = internal_libxs_mpi_dlsym("MPI_Comm_f2c");
    const int symbols = ((NULL != vbyte && NULL != vf2c) ? INTERNAL_LIBXS_MPI_OMPI : INTERNAL_LIBXS_MPI_MPICH);
    internal_libxs_mpi_version_t version_fn;
    char version[INTERNAL_LIBXS_MPI_VERSION_MAX];
    int named = INTERNAL_LIBXS_MPI_NONE, len = 0;
    if (NULL != vabi) { /* MPI 5.0: the standard ABI has handles of its own (not supported yet) */
      internal_libxs_mpi_abi_t abi_fn;
      int major = 0, minor = 0;
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_abi_t, abi_fn, vabi);
      if (0 == abi_fn(&major, &minor) && 0 < major) *reason = "standard ABI";
    }
    LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_version_t, version_fn, vversion);
    if (0 == version_fn(version, &len) && 0 < len) {
      version[LIBXS_MIN(len, INTERNAL_LIBXS_MPI_VERSION_MAX - 1)] = '\0';
      named = internal_libxs_mpi_classify(version);
    }
    if (NULL != *reason) result = INTERNAL_LIBXS_MPI_NONE;
    else if (INTERNAL_LIBXS_MPI_NONE != forced) {
      result = (symbols == forced ? forced : INTERNAL_LIBXS_MPI_NONE);
      if (INTERNAL_LIBXS_MPI_NONE == result) *reason = "LIBXS_MPI contradicts the symbols";
    }
    else if (named == symbols) result = symbols;
    else if (INTERNAL_LIBXS_MPI_NONE == named
      && (INTERNAL_LIBXS_MPI_OMPI == symbols || INTERNAL_LIBXS_MPI_MPICH == internal_libxs_mpi_hint()))
    {
      result = symbols; /* Open MPI's handles come from its symbols, nothing is assumed */
    }
    else *reason = "unknown or contradicting MPI library";
    if (INTERNAL_LIBXS_MPI_NONE != result) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_query_t, mpi->initialized, vinit);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_query_t, mpi->finalized, vfini);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_query_t, mpi->query_thread, vquery);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_query_t, mpi->is_main_thread, vmain);
    }
    if (INTERNAL_LIBXS_MPI_MPICH == result) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_info_i_t, mpi->rank.i, vrank);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_info_i_t, mpi->size.i, vsize);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_isend_i_t, mpi->isend.i, visend);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_irecv_i_t, mpi->irecv.i, virecv);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_wait_i_t, mpi->wait.i, vwait);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_dup_i_t, mpi->dup.i, vdup);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_free_i_t, mpi->release.i, vfree);
      mpi->byte.i = INTERNAL_LIBXS_MPICH_BYTE;
      mpi->status_ignore = INTERNAL_LIBXS_MPICH_STATUS_IGNORE;
    }
    else if (INTERNAL_LIBXS_MPI_OMPI == result) {
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_f2c_t, mpi->f2c, vf2c);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_info_p_t, mpi->rank.p, vrank);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_info_p_t, mpi->size.p, vsize);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_isend_p_t, mpi->isend.p, visend);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_irecv_p_t, mpi->irecv.p, virecv);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_wait_p_t, mpi->wait.p, vwait);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_dup_p_t, mpi->dup.p, vdup);
      LIBXS_FPTR_FROM_VPTR(internal_libxs_mpi_free_p_t, mpi->release.p, vfree);
      mpi->byte.p = vbyte;
      mpi->status_ignore = NULL;
    }
  }
  return result;
}
#endif


LIBXS_API_INTERN int internal_libxs_mpi_classify(const char version[])
{
  int result = INTERNAL_LIBXS_MPI_NONE;
  if (NULL != version) {
    if (NULL != libxs_stristr(version, "Open MPI")) result = INTERNAL_LIBXS_MPI_OMPI;
    else if (NULL != libxs_stristr(version, "MPICH") || NULL != libxs_stristr(version, "MVAPICH")
      || NULL != libxs_stristr(version, "Intel(R) MPI"))
    {
      result = INTERNAL_LIBXS_MPI_MPICH;
    }
  }
  return result;
}


LIBXS_API_INTERN int internal_libxs_mpi_family(void)
{
  if (0 == internal_libxs_mpi.resolved) { /* idempotent: a concurrent first call resolves the same */
    internal_libxs_mpi_t mpi;
    memset(&mpi, 0, sizeof(mpi));
#if defined(LIBXS_INTERCEPT_DYNAMIC)
    { const char *const env = getenv("LIBXS_MPI");
      const char* reason = NULL;
      if (NULL == env || '0' != env[0] || '\0' != env[1]) {
        mpi.family = internal_libxs_mpi_vote(env, &mpi, &reason);
      }
      if (NULL != reason && (LIBXS_VERBOSITY_WARN <= libxs_verbosity || 0 > libxs_verbosity)) {
        fprintf(stderr, "LIBXS WARNING: MPI is not used (%s).\n", reason);
      }
    }
#endif
    mpi.resolved = 1;
    internal_libxs_mpi = mpi;
  }
  return internal_libxs_mpi.family;
}


LIBXS_API_INTERN int internal_libxs_mpi_comm(internal_libxs_mpi_comm_t* comm, int fcomm)
{
  const int family = internal_libxs_mpi_family();
  int result = ((NULL != comm && INTERNAL_LIBXS_MPI_NONE != family) ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    int initialized = 0, finalized = 1;
    if (0 != internal_libxs_mpi.initialized(&initialized) || 0 == initialized
      || 0 != internal_libxs_mpi.finalized(&finalized) || 0 != finalized)
    {
      result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result) {
    comm->comm.p = NULL;
    if (INTERNAL_LIBXS_MPI_MPICH == family) { /* the Fortran handle is the handle */
      comm->comm.i = fcomm;
      if (0 != internal_libxs_mpi.rank.i(fcomm, &comm->rank) || 0 != internal_libxs_mpi.size.i(fcomm, &comm->size)) {
        result = EXIT_FAILURE;
      }
    }
    else {
      comm->comm.p = internal_libxs_mpi.f2c(fcomm);
      if (NULL == comm->comm.p || 0 != internal_libxs_mpi.rank.p(comm->comm.p, &comm->rank)
        || 0 != internal_libxs_mpi.size.p(comm->comm.p, &comm->size))
      {
        result = EXIT_FAILURE;
      }
    }
  }
  return result;
}


LIBXS_API_INTERN int internal_libxs_mpi_dup(internal_libxs_mpi_comm_t* comm)
{
  const int family = internal_libxs_mpi.family;
  int result = ((NULL != comm && INTERNAL_LIBXS_MPI_NONE != family) ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    internal_libxs_mpi_handle_t own;
    own.p = NULL;
    if (INTERNAL_LIBXS_MPI_MPICH == family) {
      if (0 != internal_libxs_mpi.dup.i(comm->comm.i, &own.i)) result = EXIT_FAILURE;
    }
    else if (0 != internal_libxs_mpi.dup.p(comm->comm.p, &own.p)) result = EXIT_FAILURE;
    if (EXIT_SUCCESS == result) comm->comm = own;
  }
  return result;
}


LIBXS_API_INTERN void internal_libxs_mpi_free(internal_libxs_mpi_comm_t* comm)
{
  const int family = internal_libxs_mpi.family;
  int finalized = 1;
  if (NULL != comm && 0 < comm->size && INTERNAL_LIBXS_MPI_NONE != family
    && 0 == internal_libxs_mpi.finalized(&finalized) && 0 == finalized)
  {
    if (INTERNAL_LIBXS_MPI_MPICH == family) internal_libxs_mpi.release.i(&comm->comm.i);
    else internal_libxs_mpi.release.p(&comm->comm.p);
    comm->size = 0;
  }
}


LIBXS_API_INTERN int internal_libxs_mpi_serialized(void)
{
  int level = 0;
  return (INTERNAL_LIBXS_MPI_NONE != internal_libxs_mpi.family
    && 0 == internal_libxs_mpi.query_thread(&level) && INTERNAL_LIBXS_MPI_SERIALIZED <= level);
}


LIBXS_API_INTERN int internal_libxs_mpi_main_thread(void)
{
  int flag = 0;
  return (INTERNAL_LIBXS_MPI_NONE != internal_libxs_mpi.family
    && 0 == internal_libxs_mpi.is_main_thread(&flag) && 0 != flag);
}


/* Bytes per message: equal pieces, i.e., no small tail after INT_MAX-sized messages. Both sides of an
 * exchange derive the pieces from the same size, hence agree. */
LIBXS_API_INLINE size_t internal_libxs_mpi_piece(size_t nbytes)
{
  const size_t npieces = LIBXS_UPDIV(nbytes, (size_t)INTERNAL_LIBXS_MPI_PIECE_MAX);
  return (1 >= npieces ? nbytes : LIBXS_UP2(LIBXS_UPDIV(nbytes, npieces), (size_t)LIBXS_CACHELINE));
}


LIBXS_API_INTERN int internal_libxs_mpi_sendrecv(const internal_libxs_mpi_comm_t* comm,
  const void* sbuf, size_t nsend, int dst, void* rbuf, size_t nrecv, int src)
{
  const size_t psend = internal_libxs_mpi_piece(nsend), precv = internal_libxs_mpi_piece(nrecv);
  const int family = internal_libxs_mpi.family;
  size_t soff = 0, roff = 0;
  int result = ((NULL != comm && INTERNAL_LIBXS_MPI_NONE != family
    && (0 == nsend || NULL != sbuf) && (0 == nrecv || NULL != rbuf)) ? EXIT_SUCCESS : EXIT_FAILURE);
  while (EXIT_SUCCESS == result && (soff < nsend || roff < nrecv)) {
    const int ns = (int)LIBXS_MIN(nsend - soff, psend);
    const int nr = (int)LIBXS_MIN(nrecv - roff, precv);
    const char *const s = (const char*)sbuf + soff;
    char *const r = (char*)rbuf + roff;
    internal_libxs_mpi_handle_t sreq, rreq;
    if (INTERNAL_LIBXS_MPI_MPICH == family) {
      const int byte = internal_libxs_mpi.byte.i, c = comm->comm.i;
      if (0 < nr) result |= internal_libxs_mpi.irecv.i(r, nr, byte, src, INTERNAL_LIBXS_MPI_TAG, c, &rreq.i);
      if (0 < ns) result |= internal_libxs_mpi.isend.i(s, ns, byte, dst, INTERNAL_LIBXS_MPI_TAG, c, &sreq.i);
      if (0 < nr) result |= internal_libxs_mpi.wait.i(&rreq.i, internal_libxs_mpi.status_ignore);
      if (0 < ns) result |= internal_libxs_mpi.wait.i(&sreq.i, internal_libxs_mpi.status_ignore);
    }
    else {
      void *const byte = internal_libxs_mpi.byte.p, *const c = comm->comm.p;
      if (0 < nr) result |= internal_libxs_mpi.irecv.p(r, nr, byte, src, INTERNAL_LIBXS_MPI_TAG, c, &rreq.p);
      if (0 < ns) result |= internal_libxs_mpi.isend.p(s, ns, byte, dst, INTERNAL_LIBXS_MPI_TAG, c, &sreq.p);
      if (0 < nr) result |= internal_libxs_mpi.wait.p(&rreq.p, internal_libxs_mpi.status_ignore);
      if (0 < ns) result |= internal_libxs_mpi.wait.p(&sreq.p, internal_libxs_mpi.status_ignore);
    }
    soff += (size_t)ns;
    roff += (size_t)nr;
  }
  return (EXIT_SUCCESS == result ? EXIT_SUCCESS : EXIT_FAILURE);
}


LIBXS_API_INTERN int internal_libxs_mpi_and(const internal_libxs_mpi_comm_t* comm, int* flag)
{
  int result = ((NULL != comm && NULL != flag) ? EXIT_SUCCESS : EXIT_FAILURE);
  if (EXIT_SUCCESS == result) {
    const int next = (comm->rank + 1) % comm->size, prev = (comm->rank + comm->size - 1) % comm->size;
    int value = (0 != *flag), step;
    for (step = 1; step < comm->size && EXIT_SUCCESS == result; ++step) { /* value: AND of step ranks */
      int other = 0;
      result = internal_libxs_mpi_sendrecv(comm, &value, sizeof(int), next, &other, sizeof(int), prev);
      value = (0 != value && 0 != other);
    }
    *flag = value;
  }
  return result;
}
