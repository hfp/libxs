/******************************************************************************
* Copyright (c) 2009-2026 Hans Pabst                                          *
* Copyright (c) 2009-2026 Intel Corporation                                   *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_SYRK_H
#define LIBXS_SYRK_H

#include "libxs_gemm.h"


/**
 * Block-cyclic layout of C: the communicator as Fortran handle (MPI_Comm_c2f),
 * the block and grid values of a ScaLAPACK descriptor, and the rank of every
 * grid position (column-major, e.g., BLACS' mapping). Ranks may be left out of
 * the grid, and then own no part of C. The dispatch copies the ranks.
 */
typedef struct libxs_pdist_t {
  int fcomm;
  int mb, nb, nprow, npcol;
  int rsrc, csrc;
  const int* ranks;
} libxs_pdist_t;

/** Plan of SYRK or SYR2K (see below). */
typedef struct libxs_syrk_t libxs_syrk_t;

/**
 * Part of the triangle to compute: columns c0 to c0+w-1 and rows r0 to r0+m-1,
 * which includes the block on the diagonal (rows c0 to c0+w-1). The result is
 * out := alpha*result + beta*out (column-major, host memory): C itself if the
 * plan is local, or a buffer of partial sums (alpha=1, beta=0) if distributed.
 * Like BLAS, beta=0 does not read out, and only the triangle of the diagonal
 * block is written.
 */
typedef struct libxs_syrk_panel_t {
  int c0, w, r0, m;
  void* out;
  int ldo;
  double alpha, beta;
} libxs_syrk_panel_t;

/**
 * Compute of panels by someone else than the host's BLAS, e.g., a device.
 * submit: starts a panel, and returns a ticket (>= 0) or declines (< 0), and
 * then the host computes the panel. wait: completes the panel of a ticket and
 * returns EXIT_SUCCESS or EXIT_FAILURE. The operands a and b are passed as the
 * caller gave them (what ctx makes of them, e.g., resident copies, is up to
 * ctx). Calls are never concurrent, and panels are submitted in order.
 */
typedef struct libxs_syrk_device_t {
  int (*submit)(void* ctx, const libxs_syrk_t* plan,
    const void* a, const void* b, const libxs_syrk_panel_t* panel);
  int (*wait)(void* ctx, int ticket);
  void* ctx;
} libxs_syrk_device_t;

/**
 * Shape of a symmetric update, which writes one triangle of C (n x n):
 * SYRK (ldb=0): C := alpha*op(A)*op(A)^T + beta*C, or
 * SYR2K:        C := alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C.
 * trans 'N': op(A)=A is n x k, 'T': op(A)=A^T, i.e., A is k x n.
 */
typedef struct libxs_syrk_shape_t {
  libxs_data_t datatype;
  char uplo, trans;
  /** If distributed, k is the share of this rank (which holds all of n). */
  int n, k;
  /** If distributed, ldc belongs to this rank's part of C. */
  int lda, ldb, ldc;
  /** Layout of C if distributed, NULL if local (read at dispatch). */
  const libxs_pdist_t* dist;
  /** Compute of panels, NULL for the host (copied, ctx must outlive the plan). */
  const libxs_syrk_device_t* device;
} libxs_syrk_shape_t;

/**
 * Plan of SYRK or SYR2K, local or distributed. A local plan holds no memory,
 * needs no release, and may serve any number of calls at a time (unless it
 * computes on a device). A distributed plan owns buffers and a communicator,
 * serves one call at a time, and is released collectively (libxs_syrk_release).
 */
struct libxs_syrk_t {
  /** Shape as dispatched, without its pointers (see device and dist). */
  libxs_syrk_shape_t shape;
  /** Kernel of a small shape (valid as long as its registry), or BLAS. */
  libxs_gemm_config_t gemm;
  /** Compute of panels if submit is not NULL. */
  libxs_syrk_device_t device;
  /** Columns and rows of a tile, and tiles in total (a function of n only). */
  int width, height, ntiles;
  /** State of a distributed plan, NULL otherwise. */
  void* dist;
};


/**
 * Plan a SYRK or SYR2K, F64 or F32 (runtime, explicit backend selection).
 * If shape->dist is given, the dispatch is collective over its communicator,
 * whose ranks agree on the outcome: either all of them have a plan, or none has
 * and the caller falls back (no or unknown MPI, shapes or layouts that differ
 * between ranks, or a rank with invalid arguments). The device may differ
 * between ranks. MPI is found at runtime, and LIBXS does not link it.
 * registry: registry of kernels, NULL for the default.
 * Returns non-zero if plan was populated, zero otherwise.
 */
LIBXS_API int libxs_syrk_dispatch_rt(libxs_syrk_t* plan, const libxs_syrk_shape_t* shape,
  const libxs_gemm_backend_t* backend, void* registry);

/**
 * Plan a SYRK or SYR2K (see libxs_syrk_dispatch_rt), with the backend selected
 * at the caller's compile time (see libxs_gemm_backend_init).
 */
LIBXS_API_INLINE int libxs_syrk_dispatch(libxs_syrk_t* plan, const libxs_syrk_shape_t* shape)
{
  libxs_gemm_backend_t be;
  libxs_gemm_backend_init(&be);
  return libxs_syrk_dispatch_rt(plan, shape, &be, NULL);
}

/**
 * Run the planned update in the calling thread, which is collective for a
 * distributed plan. b is NULL for SYRK. Like BLAS, beta=0 does not read C, and
 * k=0 or alpha=0 does not read A or B (which may then be NULL). C may be NULL
 * on a rank without part of it. Returns EXIT_SUCCESS or EXIT_FAILURE.
 */
LIBXS_API int libxs_syrk(const libxs_syrk_t* plan,
  double alpha, const void* a, const void* b, double beta, void* c);

/**
 * Per-task form of libxs_syrk. Every tid in [0, ntasks) calls once per update,
 * from any thread and in any order, and there may be more or fewer tasks than
 * threads. If distributed and MPI provides less than MPI_THREAD_SERIALIZED,
 * the main thread runs at least one of the tasks. C is complete when all the
 * tasks returned. The result does not depend on the threads, the tasks, or
 * the schedule, given the ranks and a BLAS which is sequential within a task
 * (MKL is, unless MKL_DYNAMIC=FALSE). With a device, one of the tasks drives
 * it and computes what it declines, and the other tasks return.
 */
LIBXS_API int libxs_syrk_task(const libxs_syrk_t* plan,
  double alpha, const void* a, const void* b, double beta, void* c,
  int tid, int ntasks);

/**
 * Number of tasks worth running with nthreads threads, and 1 if a split does
 * not pay. A local plan returns its tiles, which can exceed the threads: hand
 * them out dynamically (omp parallel for schedule(dynamic)) for a team which
 * balances itself. A distributed plan balances itself and returns nthreads,
 * but at most a task per tile and one more. A plan with a device returns 1.
 */
LIBXS_API int libxs_syrk_ntasks(const libxs_syrk_t* plan, int nthreads);

/**
 * Release a plan, collective if distributed. Safe to call with NULL, for a
 * local plan, for a failed dispatch, and repeatedly.
 */
LIBXS_API void libxs_syrk_release(libxs_syrk_t* plan);

/* header-only: include implementation (deferred from libxs_macros.h) */
#if defined(LIBXS_SOURCE) && !defined(LIBXS_SOURCE_H) \
 && !defined(LIBXS_PREDICT_H)
# include "libxs_source.h"
#endif

#endif /*LIBXS_SYRK_H*/
