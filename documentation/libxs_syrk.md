# SYRK: Symmetric Rank-k Updates

Symmetric rank-k and rank-2k updates (`libxs_syrk.h`) of one triangle of C (n x n), in double or single precision:

- SYRK: C := alpha\*op(A)\*op(A)^T + beta\*C
- SYR2K: C := alpha\*(op(A)\*op(B)^T + op(B)\*op(A)^T) + beta\*C

One plan type serves both, on one rank or distributed over the ranks of an MPI communicator. Backends and registry are those of [GEMM](libxs_gemm.md).

## Plan

```C
typedef struct libxs_syrk_shape_t {
  libxs_data_t datatype;  /* LIBXS_DATATYPE_F64 or LIBXS_DATATYPE_F32 */
  char uplo, trans;       /* 'U' or 'L'; 'N': A is n x k, 'T': A is k x n */
  int n, k;               /* k: this rank's share if distributed */
  int lda, ldb, ldc;      /* ldb=0: SYRK, else SYR2K; ldc: local part of C */
  const libxs_pdist_t* dist;          /* NULL: local (see Distributed) */
  const libxs_syrk_device_t* device;  /* NULL: host (see Device) */
} libxs_syrk_shape_t;

int libxs_syrk_dispatch(libxs_syrk_t* plan, const libxs_syrk_shape_t* shape);

int libxs_syrk_dispatch_rt(libxs_syrk_t* plan, const libxs_syrk_shape_t* shape,
  const libxs_gemm_backend_t* backend, void* registry);
```

A dispatch returns nonzero if the plan was populated. `libxs_syrk_dispatch` is an inline that detects the backend at the caller's compile time like [libxs_gemm_dispatch](libxs_gemm.md#dispatch), and `libxs_syrk_dispatch_rt` takes explicit backends and a registry (NULL for the default).

- **Tiles:** the plan splits the triangle into tiles, which depend on n alone. A small shape (one tile, n <= 32, k <= 48, n\*k <= 768) runs a generated kernel if a backend provides one. It occupies the registry entry of the GEMM of its full square (n x n x k), which a GEMM of that shape shares. Larger shapes run BLAS and take no registry entry.
- **Ownership:** a local plan holds no memory, needs no release, and can serve any number of calls at a time (unless it computes on a device). A distributed plan owns a communicator and buffers, serves one call at a time, and is released collectively.
- **Lifetimes:** the shape is read at dispatch and can be reused or discarded right away. The plan copies the layout (`dist`) and the device's functions, and keeps no pointer of the shape. Only the device's context must outlive the plan.

## Call

```C
int libxs_syrk(const libxs_syrk_t* plan,
  double alpha, const void* a, const void* b, double beta, void* c);

int libxs_syrk_task(const libxs_syrk_t* plan,
  double alpha, const void* a, const void* b, double beta, void* c,
  int tid, int ntasks);

int libxs_syrk_ntasks(const libxs_syrk_t* plan, int nthreads);

void libxs_syrk_release(libxs_syrk_t* plan);
```

- **Arguments:** b is NULL for SYRK. Like BLAS, beta=0 does not read C, and k=0 or alpha=0 does not read A or B (which may then be NULL). The functions return `EXIT_SUCCESS` or `EXIT_FAILURE`.
- **Tasks:** every tid in [0, ntasks) calls `libxs_syrk_task` once per update, from any thread and in any order, and C is complete when all of them have returned. `libxs_syrk_ntasks` returns the number of tasks worth running with nthreads threads: the tiles of a local plan (hand them out dynamically, the team balances itself), nthreads for a distributed plan, and 1 if a split does not pay (e.g., a pthread-based OpenBLAS).
- **Reproducibility:** the result does not depend on the threads, the tasks, or the schedule, given the ranks and a BLAS that is sequential within a task. MKL is sequential inside a parallel region by default; with `MKL_DYNAMIC=FALSE`, its threads split the work of a task, and the bits change with the number of threads.
- **Release:** `libxs_syrk_release` accepts NULL, a local plan, a failed dispatch, and a plan released before.

```C
libxs_syrk_shape_t shape = { LIBXS_DATATYPE_F64, 'L', 'N' };  /* the rest zero */
libxs_syrk_t plan;
shape.n = n; shape.k = k; shape.lda = n; shape.ldc = n;
if (0 != libxs_syrk_dispatch(&plan, &shape)) {
  const int ntasks = libxs_syrk_ntasks(&plan, omp_get_max_threads());
  int t;
#pragma omp parallel for schedule(dynamic)
  for (t = 0; t < ntasks; ++t) libxs_syrk_task(&plan, 1.0, a, NULL, 0.0, c, t, ntasks);
}
```

## Distributed

```C
typedef struct libxs_pdist_t {
  int fcomm;               /* communicator as Fortran handle (MPI_Comm_c2f) */
  int mb, nb, nprow, npcol;
  int rsrc, csrc;
  const int* ranks;        /* rank of every grid position (column-major) */
} libxs_pdist_t;
```

- **Operands:** every rank holds all n rows (or columns) of A and B with its share of k. C is block-cyclic as described by the values of a ScaLAPACK descriptor and the rank of every grid position, e.g., BLACS' mapping. Ranks outside the grid own no part of C and pass NULL. The dispatch copies the ranks.
- **Collective dispatch:** all ranks agree on the outcome. Either every rank has a plan, or none has, and the caller falls back, e.g., to ScaLAPACK. That is the case without a known MPI, for shapes or layouts that differ between ranks, or if one rank has invalid arguments.
- **MPI:** LIBXS does not link MPI. It finds MPI in the process (see `LIBXS_MPI`) and duplicates the communicator.
- **Algorithm:** the partial sums of each panel of columns are reduced in a ring of fixed order and delivered to the owners of C. The buffers of two panels (n x 256 elements at most) are allocated at dispatch.
- **Threads:** with `MPI_THREAD_SERIALIZED` or higher, any thread communicates. Below that level, the main thread must run one of the tasks.

```C
libxs_pdist_t dist = { 0 };
dist.fcomm = MPI_Comm_c2f(comm);
dist.mb = mb; dist.nb = nb; dist.nprow = nprow; dist.npcol = npcol;
dist.ranks = ranks;
shape.k = klocal; shape.ldc = ldc_local; shape.dist = &dist;
if (0 != libxs_syrk_dispatch(&plan, &shape)) {
  libxs_syrk(&plan, 1.0, a, NULL, 0.0, c);  /* collective */
  libxs_syrk_release(&plan);                /* collective */
}
else { /* fall back */ }
```

## Device

```C
typedef struct libxs_syrk_panel_t {
  int c0, w, r0, m;   /* columns c0..c0+w-1, rows r0..r0+m-1 (incl. the diagonal block) */
  void* out; int ldo; /* host memory: C (local plan), or partial sums (distributed) */
  double alpha, beta; /* out := alpha*result + beta*out; distributed: 1 and 0 */
} libxs_syrk_panel_t;

typedef struct libxs_syrk_device_t {
  int (*submit)(void* ctx, const libxs_syrk_t* plan,
    const void* a, const void* b, const libxs_syrk_panel_t* panel);
  int (*wait)(void* ctx, int ticket);
  void* ctx;
} libxs_syrk_device_t;
```

A device computes the plan's panels in place of the host's BLAS, while LIBXS keeps the algorithm, including the distributed reduction. The device is an implementation of these two functions, e.g., in [LIBXSTREAM](https://github.com/hfp/libxstream).

- **submit:** starts a panel and returns a ticket (>= 0). It may decline (< 0), and then the host computes the panel. Like BLAS, beta=0 does not read `out`, and only the triangle of the diagonal block is written.
- **wait:** completes the panel of a ticket, and returns `EXIT_SUCCESS` or `EXIT_FAILURE`.
- **Calls:** never concurrent, and panels are submitted in order. A local plan keeps two panels in flight. A distributed plan submits the next panel before it reduces the current one, so the device computes while the host communicates.
- **Operands:** `a` and `b` arrive as the caller passed them. What the context makes of them, e.g., copies resident on a device, or a scaling of the columns between calls, is up to the context.
- **Tasks:** one task drives the device, and `libxs_syrk_ntasks` returns 1. The device computes every panel it accepts, hence the result does not depend on a schedule. With k=0 or alpha=0, the host scales C without the device.
- **Distributed:** the device may differ between ranks, or be absent on some of them.

## Fortran

```fortran
TYPE(libxs_syrk_t) :: plan
rc = libxs_syrk_dispatch(plan, LIBXS_DATATYPE_F64, uplo, trans,
     &  n, k, lda, ldb, ldc [, dist] [, device] [, backends]
     &  [, registry])
CALL libxs_syrk(plan, alpha, a, beta, c [, info])
CALL libxs_syrk(plan, alpha, a, b, beta, c [, info])
CALL libxs_syrk_task(plan, alpha, a, [b,] beta, c, tid, ntasks
     &  [, info])
CALL libxs_syrk_release(plan)
```

- **Arguments:** a, b, and c are `C_LOC` pointers. ldb=0 selects SYRK, as in C. The backends (module LIBXS) are OPTIONAL function pointers, and module LIBXS_JIT fills them in.
- **OpenMP:** compiled with OpenMP, `libxs_syrk` runs over a team sized by `libxs_syrk_ntasks`, unless it is called from within a parallel region.
- **Errors:** without `info`, an error goes to stderr.
- **Distributed:** `dist` is a `TYPE(libxs_pdist_t)` with the Fortran handle of the communicator as is, and `ranks = C_LOC` of an `INTEGER(C_INT)` array.
- **Device:** `device` is a `TYPE(libxs_syrk_device_t)` as an implementation provides it (`C_FUNPTR` and `C_PTR` members).

The direct form dispatches and calls in one step, with the arguments in the order of BLAS:

```fortran
CALL libxs_syrk(datatype, uplo, trans, n, k, alpha, a, lda,
     &  beta, c, ldc [, info])
CALL libxs_syrk(datatype, uplo, trans, n, k, alpha, a, lda,
     &  b, ldb, beta, c, ldc [, info])
```

Without `-D__LIBXS`, `libxs_jit.F` compiles to a BLAS-only module, whose direct form calls DSYRK or DSYR2K.

The [SYRK sample](samples/libxs_syrk.md) covers plans, tasks, and the distributed update (`make MPI=1`).

## Compile-Time Tuning

    LIBXS_SYRK_WIDTH      Width of the widest tiles (default: 256)
    LIBXS_SYRK_KERNEL_N   Largest n served by a kernel (default: 32)
    LIBXS_SYRK_KERNEL_K   Largest k served by a kernel (default: 48)
    LIBXS_SYRK_KERNEL_NK  Largest n*k served by a kernel (default: 768)

## Environment Variables

    LIBXS_SYRK_BLAS=0|1   Use the BLAS SYRK/SYR2K for blocks on the
                          diagonal (default: 1). With 0, or without
                          these entry points, the blocks are computed
                          as GEMM squares and scattered.
    LIBXS_MPI=0|mpich|ompi  MPI of the distributed update (see README).

The GEMM variables (LIBXS_GEMM_BACKEND, LIBXS_GEMM_JIT_MAX, etc.) apply as well, see [GEMM](libxs_gemm.md#environment-variables).
