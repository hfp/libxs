# TAS: Tall-and-Skinny GEMM

One column-major GEMM, C := α·op(A)·op(B) + β·C (`libxs_tas.h`), whose
arithmetic intensity is low enough for memory bandwidth to bound it:
one dimension (M, N, or K) is much larger than the other two. Such a
GEMM runs along its longest dimension as a constant-stride
[GEMM batch](libxs_gemm.md#indexstrided-batch) of panels plus one
remainder:

| Longest | Panels                                   | Shared operand |
| :------ | :--------------------------------------- | :------------- |
| M       | rows of op(A) and C                      | B              |
| N       | columns of op(B) and C                   | A              |
| K       | chunks of the inner dimension            | none           |

Along K, each chunk writes a partial C of its own, and the partial
results are summed in a fixed order: the result is the same for any
number of tasks. Any other GEMM is planned as one call of the whole
GEMM. F64 and F32 are supported, with either operand transposed
('N' or 'T').

Panels are dispatched with `LIBXS_GEMM_FLAG_STREAM` and
`LIBXS_GEMM_FLAG_REUSE` (see [GEMM](libxs_gemm.md#types)): a panel is
a window into a larger matrix, which a generated kernel may serve, and
it is called often, which skips the JIT warm-up.

## Plan

```C
typedef struct libxs_tas_t {
  libxs_gemm_shape_t shape;            /* the whole GEMM */
  libxs_gemm_config_t panel, rest;     /* full panel, remainder */
  int stride_a, stride_b, stride_c;    /* between panels (elements) */
  size_t rest_a, rest_b, rest_c;       /* offsets of the remainder */
  int npanels, nscratch;               /* panels, partial C (along K) */
  char dim;                            /* 'M', 'N', 'K', or zero */
  void* scratch;                       /* partial C (along K) */
} libxs_tas_t;
```

A plan is caller-owned. It holds the scratch of the partial results
(along K), hence it serves one call at a time, and it is released by
`libxs_tas_release`. The configs inside are ordinary GEMM configs.

## Dispatch

```C
int libxs_tas_dispatch(libxs_tas_t* plan,
  libxs_data_t datatype, char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc,
  const void* alpha, const void* beta, int panel,
  void* registry /* = NULL */);

int libxs_tas_dispatch_rt(libxs_tas_t* plan,
  const libxs_gemm_shape_t* shape, int panel,
  const libxs_gemm_backend_t* backend, void* registry);
```

Plan a GEMM, and return nonzero if the plan was populated. The inline
form detects the backend at the caller's compile time, exactly like
[libxs_gemm_dispatch](libxs_gemm.md#dispatch); the runtime form takes
the backend explicitly. The panel is the extent of a panel along M or
N, or of a chunk along K; zero takes `LIBXS_TAS_PANEL` or
`LIBXS_TAS_CHUNK`. A GEMM is planned along its longest dimension if its
arithmetic intensity (FLOPs per Byte) is below `LIBXS_TAS_MAX` and if
this yields at least two panels; otherwise `dim` is zero and the plan
is one call.

## Call

```C
void libxs_tas(const libxs_tas_t* plan,
  const void* a, const void* b, void* c);

void libxs_tas_task(const libxs_tas_t* plan,
  const void* a, const void* b, void* c,
  int tid, int ntasks);

int libxs_tas_ntasks(const libxs_tas_t* plan, int nthreads);

void libxs_tas_release(libxs_tas_t* plan);
```

The operands are those of the whole GEMM. `libxs_tas_task` takes the
share of one task: every tid in [0, ntasks) must call, since along K
the last task to finish sums the partial results into C.
`libxs_tas_ntasks` tells how many tasks are worth it (one for a plan of
one call), see [libxs_gemm_ntasks](libxs_gemm.md#indexstrided-batch).

## Compile-Time Tuning

    LIBXS_TAS_PANEL      Extent of a panel along M or N (default: 256)
    LIBXS_TAS_CHUNK      Extent of a chunk along K (default: 4096)
    LIBXS_TAS_MAX        Arithmetic intensity (FLOPs per Byte) below
                         which a GEMM is planned along its longest
                         dimension (default: 7, as LIBXS_GEMM_JIT_MAX)

## Environment Variables

    LIBXS_TAS_PANEL=N    Extent of a panel along M or N; zero or unset
                         takes the compile-time default.
    LIBXS_TAS_CHUNK=N    Extent of a chunk along K; zero or unset takes
                         the compile-time default.
    LIBXS_TAS_MAX=X      Arithmetic intensity below which a GEMM is
                         planned along its longest dimension; zero
                         plans every GEMM as one call.

The GEMM variables (`LIBXS_GEMM_BACKEND`, `LIBXS_GEMM_JIT_MAX`, etc.)
apply to the panels, see [GEMM](libxs_gemm.md#environment-variables).

## Example

```C
#include <libxs/libxs_tas.h>

libxs_tas_t plan;
if (0 != libxs_tas_dispatch(&plan, LIBXS_DATATYPE_F64, 'N', 'N',
  m, n, k, lda, ldb, ldc, &alpha, &beta, 0/*panel*/, NULL))
{
  const int ntasks = libxs_tas_ntasks(&plan, omp_get_max_threads());
# pragma omp parallel num_threads(ntasks)
  libxs_tas_task(&plan, a, b, c, omp_get_thread_num(), omp_get_num_threads());
  libxs_tas_release(&plan);
}
```
