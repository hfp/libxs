# SYRK / SYR2K Sample

Demonstrates symmetric rank-k and rank-2k updates using the
LIBXS dispatch-and-call model. The sample validates correctness
against a plain Fortran reference and reports performance.

## Operations

SYRK:   C := alpha * A * A^T + beta * C  (lower triangle)
SYR2K:  C := alpha * (A * B^T + B * A^T) + beta * C  (upper triangle)

Only the specified triangle of C is written; the other triangle
is left untouched.

## Build

    make

Requires a Fortran compiler (gfortran, ifort, or ifx) and LIBXS
built first from the repository root (the sample is not built
header-only). The Makefile picks up the compiler from the top-level
Makefile.inc. BLAS is linked dynamically and sequential by default
(`BLAS=1`, `BLAS_STATIC=0`). JIT-compiled kernels
need `make MKL=1` (sequential) or `make MKL=2` (threaded) for MKL's
JIT, or `make LIBXSMM=1` with a built sibling `libxsmm` directory.

## Run

    ./syrk.x [N [K [nrepeat [direct [tasks]]]]]

Arguments (all optional, positional):

    N        Matrix dimension of C (N x N).  Default: 64
    K        Inner dimension (columns of A).  Default: N
    nrepeat  Number of timed repetitions.     Default: 100
    direct   Interface selection.             Default: 0
               0 = config (dispatch + call separately)
               1 = direct (one-shot generic)
    tasks    Tasks per thread (OpenMP).       Default: 4

## OpenMP

When the Fortran module is compiled with OpenMP (the sample's default),
`libxs_syrk` and `libxs_syr2k` run over an OpenMP team, unless called
from within a parallel region. The team size comes from
`libxs_syrk_ntasks`, which returns 1 (no split) for small shapes, with
a threaded BLAS, or with too few threads against a sequential BLAS. The
sample prints this count as `libxs_syrk=`.

The performance section compares `DSYRK` with `LIBXS` (`libxs_syrk`).
`DSYRK` is threaded only with a threaded BLAS, e.g., `make MKL=2`:

    make MKL=2
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close ./syrk.x 2048 2048 5

The `OMP` and `TASK` lines call `libxs_syrk_task` explicitly, as C code
would: one task per thread, or `tasks` per thread over a dynamic DO.

## Example Output

    SYRK: N=64 K=64 nrepeat=100 direct=0

    --- libxs_syrk (lower) ---
      max error (lower): 0.00000E+00

    --- libxs_syr2k (upper) ---
      max error (upper): 0.00000E+00

    --- libxs_syrk_task (OpenMP) ---
      threads=8 ntasks=32 libxs_syrk=1
      max error (omp):  0.00000E+00
      max error (omp tasks):  0.00000E+00

    --- SYRK performance ---
      DSYRK:     <time> s (100 calls)
                     <rate> GFLOPS/s
      LIBXS:     <time> s (100 calls)
                     <rate> GFLOPS/s
      OMP:       <time> s (100 calls)
                     <rate> GFLOPS/s
      TASK:      <time> s (100 calls)
                     <rate> GFLOPS/s

Here `libxs_syrk=1`, i.e., the update is not split (see above).

## Notes

- The dispatch step (libxs_syrk_dispatch / libxs_syr2k_dispatch)
  returns a pointer to a registry-owned config. This pointer
  remains valid until libxs_finalize or the registry is destroyed.
  There is no need to release it manually.

- Splitting is controlled by `LIBXS_GEMM_PANEL` (smallest column
  panel), `LIBXS_SYRK_BLAS` (BLAS SYRK for diagonal blocks, default
  on), and the tile sizes `LIBXS_GEMM_BM`, `LIBXS_GEMM_BN`,
  `LIBXS_GEMM_BK` (default 192x32x48); see
  `documentation/libxs_syrk.md`.

- Tasks need no synchronization. Scratch memory is thread-local and
  drawn from the LIBXS memory pool, so it appears in the pool
  statistics.
