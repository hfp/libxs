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

Requires a Fortran compiler (gfortran, ifort, or ifx). The
Makefile picks up the compiler from the top-level Makefile.inc.
MKL or BLAS linkage is enabled (BLAS=1) so that dispatch can
use JIT-compiled kernels when available.

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
`libxs_syrk_ntasks`, which returns 1 where the unsplit call is faster:
small shapes, a threaded BLAS (which then parallelizes DSYRK itself),
or too few threads against a sequential BLAS. The sample prints this
count as `libxs_syrk=`.

The performance section compares `DSYRK` with `LIBXS` (`libxs_syrk`).
`DSYRK` is only threaded with a threaded BLAS, hence build with
`make MKL=2` to compare parallel DSYRK with the parallel `libxs_syrk`:

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
      DSYRK:     0.002 s (100 calls)
                     28.4 GFLOPS/s
      LIBXS:     0.002 s (100 calls)
                     28.1 GFLOPS/s
      OMP:       0.002 s (100 calls)
                     28.0 GFLOPS/s
      TASK:      0.002 s (100 calls)
                     27.9 GFLOPS/s

At N=64 the update fits a single tile (`libxs_syrk=1`), hence all lines
match; a split pays off at larger N.

## Notes

- The dispatch step (libxs_syrk_dispatch / libxs_syr2k_dispatch)
  returns a pointer to a registry-owned config. This pointer
  remains valid until libxs_finalize or the registry is destroyed.
  There is no need to release it manually.

- Internally, SYRK/SYR2K decompose into GEMM tiles (default
  192x32x48, see `LIBXS_GEMM_BM`, `LIBXS_GEMM_BN`, `LIBXS_GEMM_BK`)
  on the diagonal and off-diagonal blocks.

- Scratch memory for the temporary block products is a thread-local
  buffer that grows on demand, hence tasks need no synchronization:
  each task holds its own scratch and writes its own blocks of C. It
  is drawn from the LIBXS memory pool, so it appears in the pool
  statistics rather than in an untracked allocation.
