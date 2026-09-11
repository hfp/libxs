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

`libxs_syrk_task` and `libxs_syr2k_task` take a task index and a task
count instead of running the whole update. The sample shows both ways
to supply them:

    OMP      one task per thread (tid = omp_get_thread_num)
    TASK     tasks * threads tasks over an OpenMP DO (dynamic)

Oversubscribing is worthwhile because the tasks cover a triangle: tasks
are handed out as contiguous ranges of C blocks, and the blocks skipped
outside the triangle are not spread evenly over those ranges. A dynamic
schedule with more tasks than threads evens this out.

Both forms stay correct without OpenMP: the first runs as one task, the
second as a serial loop over all tasks. No build-time branch is needed.

The parallel split only engages where SYRK is decomposed into blocks.
With `N` or `K` beyond the block size (`LIBXS_GEMM_BM`, `LIBXS_GEMM_BN`,
`LIBXS_GEMM_BK`), a BLAS `dsyrk` found at runtime is preferred over the
decomposition, and it runs on the first task alone. Set
`LIBXS_SYRK_BLAS=0` to keep the blocked path, which is what the OpenMP
variants parallelize:

    LIBXS_SYRK_BLAS=0 OMP_NUM_THREADS=8 ./syrk.x 2000 2000 20

## Example Output

    SYRK: N=64 K=64 nrepeat=100 direct=0

    --- libxs_syrk (lower) ---
      max error (lower): 0.00000E+00

    --- libxs_syr2k (upper) ---
      max error (upper): 0.00000E+00

    --- libxs_syrk_task (OpenMP) ---
      threads=8 ntasks=32
      max error (omp):  0.00000E+00
      max error (omp tasks):  0.00000E+00

    --- SYRK performance ---
      BLAS:      0.002 s (100 calls)
                     28.4 GFLOPS/s
      LIBXS:     0.002 s (100 calls)
                     28.1 GFLOPS/s
      OMP:       0.002 s (100 calls)
                     28.0 GFLOPS/s
      TASK:      0.002 s (100 calls)
                     27.9 GFLOPS/s

At the default N=64 the update is a handful of blocks and the BLAS
path is preferred, so OMP and TASK match LIBXS. Both only pull ahead
at a size worth splitting, and with `LIBXS_SYRK_BLAS=0`.

## Notes

- The dispatch step (libxs_syrk_dispatch / libxs_syr2k_dispatch)
  returns a pointer to a registry-owned config. This pointer
  remains valid until libxs_finalize or the registry is destroyed.
  There is no need to release it manually.

- Internally, SYRK/SYR2K decompose into GEMM tiles on the
  diagonal and off-diagonal blocks. The dispatched GEMM kernel
  (MKL JIT, LIBXSMM, or fallback BLAS) handles the inner loop.

- Scratch memory for the temporary block products is a thread-local
  buffer that grows on demand, hence tasks need no synchronization:
  each task holds its own scratch and writes its own blocks of C. It
  is drawn from the LIBXS memory pool, so it appears in the pool
  statistics rather than in an untracked allocation.
