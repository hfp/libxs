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

    ./syrkf.x [N [K [nrepeat]]]

Arguments (all optional, positional):

    N        Matrix dimension of C (N x N).  Default: 64
    K        Inner dimension (columns of A).  Default: N
    nrepeat  Number of timed repetitions.     Default: 100

## Example Output

    syrk(F): N=64 K=64 nrepeat=100

    --- libxs_syrk (lower) ---
      max error (lower): 0.00000E+00

    --- libxs_syr2k (upper) ---
      max error (upper): 0.00000E+00

    --- SYRK performance ---
      time:      0.002 s (100 calls)
      perf:      28.4 GFLOPS/s

## Notes

- The dispatch step (libxs_syrk_dispatch / libxs_syr2k_dispatch)
  returns a pointer to a registry-owned config. This pointer
  remains valid until libxs_finalize or the registry is destroyed.
  There is no need to release it manually.

- Internally, SYRK/SYR2K decompose into GEMM tiles on the
  diagonal and off-diagonal blocks. The dispatched GEMM kernel
  (MKL JIT, LIBXSMM, or fallback BLAS) handles the inner loop.

- Scratch memory for the temporary full-panel product is managed
  via a thread-local buffer that grows on demand and is freed at
  finalization.
