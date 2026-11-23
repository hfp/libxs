# SYRK / SYR2K Sample

Demonstrates symmetric rank-k and rank-2k updates with LIBXS plans
(dispatch once, call many times). The sample validates correctness
against a plain Fortran reference and reports performance.

## Operations

SYRK:   C := alpha * A * A^T + beta * C  (lower triangle)
SYR2K:  C := alpha * (A * B^T + B * A^T) + beta * C  (upper triangle)
SYRK:   C := alpha * A^T * A + beta * C  (transposed, upper triangle)

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
`make MPI=1` builds with the MPI wrapper of the Fortran compiler
(`MPIFC`, default `mpif90`) and adds the distributed SYRK (see below).

## Run

    ./syrk.x [N [K [nrepeat [direct [tasks [block]]]]]]

Arguments (all optional, positional):

    N        Matrix dimension of C (N x N).  Default: 64
    K        Inner dimension (columns of A).  Default: N
    nrepeat  Number of timed repetitions.     Default: 100
    direct   Interface selection.             Default: 0
               0 = plan (dispatch + call separately)
               1 = direct (one step, arguments as in BLAS)
    tasks    Tasks per thread (OpenMP).       Default: 4
    block    Block size of distributed C (MPI). Default: 32

## OpenMP

When the Fortran module is compiled with OpenMP (the sample's default),
`libxs_syrk` runs over an OpenMP team, unless called from within a
parallel region. The number of tasks comes from `libxs_syrk_ntasks`,
which returns the tiles of the triangle (1 for small shapes, no split),
or 1 with a pthread-based OpenBLAS. The sample prints this count as
`libxs_syrk=`. The result is the same for any number of threads.

The performance section compares `DSYRK` with `LIBXS` (`libxs_syrk`).
`DSYRK` is threaded only with a threaded BLAS, e.g., `make MKL=2`:

    make MKL=2
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close ./syrk.x 2048 2048 5

The `OMP` and `TASK` lines call `libxs_syrk_task` explicitly, as C code
would: one task per thread, or `tasks` per thread over a dynamic DO.

## MPI

Built with `make MPI=1`, the sample also runs the distributed SYRK
(`libxs_syrk_dispatch` with a `libxs_pdist_t`): the ranks split K into
stripes of columns, and C (lower triangle) is block-cyclic on a near-square
grid of the ranks with blocks of `block` x `block`. Every rank computes the
reference itself, and rank 0 prints the largest error of any rank and the
time of the distributed update (`DIST`, GFLOPS/s over all ranks), run by
the thread team as tasks. The other sections run on every rank as well, and
rank 0 prints them.

    make MPI=1
    mpiexec -n 4 ./syrk.x 2048 8192 5 0 4 64

## Example Output

    SYRK: N=64 K=64 nrepeat=100 direct=0

    --- SYRK (lower) ---
      max error (lower):  0.00000E+00

    --- SYR2K (upper) ---
      max error (upper):  0.00000E+00

    --- SYRK (transposed, upper) ---
      max error (transposed):  0.00000E+00

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

Here `libxs_syrk=1`, i.e., the update is a single tile (see above).

## Notes

- A plan is a plain value: a local plan holds no memory and needs no
  release, whereas a distributed plan is released collectively
  (`libxs_syrk_release`).
- `LIBXS_SYRK_BLAS=0` computes the blocks on the diagonal as GEMM
  squares instead of BLAS SYRK/SYR2K; see `documentation/libxs_syrk.md`.
