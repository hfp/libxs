# Batched GEMM

Exercises the LIBXS batched GEMM API (`libxs_gemm.h`) in several
flavors: strided, pointer-array, indexed, and grouped. All programs
are multithreaded via OpenMP, report wall-clock time and GFLOPS/s,
and optionally print a checksum of the results via `libxs_matdiff`.

## Building

```bash
cd samples/gemm
make
```

The C programs build header-only when LIBXS is not built yet; only
`gemm_indexf.x` (Fortran) needs LIBXS built first from the repository
root. BLAS is linked dynamically and sequential by default (`BLAS=1`,
`BLAS_STATIC=0`), except that OpenBLAS is taken in its OpenMP (or pthread)
build where available, since the serial build may be unsafe to call from
several threads. `make MKL=1` (sequential) or `make MKL=2` (threaded) links MKL and
enables MKL's JIT kernels; `make LIBXSMM=1` enables LIBXSMM kernels and
needs a built sibling `libxsmm` directory.

## Programs

### gemm_strided

Strided batch DGEMM: all matrices packed contiguously with constant
stride. Uses `libxs_gemm_index` with index_stride=0.

```bash
./gemm_strided.x [M [N [K [batchsize [nrepeat [beta [pad]]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, beta=1.0, pad=0.

### gemm_batch

Pointer-array batch DGEMM: matrices accessed through pointer arrays.
Supports a duplicate C-matrix mode to exercise lock-forward sync.

```bash
./gemm_batch.x [M [N [K [batchsize [nrepeat [dup [beta [pad]]]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, dup=0, beta=1.0,
pad=0.

| dup | Mode     | Description                                      |
|-----|----------|--------------------------------------------------|
| 0   | None     | Each C-matrix is unique (LIBXS_GEMM_FLAG_NOLOCK) |
| 1   | Sorted   | Half-unique C-pointers, duplicates consecutive   |
| 2   | Shuffled | Half-unique C-pointers, randomly shuffled        |

### gemm_index

Index-array batch DGEMM: matrices in contiguous buffers, addressed
through explicit element-offset index arrays (ia, ib, ic).

```bash
./gemm_index.x [M [N [K [batchsize [nrepeat [beta [pad]]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, beta=1.0, pad=0.

### gemm_indexf

Fortran variant of gemm_index. Demonstrates the LIBXS Fortran module
interface with one-based index arrays and C_LOC/C_SIZEOF interop.
Requires a Fortran compiler and a built LIBXS (skipped otherwise).

Compiled with OpenMP, `libxs_gemm_index` (and `libxs_gemm_batch`) run
over an OpenMP team unless called from within a parallel region, sized by
`libxs_gemm_ntasks` (printed as `libxs_gemm_index tasks`). `Performance`
times this plain call, `Task split` the explicit `libxs_gemm_index_task`.

```bash
./gemm_indexf.x [M [N [K [batchsize [nrepeat]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3.

### gemm_groups

Grouped batch DGEMM: multiple groups of different matrix shapes
dispatched in sequence, each with its own `libxs_gemm_config_t`.
Per-group dimensions grow by 4 starting from base_m.

```bash
./gemm_groups.x [ngroups [batch_per_group [nrepeat [base_m [beta [pad]]]]]]
```

Defaults: ngroups=2 (max 4), batch_per_group=30000, nrepeat=3,
base_m=8, beta=1.0, pad=0.

### gemm_tas

Tall-and-skinny DGEMM: one column-major GEMM whose longest dimension
is cut into panels, run as a constant-stride `libxs_gemm_index` batch
plus one remainder call. Along K, each chunk writes its own partial C,
and the partial results are summed in a fixed order, so the result is
the same for any number of threads. The program times one BLAS call
of the full shape against the batch, and reports both in GB/s
(the bytes a GEMM must move at least) and GFLOPS/s.

```bash
./gemm_tas.x [M [N [K [panel [nrepeat [beta [hint]]]]]]]
```

Defaults: M=1000000, N=16, K=N, panel=0, nrepeat=3, beta=1.0, hint=1.
The panel is the width along the longest dimension; zero takes 256
rows or columns, or a chunk of 4096 along K. Each panel reports
whether a generated kernel or BLAS serves it. hint=1 dispatches with
`LIBXS_GEMM_FLAG_STREAM` and `LIBXS_GEMM_FLAG_REUSE`, i.e., a panel
that is a window into a larger matrix (along M, or a chunk along K)
may take a generated kernel, without JIT warm-up. hint=0 leaves the
choice to the automatic backend, which serves such windows with BLAS.
Compare against a threaded BLAS with `make MKL=2`.

## Validation

Set the CHECK environment variable to any non-zero value to print a
checksum of the first and the last C-matrix (`libxs_matdiff`):

```bash
CHECK=1 ./gemm_strided.x
CHECK=1 ./gemm_batch.x 23 23 23 1000 1 2
CHECK=1 ./gemm_index.x
CHECK=1 ./gemm_groups.x
CHECK=1 ./gemm_tas.x
```

`gemm_tas` also compares its result with the BLAS call and fails if the
relative difference (Frobenius norm) exceeds 1E-12.

When padding is enabled (pad>0), the check also verifies that
leading-dimension padding in C-matrices has not been overwritten.

## Runtime Controls

`LIBXS_GEMM_PANEL=N` sets the smallest column panel a split SYRK/SYR2K
uses; zero (the default) takes `LIBXS_GEMM_BN`. A larger value yields
fewer and wider panels, i.e., fewer tasks. `LIBXS_GEMM_BM`,
`LIBXS_GEMM_BN` and `LIBXS_GEMM_BK` size the tiles of a panel's diagonal
block when no BLAS SYRK is available.

Set `LIBXS_GEMM_PRINT=0` to print a compact GEMM registry summary at
termination. The first line reports registry size, capacity, memory,
and the selected `LIBXS_GEMM_BACKEND` policy; the second line reports
a histogram by datatype and backend; the third line reports the
occupied warm-up slots and how many of them settled without a JIT
kernel (`nojit`).
