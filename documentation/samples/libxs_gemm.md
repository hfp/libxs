# Batched GEMM

This sample code exercises the LIBXS batched GEMM API (`libxs_gemm.h`) in several flavors: strided, pointer-array, indexed, and grouped. All programs are multithreaded via OpenMP (using the `_task` variants where applicable), report wall-clock time and GFLOPS/s, and optionally validate results via `libxs_matdiff`.

## Building

LIBXS must be built first from the repository root. Then:

```bash
cd samples/gemm
make
```

For strict C89 pedantic checking:

```bash
make GNU=1 DBG=1 PEDANTIC=2
```

## Programs

### gemm_strided

Strided batch DGEMM: all matrices are packed contiguously in memory with constant stride between consecutive A, B, and C matrices. Uses `libxs_gemm_index` with `index_stride=0` (constant-stride mode). With OpenMP, work is distributed across threads via `libxs_gemm_index_task`.

```
./gemm_strided.x [M [N [K [batchsize [nrepeat [beta [pad]]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, beta=1.0, pad=0.

### gemm_batch

Pointer-array batch DGEMM: matrices are accessed through arrays of pointers. Supports a duplicate C-matrix mode to exercise the lock-forward synchronization path. With OpenMP, work is distributed via `libxs_gemm_batch_task`.

```
./gemm_batch.x [M [N [K [batchsize [nrepeat [dup [beta [pad]]]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, dup=0, beta=1.0, pad=0.

The `dup` argument controls duplicate C-matrix references:

| dup | Mode | Description |
|-----|------|-------------|
| 0 | None | Each C-matrix is unique (`LIBXS_GEMM_FLAG_NOLOCK`). |
| 1 | Sorted | Half-unique C-pointers, duplicates consecutive. |
| 2 | Shuffled | Half-unique C-pointers, randomly shuffled (stresses lock-forward). |

### gemm_index

Index-array batch DGEMM: matrices live in contiguous buffers, but each batch element is addressed through explicit element-offset index arrays (`ia`, `ib`, `ic`). Uses `libxs_gemm_index` with `index_stride=sizeof(int)` (per-element index mode). With OpenMP, work is distributed via `libxs_gemm_index_task`.

```
./gemm_index.x [M [N [K [batchsize [nrepeat [beta [pad]]]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3, beta=1.0, pad=0.

### gemm_indexf

Fortran variant of `gemm_index`. Demonstrates the LIBXS Fortran module interface with one-based index arrays and `C_LOC`/`C_SIZEOF` interoperability. Requires a Fortran compiler.

```
./gemm_indexf.x [M [N [K [batchsize [nrepeat]]]]]
```

Defaults: M=N=K=23, batchsize=30000, nrepeat=3.

### gemm_groups

Grouped batch DGEMM: multiple groups of different matrix shapes are dispatched in sequence, each with its own `libxs_gemm_config_t` and JIT kernel. Per-group dimensions grow by 4 starting from `base_m`. All groups use pointer-array mode via `libxs_gemm_batch`.

```
./gemm_groups.x [ngroups [batch_per_group [nrepeat [base_m [beta [pad]]]]]]
```

Defaults: ngroups=2 (max 4), batch_per_group=30000, nrepeat=3, base_m=8, beta=1.0, pad=0.

## MKL JIT Support

The Fortran program (`gemm_indexf`) explicitly creates a JIT-compiled DGEMM kernel via `mkl_jit_create_dgemm` when built with Intel MKL (`__MKL` defined) and wires it into `libxs_gemm_config_t`. The C programs use `libxs_gemm_dispatch`, which may select a JIT kernel internally depending on the build configuration.

## Validation

Set the `CHECK` environment variable to enable a post-run sanity check using `libxs_matdiff`:

```bash
CHECK=1 ./gemm_strided.x
CHECK=1 ./gemm_batch.x 23 23 23 1000 1 2
CHECK=1 ./gemm_index.x
CHECK=1 ./gemm_groups.x
```

When padding is enabled (`pad>0`), the check also verifies that leading-dimension padding in C-matrices has not been overwritten.
