# Batched GEMM

Exercises the LIBXS batched GEMM API (`libxs_gemm.h`) in several
flavors: strided, pointer-array, indexed, and grouped. All programs
are multithreaded via OpenMP, report wall-clock time and GFLOPS/s,
and optionally validate results via `libxs_matdiff`.

## Building

```bash
cd samples/gemm
make GNU=1
```

LIBXS must be built first from the repository root.

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
Requires a Fortran compiler.

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

## Validation

Set the CHECK environment variable to validate results:

```bash
CHECK=1 ./gemm_strided.x
CHECK=1 ./gemm_batch.x 23 23 23 1000 1 2
CHECK=1 ./gemm_index.x
CHECK=1 ./gemm_groups.x
```

When padding is enabled (pad>0), the check also verifies that
leading-dimension padding in C-matrices has not been overwritten.
