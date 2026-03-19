# LIBXS

LIBXS is a portable C library with common (and not so common) functionality that has proven useful across a variety of applications. It provides building blocks for memory operations, numerics, synchronization, and more — with a focus on performance and minimal dependencies. LIBXS targets x86-64, AArch64, and RISC-V, and requires only a C89 compiler.

LIBXS was originally developed as part of [LIBXSMM](https://github.com/libxsmm/libxsmm) (its existence started even before LIBXSMM).

## Functionality

| Domain | Header | Description |
|--------|--------|-------------|
| [Memory](libxs_mem.md) | `libxs_mem.h` | Comparison (`memcmp`/`diff`), hashing (CRC32), matrix copy, transpose, and data shuffling. |
| [GEMM Batch](libxs_gemm.md) | `libxs_gemm.h` | Batched dense matrix multiplication (strided, pointer-array, grouped) with pluggable BLAS/JIT kernels. |
| [Math](libxs_math.md) | `libxs_math.h` | Matrix comparison, GCD/LCM, coprime calculation, and BF16 conversion. |
| [MHD](libxs_mhd.md) | `libxs_mhd.h` | Read and write MetaImage (MHD/MHA) format files for multidimensional arrays. |
| [Histogram](libxs_hist.md) | `libxs_hist.h` | Thread-safe histogram with running statistics and per-entry callbacks. |
| [Malloc](libxs_malloc.md) | `libxs_malloc.h` | Pool-based allocator for steady-state performance without system calls. |
| [RNG](libxs_rng.md) | `libxs_rng.h` | Thread-safe pseudo-random number generation (SplitMix64). |
| [Sync](libxs_sync.md) | `libxs_sync.h` | Portable atomics, locks, thread-local storage, and file locking. |
| [Timer](libxs_timer.md) | `libxs_timer.h` | High-resolution timing via calibrated TSC with OS clock fallback. |
| [CPUID](libxs_cpuid.md) | `libxs_cpuid.h` | CPU feature detection for x86-64 (SSE–AVX-512), AArch64, and RISC-V. |
| [Registry](libxs_reg.md) | `libxs_reg.h` | Thread-safe key-value store backed by hash table with per-thread caching. |
| [Utils](libxs_utils.md) | `libxs_utils.h` | Target-attribute machinery, ISA feature gates, bit-scan, and SIMD helpers. |

See also: [Fortran Interface](libxs_fortran.md), [Scripts](libxs_scripts.md).

## Build

The primary build system is GNU Make:

```
make
```

Strict C89 pedantic build (recommended for development):

```
make GNU=1 DBG=1 PEDANTIC=2
```

CMake is also supported (header-only or library):

```
cmake -S . -B build -DLIBXS_HEADER_ONLY=ON
cmake --build build
```

## Usage

**Library** — link against `libxs.a` (or `.so`) and include the desired headers:

```c
#include <libxs_mem.h>
#include <libxs_timer.h>
```

**Header-only** — include `libxs_source.h` in exactly one translation unit (no separate library needed):

```c
#include <libxs_source.h>
```

**Fortran** — use the provided module ([documentation](libxs_fortran.md)):

```fortran
USE :: libxs, ONLY: libxs_memcmp
```

## Samples

| Sample | Description |
|--------|-------------|
| [gemm](samples/libxs_gemm.md) | Batched DGEMM (strided, pointer-array, grouped) with OpenMP and optional MKL JIT. |
| [memory](samples/libxs_memory.md) | Benchmarks for `libxs_diff`, `libxs_memcmp`, matrix copy, and transpose. |
| [ozaki](samples/libxs_ozaki.md) | Ozaki-scheme low-precision GEMM with intercepted BLAS dispatch. |
| [registry](samples/libxs_registry.md) | Registry dispatch microbenchmark under various access patterns. |
| [scratch](samples/libxs_scratch.md) | Pool allocator (`libxs_malloc`/`libxs_free`) vs. system `malloc`. |
| [shuffle](samples/libxs_shuffle.md) | Comparison of shuffling strategies and throughput. |
| [sync](samples/libxs_sync.md) | Lock implementation microbenchmarks. |

## License

LIBXS is licensed under the [BSD 3-Clause License](LICENSE.md).
