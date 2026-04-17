# LIBXS

LIBXS is a portable C library providing building blocks for memory
operations, numerics, synchronization, and more -- with a focus on
performance and minimal dependencies. Targets x86-64, AArch64, and
RISC-V; requires only a C89 compiler. Originally developed as part
of [LIBXSMM](https://github.com/libxsmm/libxsmm).

## Functionality

| Domain                                            | Header               | Description                                                   |
|---------------------------------------------------|----------------------|---------------------------------------------------------------|
| [Memory](documentation/libxs_mem.md)              | `libxs_mem.h`        | Comparison, hashing (CRC32), matrix copy/transpose, shuffle   |
| [GEMM](documentation/libxs_gemm.md)               | `libxs_gemm.h`       | Batched dense GEMM (strided, pointer-array, grouped)          |
| [Math](documentation/libxs_math.md)               | `libxs_math.h`       | Matrix comparison, GCD/LCM, coprime, BF16 conversion          |
| [MHD](documentation/libxs_mhd.md)                 | `libxs_mhd.h`        | Read/write MetaImage (MHD/MHA) files                          |
| [Histogram](documentation/libxs_hist.md)          | `libxs_hist.h`       | Thread-safe histogram with running statistics                 |
| [Malloc](documentation/libxs_malloc.md)           | `libxs_malloc.h`     | Pool-based allocator (steady-state, no system calls)          |
| [RNG](documentation/libxs_rng.md)                 | `libxs_rng.h`        | Thread-safe pseudo-random number generation (SplitMix64)      |
| [Sync](documentation/libxs_sync.md)               | `libxs_sync.h`       | Portable atomics, locks, TLS, and file locking                |
| [Timer](documentation/libxs_timer.md)             | `libxs_timer.h`      | High-resolution timing via calibrated TSC                     |
| [CPUID](documentation/libxs_cpuid.md)             | `libxs_cpuid.h`      | CPU feature detection (SSE to AVX-512, AArch64, RISC-V)       |
| [Registry](documentation/libxs_reg.md)            | `libxs_reg.h`        | Thread-safe key-value store with per-thread caching           |
| [Utils](documentation/libxs_utils.md)             | `libxs_utils.h`      | ISA feature gates, bit-scan, SIMD helpers                     |

See also: [Fortran Interface](documentation/libxs_fortran.md),
[Scripts](documentation/libxs_scripts.md).

## Build

```bash
make GNU=1
```

The library is compiled for SSE4.2 by default but dynamically
dispatches to the best ISA available at runtime (up to AVX-512).
Use `SSE=0` to compile natively for the build host.

| Variable   | Default   | Description                                     |
|------------|-----------|-------------------------------------------------|
| GNU        | 0         | Use GNU GCC-compatible compiler                 |
| DBG        | 0         | Debug build                                     |
| SYM        | 0         | Include debug symbols (-g)                      |
| SSE        | 1         | x86 baseline: 0=native, 1=SSE4.2 (portable)     |

CMake is also supported (header-only or library):

```bash
cmake -S . -B build -DLIBXS_HEADER_ONLY=ON
cmake --build build
```

## Usage

**Library** -- link against `libxs.a` (or `.so`) and include the
desired headers:

```c
#include <libxs_mem.h>
#include <libxs_timer.h>
```

**Header-only** -- include `libxs_source.h` in exactly one
translation unit (no separate library needed):

```c
#include <libxs_source.h>
```

**Fortran** -- use the provided module
([documentation](documentation/libxs_fortran.md)):

```fortran
USE :: libxs, ONLY: libxs_memcmp
```

## Samples

| Sample                                                    | Description                                                  |
|-----------------------------------------------------------|--------------------------------------------------------------|
| [gemm](documentation/samples/libxs_gemm.md)               | Batched DGEMM (strided, pointer-array, grouped) with OMP     |
| [memory](documentation/samples/libxs_memory.md)           | Benchmarks for comparison, matrix copy, and transpose        |
| [ozaki](documentation/samples/libxs_ozaki.md)             | Ozaki-scheme low-precision GEMM with intercepted BLAS        |
| [registry](documentation/samples/libxs_registry.md)       | Registry dispatch microbenchmark                             |
| [scratch](documentation/samples/libxs_scratch.md)         | Pool allocator vs system malloc                              |
| [shuffle](documentation/samples/libxs_shuffle.md)         | Shuffling strategies comparison                              |
| [sync](documentation/samples/libxs_sync.md)               | Lock implementation microbenchmarks                          |

## License

[BSD 3-Clause](LICENSE.md)
