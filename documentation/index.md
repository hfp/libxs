# LIBXS

LIBXS is a portable C library providing building blocks for memory
operations, numerics, synchronization, and more -- with a focus on
performance and minimal dependencies. Targets x86-64, AArch64, and
RISC-V; requires only a C89 compiler. Originally developed as part
of [LIBXSMM](https://github.com/libxsmm/libxsmm).

## Functionality

| Domain                                            | Header              | Description                                                    |
|---------------------------------------------------|---------------------|----------------------------------------------------------------|
| [Permutation](libxs_perm.md)        | `libxs_perm.h`      | Co-prime shuffling, smooth row permutations                    |
| [Histogram](libxs_hist.md)          | `libxs_hist.h`      | Thread-safe histogram with running statistics                  |
| [Registry](libxs_reg.md)            | `libxs_reg.h`       | Thread-safe key-value store with per-thread caching            |
| [Hashing](libxs_hash.md)            | `libxs_hash.h`      | CRC32-based hashing, Adler-32, string hashing                  |
| [Predict](libxs_predict.md)         | `libxs_predict.h`   | Fingerprint-guided parameter prediction with model persistence |
| [Malloc](libxs_malloc.md)           | `libxs_malloc.h`    | Pool-based allocator (steady-state, no system calls)           |
| [Memory](libxs_mem.md)              | `libxs_mem.h`       | Byte comparison, matrix copy/transpose, alignment queries      |
| [String](libxs_str.md)              | `libxs_str.h`       | Edit distance, substring search, word similarity, formatting   |
| [Timer](libxs_timer.md)             | `libxs_timer.h`     | High-resolution timing via calibrated TSC                      |
| [CPUID](libxs_cpuid.md)             | `libxs_cpuid.h`     | CPU feature detection (SSE to AVX-512, AArch64, RISC-V)        |
| [Utils](libxs_utils.md)             | `libxs_utils.h`     | ISA feature gates, bit-scan, SIMD helpers                      |
| [Sync](libxs_sync.md)               | `libxs_sync.h`      | Portable atomics, locks, TLS, and file locking                 |
| [GEMM](libxs_gemm.md)               | `libxs_gemm.h`      | Batched dense GEMM (strided, pointer-array, grouped)           |
| [Math](libxs_math.md)               | `libxs_math.h`      | Matrix comparison, GCD/LCM, coprime, BF16 conversion           |
| [MHD](libxs_mhd.md)                 | `libxs_mhd.h`       | Read/write MetaImage (MHD/MHA) files                           |
| [RNG](libxs_rng.md)                 | `libxs_rng.h`       | Thread-safe pseudo-random number generation (SplitMix64)       |

See also: [Fortran Interface](libxs_fortran.md),
[Scripts](libxs_scripts.md).

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

## Installation

Install into a chosen prefix:

```bash
make GNU=1 -j $(nproc) install PREFIX=$HOME/libxs
```

This installs headers, the Fortran module, the static and shared
libraries, and the header-only source tree under `PREFIX`.

Out-of-tree builds are also supported:

```bash
mkdir /tmp/libxs-build && cd /tmp/libxs-build
make -j $(nproc) -f /path/to/libxs/Makefile
```

## Usage

**Library** -- link against `libxs.a` (or `.so`) and include the
desired headers:

```c
#include <libxs_mem.h>
#include <libxs_timer.h>
```

**Header-only** (explicit) -- include `libxs_source.h` (no
separate library needed). Safe to include from multiple
translation units:

```c
#include <libxs_source.h>
```

**Header-only** (implicit) -- compile with `-DLIBXS_SOURCE` and
any LIBXS public header automatically includes the implementation.
No special include order is required. When used through
[LIBXSTREAM](https://github.com/hfp/libxstream) without a
pre-built library (`-DLIBXSTREAM_SOURCE`), `LIBXS_SOURCE` is
implied automatically.

**Fortran** -- use the provided module
([documentation](libxs_fortran.md)):

```fortran
USE :: libxs, ONLY: libxs_memcmp
```

## Samples

| Sample                                                    | Description                                                  |
|-----------------------------------------------------------|--------------------------------------------------------------|
| [registry](samples/libxs_registry.md)       | Registry dispatch microbenchmark                             |
| [rosetta](samples/libxs_rosetta.md)         | Hierarchical type discovery on opaque binary data            |
| [predict](samples/libxs_predict.md)         | Train a prediction model from CSV and save it                |
| [shuffle](samples/libxs_shuffle.md)         | Shuffling strategies comparison                              |
| [scratch](samples/libxs_scratch.md)         | Pool allocator vs system malloc                              |
| [memory](samples/libxs_memory.md)           | Benchmarks for comparison, matrix copy, and transpose        |
| [gemm](samples/libxs_gemm.md)               | Batched DGEMM (strided, pointer-array, grouped) with OMP     |
| [syrk](samples/libxs_syrk.md)               | Symmetric rank-k/2k update (SYRK/SYR2K) with validation      |
| [ozaki](samples/libxs_ozaki.md)             | Ozaki-scheme low-precision GEMM with intercepted BLAS        |
| [sync](samples/libxs_sync.md)               | Lock implementation microbenchmarks                          |

## License

[BSD 3-Clause](LICENSE.md)
