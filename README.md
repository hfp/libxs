# LIBXS

LIBXS is a portable C library providing building blocks for memory
operations, numerics, synchronization, and more — with a focus on
performance and minimal dependencies. Targets x86-64, AArch64, and
RISC-V; requires only a C89 compiler. Originally developed as part
of [LIBXSMM](https://github.com/libxsmm/libxsmm).

## Functionality

| Domain           | Header             | Description                                |
|------------------|--------------------|--------------------------------------------|
| [Registry][dreg] | `libxs_reg.h`      | Thread-safe key-value store                |
| [Predict][dprd]  | `libxs_predict.h`  | Fingerprint-guided parameter prediction    |
| [Malloc][dmal]   | `libxs_malloc.h`   | Pool allocator (no system calls at steady) |
| [Memory][dmem]   | `libxs_mem.h`      | Byte compare, matrix copy/transpose        |
| [String][dstr]   | `libxs_str.h`      | Edit distance, substring, similarity       |
| [Token][dtok]    | `libxs_token.h`    | Fixed-width 8-byte tokenizer               |
| [Timer][dtmr]    | `libxs_timer.h`    | High-resolution timing via calibrated TSC  |
| [CPUID][dcpu]    | `libxs_cpuid.h`    | CPU feature detection (SSE..AVX-512, etc.) |
| [Utils][dutil]   | `libxs_utils.h`    | ISA gates, bit-scan, SIMD helpers          |
| [Sync][dsync]    | `libxs_sync.h`     | Portable atomics, locks, TLS, file locks   |
| [GEMM][dgemm]    | `libxs_gemm.h`     | Batched dense GEMM (strided, grouped)      |
| [Math][dmath]    | `libxs_math.h`     | Matrix compare, GCD/LCM, BF16 conversion   |
| [Hash][dhash]    | `libxs_hash.h`     | CRC32, Adler-32, string hashing            |
| [Perm][dperm]    | `libxs_perm.h`     | Shuffle, kd-tree, Hilbert/Morton curves    |
| [Hist][dhist]    | `libxs_hist.h`     | Thread-safe histogram, running statistics  |
| [MHD][dmhd]      | `libxs_mhd.h`      | Read/write MetaImage (MHD/MHA) files       |
| [RNG][drng]      | `libxs_rng.h`      | Pseudo-random generation (SplitMix64)      |

[dreg]:  documentation/libxs_reg.md
[dprd]:  documentation/libxs_predict.md
[dmal]:  documentation/libxs_malloc.md
[dmem]:  documentation/libxs_mem.md
[dstr]:  documentation/libxs_str.md
[dtok]:  documentation/libxs_token.md
[dtmr]:  documentation/libxs_timer.md
[dcpu]:  documentation/libxs_cpuid.md
[dutil]: documentation/libxs_utils.md
[dsync]: documentation/libxs_sync.md
[dgemm]: documentation/libxs_gemm.md
[dmath]: documentation/libxs_math.md
[dhash]: documentation/libxs_hash.md
[dperm]: documentation/libxs_perm.md
[dhist]: documentation/libxs_hist.md
[dmhd]:  documentation/libxs_mhd.md
[drng]:  documentation/libxs_rng.md

See also: [Fortran Interface](documentation/libxs_fortran.md),
[Scripts](documentation/libxs_scripts.md).

## Build

```bash
make GNU=1
```

The library is compiled for SSE4.2 by default but dynamically
dispatches to the best ISA available at runtime (up to AVX-512).
Use `SSE=0` to compile natively for the build host.

| Variable | Default | Description                                 |
|----------|---------|---------------------------------------------|
| GNU      | 0       | Use GNU GCC-compatible compiler             |
| DBG      | 0       | Debug build                                 |
| SYM      | 0       | Include debug symbols (-g)                  |
| SSE      | 1       | x86 baseline: 0=native, 1=SSE4.2 (portable) |

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

**Library** — link against `libxs.a` (or `.so`) and include the
desired headers:

```c
#include <libxs/libxs_mem.h>
#include <libxs/libxs_timer.h>
```

**Header-only** (explicit) — include `libxs_source.h` (no
separate library needed). Safe to include from multiple
translation units:

```c
#include <libxs/libxs_source.h>
```

**Header-only** (implicit) — compile with `-DLIBXS_SOURCE` and
any LIBXS public header automatically includes the implementation.
No special include order is required. When used through
[LIBXSTREAM](https://github.com/hfp/libxstream) without a
pre-built library (`-DLIBXSTREAM_SOURCE`), `LIBXS_SOURCE` is
implied automatically.

**Fortran** — use the provided module
([documentation](documentation/libxs_fortran.md)):

```fortran
USE :: libxs, ONLY: libxs_memcmp
```

## Samples

| Sample        | Description                                     |
|---------------|-------------------------------------------------|
| [tokenizer]   | Reversible byte-level tokenizer                 |
| [converse]    | Extractive summarization via tokenized fprints  |
| [registry]    | Registry dispatch microbenchmark                |
| [predict]     | Fingerprint-guided parameter prediction         |
| [rosetta]     | Hierarchical type discovery on opaque data      |
| [setdiff]     | Deterministic set-difference tolerance          |
| [shuffle]     | Shuffling strategies comparison                 |
| [scratch]     | Pool allocator vs system malloc                 |
| [spatial]     | Stratification, pair counting, kd-tree          |
| [fprint]      | Foeppl fingerprint structure/geometry tests     |
| [memory]      | Byte comparison, matrix copy, transpose         |
| [ozaki]       | Ozaki-scheme low-precision GEMM                 |
| [gemm]        | Batched DGEMM (strided, pointer, grouped)       |
| [syrk]        | Symmetric rank-k/2k update with validation      |
| [sync]        | Lock implementation microbenchmarks             |

[tokenizer]: documentation/samples/libxs_tokenizer.md
[converse]:  documentation/samples/libxs_converse.md
[registry]:  documentation/samples/libxs_registry.md
[predict]:   documentation/samples/libxs_predict.md
[rosetta]:   documentation/samples/libxs_rosetta.md
[setdiff]:   documentation/samples/libxs_setdiff.md
[shuffle]:   documentation/samples/libxs_shuffle.md
[scratch]:   documentation/samples/libxs_scratch.md
[spatial]:   documentation/samples/libxs_spatial.md
[fprint]:    documentation/samples/libxs_fprint.md
[memory]:    documentation/samples/libxs_memory.md
[ozaki]:     documentation/samples/libxs_ozaki.md
[gemm]:      documentation/samples/libxs_gemm.md
[syrk]:      documentation/samples/libxs_syrk.md
[sync]:      documentation/samples/libxs_sync.md

## License

[BSD 3-Clause](LICENSE.md)
