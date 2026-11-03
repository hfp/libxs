# LIBXS

LIBXS is a portable C library providing building blocks for memory
operations, numerics, synchronization, and more — with a focus on
performance and minimal dependencies. Targets x86-64, AArch64, and
RISC-V; requires only a C89 compiler. Originally developed as part
of [LIBXSMM](https://github.com/libxsmm/libxsmm).

## Functionality

| Domain           | Header             | Description                                                                      |
|------------------|--------------------|----------------------------------------------------------------------------------|
| [Core][dcore]    | `libxs.h`          | Initialization, verbosity, data types                                            |
| [Registry][dreg] | `libxs_reg.h`      | Thread-safe key-value store                                                      |
| [Predict][dprd]  | `libxs_predict.h`  | Fingerprint-guided parameter prediction                                          |
| [Malloc][dmal]   | `libxs_malloc.h`   | Pool allocator (steady state: no syscalls)                                       |
| [Memory][dmem]   | `libxs_mem.h`      | Byte compare, matrix copy/transpose                                              |
| [String][dstr]   | `libxs_str.h`      | Edit distance, substring, similarity                                             |
| [N-gram][dngr]   | `libxs_ngram.h`    | Variable-order n-gram counts and backoff                                         |
| [Token][dtok]    | `libxs_token.h`    | Reversible metatokens and lexical IDs                                            |
| [Timer][dtmr]    | `libxs_timer.h`    | High-resolution timing via calibrated TSC                                        |
| [CPUID][dcpu]    | `libxs_cpuid.h`    | CPU feature detection (x86-64, AArch64, RISC-V)                                  |
| [Utils][dutil]   | `libxs_utils.h`    | ISA gates, bit-scan, SIMD helpers                                                |
| [Sync][dsync]    | `libxs_sync.h`     | Atomics, locks, TLS, barrier, ring buffer, singleton                             |
| [GEMM][dgemm]    | `libxs_gemm.h`     | GEMM batches (pointer, index, strided), SYRK/SYR2K                               |
| [Math][dmath]    | `libxs_math.h`     | Matrix compare, fingerprints, setdiff, compensated sums, modular math, FP16/BF16 |
| [Hash][dhash]    | `libxs_hash.h`     | CRC32, Adler-32, string hashing                                                  |
| [Pack][dpack]    | `libxs_pack.h`     | Bit packing at fixed or per-group widths                                         |
| [Perm][dperm]    | `libxs_perm.h`     | Shuffle, kd-tree, Hilbert/Morton curves                                          |
| [Hist][dhist]    | `libxs_hist.h`     | Thread-safe histogram, running statistics                                        |
| [Mix][dmix]      | `libxs_mix.h`      | Fixed-share mixture over expert banks                                            |
| [MHD][dmhd]      | `libxs_mhd.h`      | Read/write MetaImage (MHD/MHA) files                                             |
| [RNG][drng]      | `libxs_rng.h`      | Pseudo-random generation (SplitMix64)                                            |

[dcore]: libxs_core.md
[dreg]: libxs_reg.md
[dprd]: libxs_predict.md
[dmal]: libxs_malloc.md
[dmem]: libxs_mem.md
[dstr]: libxs_str.md
[dngr]: libxs_ngram.md
[dtok]: libxs_token.md
[dtmr]: libxs_timer.md
[dcpu]: libxs_cpuid.md
[dutil]: libxs_utils.md
[dsync]: libxs_sync.md
[dgemm]: libxs_gemm.md
[dmath]: libxs_math.md
[dhash]: libxs_hash.md
[dpack]: libxs_pack.md
[dperm]: libxs_perm.md
[dhist]: libxs_hist.md
[dmix]: libxs_mix.md
[dmhd]: libxs_mhd.md
[drng]: libxs_rng.md

See also: [Fortran Interface](libxs_fortran.md),
[Scripts](libxs_scripts.md).

## Build

```bash
make
```

The library is compiled for the build host's highest SSE level (up to SSE4.2)
by default but dynamically dispatches to the best ISA available at runtime
(up to AVX-512).
Use `SSE=0` to compile natively for the build host.
The library is built into `lib/<target>`, e.g., `lib/x86_64` by default or
`lib/<cpu>` with `SSE=0`, and `lib/` refers to the latest build.

| Variable | Default | Description                                          |
|----------|---------|------------------------------------------------------|
| GNU      | 1       | GNU GCC-compatible compiler (0: Intel)               |
| DBG      | 0       | Debug build                                          |
| SYM      | 0       | Include debug symbols (-g)                           |
| SSE      | 1       | x86 baseline: 0=native, 1=host's SSE level           |
| LIBARCH  | (auto)  | Name of the build folder (`lib/<target>`)            |
| STATIC   | (both)  | 1: static library only, 0: shared library only       |
| FORTRAN  | (auto)  | 0: no Fortran module, even if a compiler is found    |
| OMP      | 0       | 1: compile with OpenMP (the library does not use it) |

Tests and samples:

```bash
make tests
make samples
```

CMake is also supported:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix $HOME/libxs
```

The C library uses OpenMP SIMD when supported by the compiler, without an
OpenMP runtime dependency. The Fortran module's internal OpenMP parallelism is
opt-in (`LIBXS_FORTRAN_OMP=ON`). Tests and samples enable their own OpenMP by
default; set `LIBXS_SAMPLES_OMP=OFF` to disable it. Both options are available
only when their corresponding targets are built. On macOS, Homebrew's `libomp`
is detected via `brew --prefix libomp`; use `OpenMP_ROOT` for another installation.

| Option                    | Default | Description                                 |
|---------------------------|---------|---------------------------------------------|
| LIBXS_FORTRAN             | (auto)  | Build the Fortran module                    |
| LIBXS_FORTRAN_OMP         | OFF     | Enable OpenMP in the Fortran module          |
| LIBXS_SAMPLES_OMP         | ON      | Enable OpenMP in samples and tests           |
| LIBXS_INSTALL_HEADER_ONLY | OFF     | Also install the header-only implementation |
| BUILD_TESTING             | OFF     | Build the tests (requires BLAS)             |
| BUILD_SHARED_LIBS         | OFF     | Build the shared library                    |

## Installation

Install into a chosen prefix:

```bash
make -j $(nproc) install PREFIX=$HOME/libxs
```

This installs headers, the Fortran module, the static and shared
libraries, the header-only source tree, pkg-config files, and CMake
package files under `PREFIX`. A given `DESTDIR` is prepended to
`PREFIX` (staged installation).

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

Both builds install pkg-config files (`libxs.pc`) and a CMake package
(target `libxs::libxs`). A static library needs `--static` to also
list its dependencies (pthread, libm, libdl, librt):

```bash
cc app.c $(pkg-config --cflags libxs) $(pkg-config --libs --static libxs)
```

```cmake
find_package(libxs REQUIRED)
target_link_libraries(app PRIVATE libxs::libxs)
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
implied automatically. Either way, the application still links
`-lpthread -lm -ldl` (and `-lrt` with older C libraries).

**Fortran** — `USE LIBXS` is part of the library, and `libxs.mod`
is installed next to the headers (`include/libxs`). `USE LIBXS_JIT`
adds GEMM dispatch with the kernel providers filled in; the
application compiles `libxs_jit.F` itself (see
[Kernel Providers](libxs_gemm.md#kernel-providers)
and the [Fortran Interface](libxs_fortran.md)):

```fortran
      USE :: LIBXS, ONLY: libxs_memcmp
```

**Environment** — read by the library at initialization or exit:

| Variable         | Default   | Description                                                                                                     |
|------------------|-----------|-----------------------------------------------------------------------------------------------------------------|
| LIBXS_VERBOSE    | 0         | Verbosity level; empty means 1, negative means all (debug build: all)                                           |
| LIBXS_PACK       | 1         | Format of libxs_vpack (at the first call): 1 as vpack.f, 2 smaller, 3 smallest, unpacks slower (see Pack)       |
| LIBXS_PTRACE     | 0         | Linux: any process may attach a debugger (not only an ancestor)                                                 |
| LIBXS_SIGNAL     | (verbose) | Report at SIGABRT/SIGSEGV, then pass the signal on (on if verbose); 0 leaves handlers alone                     |
| LIBXS_DUMP_BUILD | 0         | Print the build state (GNU Make build): >0 at exit, <0 at startup                                               |
| LIBXS_DUMP_FILES | -         | At exit, print the listed files (`,;:`-separated, `PID` expands); a number N prints the PID and waits N seconds |

`LIBXS_DUMP_FILE` is an alias of `LIBXS_DUMP_FILES`. Both dumps are
printed by the first LIBXS process on the system only.

## Samples

| Sample        | Description                                     |
|---------------|-------------------------------------------------|
| [tokenizer]   | Reversible configurable metatokenizer           |
| [converse]    | Extractive summarization via tokenized fprints  |
| [registry]    | Registry dispatch microbenchmark                |
| [predict]     | Fingerprint-guided parameter prediction         |
| [rosetta]     | Hierarchical type discovery on opaque data      |
| [setdiff]     | Deterministic set-difference tolerance          |
| [shuffle]     | Shuffling strategies comparison                 |
| [pack]        | Bit packing at fixed vs per-group width         |
| [scratch]     | Pool allocator vs system malloc                 |
| [spatial]     | Stratification, pair counting, kd-tree          |
| [fprint]      | Fingerprint structure/geometry tests            |
| [bracket]     | Sharp edges from bracket units (Adam, GEMM)     |
| [memory]      | Byte comparison, matrix copy, transpose         |
| [ozaki]       | Ozaki-scheme low-precision GEMM                 |
| [gemm]        | Batched DGEMM (strided, pointer, grouped)       |
| [syrk]        | Symmetric rank-k/2k update with validation      |
| [sync]        | Lock implementation microbenchmarks             |

[tokenizer]: samples/libxs_tokenizer.md
[converse]: samples/libxs_converse.md
[registry]: samples/libxs_registry.md
[predict]: samples/libxs_predict.md
[rosetta]: samples/libxs_rosetta.md
[setdiff]: samples/libxs_setdiff.md
[shuffle]: samples/libxs_shuffle.md
[pack]: samples/libxs_pack.md
[scratch]: samples/libxs_scratch.md
[spatial]: samples/libxs_spatial.md
[fprint]: samples/libxs_fprint.md
[bracket]: samples/libxs_bracket.md
[memory]: samples/libxs_memory.md
[ozaki]: samples/libxs_ozaki.md
[gemm]: samples/libxs_gemm.md
[syrk]: samples/libxs_syrk.md
[sync]: samples/libxs_sync.md

## License

[BSD 3-Clause](LICENSE.md)
