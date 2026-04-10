# Ozaki-Scheme Low-Precision GEMM

## Intercepted GEMM

This code sample intercepts all four standard BLAS GEMM routines - DGEMM, SGEMM, ZGEMM, and CGEMM - and only relies on the LAPACK/BLAS interface. Real GEMM calls (DGEMM/SGEMM) are executed via one of two Ozaki low-precision schemes (mantissa slicing or CRT); complex GEMM calls (ZGEMM/CGEMM) are implemented via the 3M (Karatsuba) method using three real GEMM calls each. The wrapper sources are compiled twice (once for `double`, once for `float`), so all four symbols coexist in a single binary.

Two link-time variants are built per precision: (1) code which is dynamically linked against LAPACK/BLAS (`dgemm-blas.x`/`sgemm-blas.x`), (2) code which is linked using `--wrap=`*symbol* supported by GNU GCC compatible tool chains (`dgemm-wrap.x`/`sgemm-wrap.x`). Running `test-wrap.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux). The `test-check.sh` script validates both Ozaki schemes for correctness, and `test-mhd.sh` tests MHD file-based GEMM input pairs.

Both the static wrapper library (`libwrap.a`) and the shared wrapper library (`libwrap.so`) are built by default. The static library is suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_`). The shared library is used via `LD_PRELOAD`:

```bash
cd /path/to/libxs
make -j $(nproc)

cd samples/ozaki
make

LD_PRELOAD=/path/to/libwrap.so ./application
```

Note: LIBXS has to be built upfront for the sample code to link.

## Performance

The Ozaki sample code demonstrates how to intercept GEMMs (any BLAS library), to run a low-precision GEMM instead of the original GEMM, to compare the results, and to maintain running statistics presented every *N*th call of GEMM or when the application terminates. The code explores high-precision emulation using low-precision calculation.

The blocking structure emulates fixed-size matrix-multiply hardware. All computation operates on tiles of size `BLOCK_M` x `BLOCK_K` (A) and `BLOCK_K` x `BLOCK_N` (B), with defaults BLOCK_M = BLOCK_N = BLOCK_K = 16.

The compile-time parameter `BATCH_K` (default 4) groups consecutive BLOCK_K panels into a single batch (effective K-step = BATCH_K x BLOCK_K). Batching reduces OpenMP barrier overhead and improves temporal reuse of the C tile across K iterations, while keeping the fundamental tile size visible throughout the code.

Preprocessing (exponent alignment, mantissa slicing or modular reduction) accounts for roughly 5% of runtime; the remaining 95% is spent in the inner dot-product loops. Int8 dot products are dispatched once per GEMM via a function pointer: AVX-VNNI-INT8 (single instruction) when available, AVX-512 VNNI with bias correction (two instructions) otherwise, or a scalar fallback. The number of pairwise slice products is quadratic in the number of slices, so for double-precision (8 slices, default) the inner loop performs up to 36 dot products per block pair. Scheme 2 performs one modular dot product per prime, so its cost is linear in the number of primes.

OpenMP parallelizes all three phases of each K-batch: Phase 1 preprocesses A panels (`schedule(dynamic) nowait`), Phase 2 preprocesses B panels (`schedule(dynamic)`, implicit barrier), and Phase 3 accumulates the dot products into C (`collapse(2) schedule(static)`). Panel buffers are shared across the parallel region and sized by `BATCH_K` x number-of-blocks, allocated via `libxs_malloc`.

An arithmetic intensity threshold (`OZAKI_THRESHOLD`, default 12) controls whether the Ozaki scheme is applied. GEMM calls with insufficient arithmetic intensity (flops / (bytes x threshold) < 1) fall through to the original BLAS. Setting the threshold to zero forces all GEMM calls through the Ozaki path.

When built with [LIBXSTREAM](https://github.com/hfp/libxstream) support (sibling `libxstream` repository detected at build time), an optional OpenCL/GPU path is available for both Scheme 1 and Scheme 2 (see `OZAKI_OCL` environment variable). The GPU bridge (`ozaki_ocl.c`) wraps LIBXSTREAM behind an opaque handle so that the CPU code has no OpenCL dependency. At runtime, the GPU path is enabled by default (`OZAKI_OCL=1`) and can be disabled with `OZAKI_OCL=0` to fall back to the CPU implementation. The [LIBXSTREAM Ozaki sample](https://github.com/hfp/libxstream/tree/main/samples/ozaki) documents additional GPU-specific environment variables and features (XMX/DPAS acceleration, TinyTC kernels, preprocessing cache, kernel profiling). The CPU-only code performs the low-precision conversion on-the-fly and only requires a reasonable stack size to buffer small matrix blocks.

## Scheme 1 - Mantissa Slicing

Scheme 1 (`OZAKI=1`, the default) decomposes each IEEE-754 mantissa into 7-bit int8 slices and accumulates all pairwise slice products via low-precision GEMM. The number of slices determines achievable accuracy and can be set at runtime via `OZAKI_N`. The default and maximum vary by precision (double: default 8, max 16; float: default 4, max 8). The size of the matrices employed by potential "matrix cores" is set at compile-time with `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`; grouping of consecutive K-panels is controlled by `BATCH_K`. The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

## Scheme 1 Flags (`OZAKI_FLAGS`) and Diagonal Trim (`OZAKI_TRIM`)

The Scheme 1 loop is controlled by two runtime knobs:

### Flags

The environment variable `OZAKI_FLAGS` is an integer bitmask:

| Bit | Value | Flag | Description |
|:---:|:-----:|------|-------------|
| 0 | 1 | Triangular | Only iterate slice pairs (sa,sb) with sb>=sa (upper triangle). |
| 1 | 2 | Symmetrize | For each off-diagonal pair (sa,sb) in the upper triangle, also compute the mirror dot product D(sb,sa) at negligible extra cost (one additional int8 dot product per pair). |

The default value is **3** (both flags). With both flags, all S^2 contributions are covered using only S*(S+1)/2 dot products. Setting `OZAKI_FLAGS=0` runs the full S^2 square of slice pairs (same accuracy, higher cost).

### Diagonal Trim

The environment variable `OZAKI_TRIM` drops the T least significant diagonals from the slice-pair iteration. Pairs with sa+sb > 2*(S-1)-T are skipped. Pair significance scales as 2^(low_bit[sa]+low_bit[sb]), so sa+sb directly determines significance - smaller sums are more significant. Each dropped diagonal loses approximately 7 bits of relative precision.

The default value is **0** (exact: all pairs). The value is clamped so that at least diagonal 0 is always computed.

**Cost overview** for *S* slices with TRIANGULAR+SYMMETRIZE (default flags):

| `OZAKI_TRIM` | Dot products | Pairs covered | Relative bits lost |
|:---:|:---:|:---:|:---:|
| 0 (default) | S*(S+1)/2 | all S^2 | 0 (exact) |
| S-1 | ~S^2/4 | (S+1)*S/2 | ~7*(S-1) least significant |
| 3S/2 | ~S^2/8 | ~S^2/4 | ~7*3S/2 least significant |
| 2*(S-1) | 1 | 1 (only diagonal) | ~7*2*(S-1) |

Examples:

```bash
./dgemm-wrap.x 256                      # exact (default: flags=3, trim=0)
OZAKI_TRIM=4 ./dgemm-wrap.x 256        # drop 4 least significant diagonals
OZAKI_FLAGS=0 ./dgemm-wrap.x 256       # full S^2 square, no symmetrize
```

## Scheme 2 - Chinese Remainder Theorem

Scheme 2 (`OZAKI=2`) uses modular arithmetic instead of mantissa slicing. Each matrix element is reduced modulo a set of small pairwise coprime moduli (<= 256) so that residues fit in uint8 and dot products use VNNI instructions when available. GEMM is performed independently modulo each modulus, and the exact integer result is recovered via the Chinese Remainder Theorem (Garner's algorithm with grouped uint64 Horner evaluation). Because the work is linear in the number of moduli - versus quadratic in the number of slices for Scheme 1 - Scheme 2 can be more efficient when many moduli/slices are needed.

Residues are unsigned uint8 in [0, p-1] with the element sign encoded via modular additive inverse (p - r). This enables u8 VNNI dot products: VPDPBUUD (single instruction) on AVX-VNNI-INT8 hardware, or VPDPBUSD with bias correction otherwise. A signed i8 equivalent path is available via `OZAKI_I8=1`.

The number of moduli can be set at runtime via `OZAKI_N`. Defaults: double 16, float 9. Maximum: 20.

Example:

```bash
OZAKI=2 ./dgemm-wrap.x 256                        # use CRT scheme (u8 default)
```

## Complex GEMM (3M Method)

Intercepted ZGEMM (double-complex) and CGEMM (single-complex) calls are implemented via the Karatsuba 3M method. Each complex GEMM is decomposed into three real GEMM calls:

- P1 = Re(A) * Re(B)
- P2 = Im(A) * Im(B)
- P3 = (Re(A) + Im(A)) * (Re(B) + Im(B))

The real and imaginary parts of the product are recovered as Re(A*B) = P1 - P2 and Im(A*B) = P3 - P1 - P2. Complex alpha/beta scaling is applied in a final pass. The three real GEMM calls flow through the same wrapper, so they are accelerated by the Ozaki scheme as well. When built with LIBXSTREAM and GPU support is available, the 3M method can keep intermediates on-device to reduce transfers.

The 3M dispatch is controlled independently from the real GEMM scheme via `OZAKI_3M`:

- `OZAKI_3M=0` passes complex GEMM calls through to the original BLAS.
- `OZAKI_3M=1` uses the CPU-based 3M path (three real sub-GEMMs, each dispatched through the Ozaki real GEMM wrapper).
- `OZAKI_3M=2` uses the GPU-native 3M path (all intermediates on device), falling back to CPU 3M on failure.

By default, `OZAKI_3M` follows `OZAKI`: if `OZAKI=0`, complex GEMMs also pass through; otherwise GPU 3M is preferred (`OZAKI_3M=2`).

## Compile-Time Parameters

The block and batch sizes can be overridden at compile time via `-D`:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `BLOCK_M` | 16 | Tile rows (A and C). |
| `BLOCK_N` | 16 | Tile columns (B and C). |
| `BLOCK_K` | 16 | Tile depth: the K-dimension of each low-precision matrix multiply. Maps to a single SIMD register width for VNNI (128-bit at BLOCK_K=16, 256-bit at 32, 512-bit at 64). |
| `BATCH_K` | 4 | Number of BLOCK_K panels grouped into one batch. The effective K-step per batch is BATCH_K x BLOCK_K. Larger values reduce OpenMP barrier overhead and improve C-tile reuse at the cost of increased panel memory. |

Example:

```bash
make ECFLAGS="-DBLOCK_K=32 -DBATCH_K=2" dgemm-wrap.x
```

## Environment Variables

| Variable | Default | Description |
|----------|:-------:|-------------|
| `OZAKI` | 1 | Scheme selector for real GEMM: 0 = bypass (call original BLAS directly), 1 = Scheme 1 (mantissa slicing, int8), 2 = Scheme 2 (CRT). |
| `OZAKI_3M` | *auto* | Complex GEMM (ZGEMM/CGEMM) dispatch: 0 = pass through to original BLAS, 1 = CPU 3M (Karatsuba), 2 = GPU 3M (all on device, CPU fallback). Default: 0 if `OZAKI=0`, else 2. |
| `OZAKI_MAXK` | 32768 | Max K elements per preprocessing pass (K-group size). 0 = no grouping (full K in one pass). Smaller values narrow the exponent scope per group (better local precision, more FP accumulation steps). Larger values reduce accumulation rounding but widen the exponent scope. Applies to both CPU and GPU, all schemes. |
| `OZAKI_N` | *per scheme* | Number of decomposition units: slices for Scheme 1 (double: 1..16, default 8; float: 1..8, default 4) or moduli for Scheme 2 (double: 1..20, default 16; float: 1..12, default 9). |
| `OZAKI_I8` | 0 | Scheme 2 only: use signed i8 residues (moduli <= 128) instead of the default unsigned u8. Compile-time for CPU (`-DOZAKI_I8=1`), runtime for GPU. |
| `OZAKI_FLAGS` | 3 | Scheme 1 bitmask: Triangular (1), Symmetrize (2); see above. |
| `OZAKI_TRIM` | 0 | Precision levels to trim: 0 = exact. Scheme 1: drops T diagonals (~7 product bits/level). Scheme 2: truncates mantissa before CRT (~2 input bits/level, ~4 product bits/level). The level semantics are calibrated so the same `OZAKI_TRIM` value gives comparable accuracy across schemes. |
| `OZAKI_THRESHOLD` | 12 | Arithmetic intensity threshold. Ozaki is bypassed when flops/(bytes x threshold) < 1. Set to 0 to always apply Ozaki. Debug builds default to 0. |
| `OZAKI_VERBOSE` | 0 | 0 = silent; 1 = print accumulated statistics at exit; *N* = print every *N*th GEMM call. Auto-set to 1 when `OZAKI_EPS` or `OZAKI_RSQ` is set. |
| `OZAKI_STAT` | 0 | Track C-matrix (0), A-matrix representation (1), or B-matrix representation (2). |
| `OZAKI_EPS` | inf | Dump A/B matrices as MHD-files when the epsilon error exceeds the given threshold. |
| `OZAKI_RSQ` | 0 | Dump A/B matrices as MHD-files when RSQ drops below the given threshold; the threshold is updated after each dump. |
| `OZAKI_EXIT` | 1 | Exit with failure after dumping matrices on accuracy violation (eps/rsq threshold exceeded). Set to 0 to continue execution. |
| `CHECK` | 0 | Accuracy validation against BLAS reference: 0 = disabled, negative = auto-threshold (1e-10 for double, 1e-3 for float), positive = use value as threshold. |
| `NREPEAT` | 3 | Number of GEMM calls; when > 1 the first call is warmup and excluded from timing. |
| `OZAKI_OCL` | 1 | Enable (1) or disable (0) the OpenCL/GPU path at runtime. Only effective when built with LIBXSTREAM support. |
| `OZAKI_TM` | auto | GPU output tile height (multiple of 8). 0 = auto-select. |
| `OZAKI_TN` | auto | GPU output tile width (multiple of 16). 0 = auto-select. |
| `OZAKI_GROUPS` | 0 | Scheme 2 only: K-grouping factor (0/1 = disabled). When > 1, that many consecutive K sub-panels share one Garner reconstruction. |
| `OZAKI_PROFILE` | 0 | Profile mode: 0 = off, 1 or negative = all phases (preprocessing + kernel), 2 = kernel only, 3 = preprocess A, 4 = preprocess B. Modes 3 and 4 are equivalent on the CPU (preprocessing is combined). Reports GFLOPS/s and INT8-TOPS/s at exit. Works for both CPU and GPU paths. |

## Profiling

The `OZAKI_PROFILE` environment variable enables per-GEMM timing that is collected
into a histogram and reported at program exit. The histogram tracks effective
GFLOPS/s (based on `2*M*N*K / time`) and derives INT8-TOPS/s from the scheme's
decomposition multiplier (number of int8 GEMMs per FP GEMM).

```
OZAKI PROF: 850 DP-GFLOPS/s (17.0 INT8-TOPS/s, 20x)
```

The multiplier depends on the scheme and configuration:
- **Scheme 1**: Number of slice-pair int8 GEMMs, accounting for triangular
  iteration, symmetrize, and trim. E.g., 8 slices with default flags: 36x.
- **Scheme 2**: Number of primes. E.g., 19 primes: 19x.

Profile modes select which phase is measured:

| Mode | CPU | GPU |
|------|-----|-----|
| 1 / negative | All phases (preprocess + kernel) | All profiled kernels |
| 2 | Kernel only (int8 dot products + accumulation) | Dotprod/compute kernel only |
| 3 | Preprocessing (A+B combined) | Preprocess A kernel |
| 4 | Preprocessing (A+B combined) | Preprocess B kernel |

On the CPU, modes 3 and 4 are equivalent because A and B preprocessing is
interleaved across OpenMP threads. On the GPU, they select individual
preprocessing kernels running on separate streams.

Both CPU and GPU paths push into the same histogram, so in a mixed-path
scenario the reported median reflects whichever path handled more calls.

## Test Driver

The test driver (`gemm.c`) accepts positional arguments:

```text
dgemm-wrap.x  [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
sgemm-wrap.x  [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
dgemm-blas.x  [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
sgemm-blas.x  [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
zgemm-wrap.x  [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
cgemm-wrap.x  [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
```

Defaults: M=N=K=257, TA=TB=0, ALPHA=BETA=1.0.

TA and TB select transposition: 0 means 'N' (no transpose), non-zero means 'T' (transpose). The `dgemm-*` and `sgemm-*` drivers call DGEMM or SGEMM respectively. The `zgemm-wrap.x` and `cgemm-wrap.x` drivers call ZGEMM (double-complex) and CGEMM (single-complex) respectively; complex GEMM calls are implemented via the 3M method using three real GEMM calls each.

If the first argument is 0, the remaining arguments are treated as MHD filenames for the A and B matrices, allowing accuracy issues to be analyzed outside of an application.

## Source Layout

| File | Purpose |
|------|----------|
| `ozaki.h` | Shared header: block sizes, slice and prime constants, IEEE-754 decomposition helpers, flag definitions, and inline utility functions used by both schemes. |
| `gemm.h` | Common header: type macros, precision-specific name redirects, function prototypes for all four GEMM flavors. |
| `ozaki.c` | Wrapper/orchestration for real GEMM: initialization, environment handling, threshold check, fallback dispatch, and global state management. Compiled twice (double + float). |
| `ozaki1_int8.c` | Scheme 1 computational kernel: decomposes IEEE-754 mantissa into 7-bit int8 slices for low-precision dot products. Uses function-pointer dispatch for VNNI vs scalar int8 dot product. Compiled twice (double + float). |
| `ozaki2_int8.c` | Scheme 2 computational kernel: CRT-based modular arithmetic using small pairwise coprime moduli. Barrett reduction, Garner's algorithm with grouped Horner reconstruction. Compiled twice (double + float). |
| `wrap3m.c` | Complex GEMM 3M wrapper: deinterleaves complex matrices, issues 3 real GEMM calls (Karatsuba), recombines. Compiled twice (double + float). |
| `ozaki_ocl.c` | OpenCL bridge: wraps LIBXSTREAM behind an opaque handle. Compiled only when LIBXSTREAM is detected; isolates all OpenCL includes from the rest of the code. |
| `wrap.c` | Entry points and dlsym fallbacks for the LD_PRELOAD path. |
| `gemm.c` | Test driver. Compiled as `dgemm-{wrap,blas}.x` (double) and `sgemm-{wrap,blas}.x` (float). |
| `gemm-print.c` | `print_gemm` and `print_diff` utilities. |
| `test-wrap.sh` | Integration test: exercises STATIC WRAP, ORIGINAL BLAS, and LD_PRELOAD paths. |
| `test-check.sh` | Correctness test: runs both Ozaki schemes with `CHECK` validation. |
| `test-mhd.sh` | MHD file test: runs GEMM on all A/B MHD-file pairs in a directory. |
