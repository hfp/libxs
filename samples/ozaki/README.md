# Ozaki-Scheme Low-Precision GEMM

## Intercepted GEMM

This code sample intercepts all four standard BLAS GEMM routines — DGEMM, SGEMM, ZGEMM, and CGEMM — and only relies on the LAPACK/BLAS interface. Real GEMM calls (DGEMM/SGEMM) are executed via one of three Ozaki low-precision schemes (mantissa slicing, CRT, or BF16 Dekker split); complex GEMM calls (ZGEMM/CGEMM) are implemented via the 3M (Karatsuba) method using three real GEMM calls each. The wrapper sources are compiled twice (once for `double`, once for `float`), so all four symbols coexist in a single binary.

Two link-time variants are built per precision: (1)&#160;code which is dynamically linked against LAPACK/BLAS (`dgemm-blas.x`/`sgemm-blas.x`), (2)&#160;code which is linked using `--wrap=`*symbol* supported by GNU&#160;GCC compatible tool chains (`dgemm-wrap.x`/`sgemm-wrap.x`). Running `test-wrap.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux). The `test-check.sh` script validates all four Ozaki schemes for correctness, and `test-mhd.sh` tests MHD file-based GEMM input pairs.

The static wrapper library is built by default (`make`), and suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_`). To build and use the shared wrapper library:

```bash
cd /path/to/libxs
make -j $(nproc)

cd samples/ozaki
make BLAS_STATIC=0

LD_PRELOAD=/path/to/libwrap.so ./application
```

Note: LIBXS has to be built upfront for the sample code to link.

## Performance

The Ozaki sample code demonstrates how to intercept GEMMs (any BLAS library), to run a low-precision GEMM instead of the original GEMM, to compare the results, and to maintain running statistics presented every *N*th call of GEMM or when the application terminates. The code explores high-precision emulation using low-precision calculation.

The blocking structure is designed to conceptually emulate fixed-size matrix-multiply hardware. All computation operates on tiles of size `BLOCK_M`&#160;×&#160;`BLOCK_K` (A) and `BLOCK_K`&#160;×&#160;`BLOCK_N` (B), with defaults BLOCK_M&#160;=&#160;BLOCK_N&#160;=&#160;BLOCK_K&#160;=&#160;16. The compile-time parameter `BATCH_K` (default&#160;4) groups `BATCH_K` consecutive BLOCK_K panels into a single batch, so the effective K-dimension step per batch is BATCH_K&#160;×&#160;BLOCK_K. Batching reduces OpenMP barrier overhead and improves temporal reuse of the C&#160;tile, while keeping the fundamental tile size visible throughout the code.

Preprocessing (exponent alignment, mantissa slicing or modular reduction) accounts for roughly 5% of runtime; the remaining 95% is spent in the inner dot-product loops. In Scheme&#160;1, int8 dot products are dispatched once per GEMM via a function pointer: AVX-512 VNNI (`VPDPBUSD`) when available, otherwise a scalar fallback. Scheme&#160;3 uses BF16 Dekker splitting (no exponent alignment needed) and BF16 dot products (`VDPBF16PS`) instead of int8. The number of pairwise slice products is quadratic in the number of slices, so for double-precision (8&#160;slices, default) the inner loop performs up to 36&#160;dot products per block pair. Scheme&#160;2 performs one modular dot product per prime, so its cost is linear in the number of primes.

OpenMP parallelizes all three phases of each K-batch: Phase&#160;1 preprocesses A&#160;panels (`schedule(dynamic) nowait`), Phase&#160;2 preprocesses B&#160;panels (`schedule(dynamic)`, implicit barrier), and Phase&#160;3 accumulates the dot products into C (`collapse(2) schedule(static)`). Panel buffers are shared across the parallel region and sized by `BATCH_K`&#160;×&#160;number-of-blocks, allocated via `libxs_malloc`.

Practically all CPUs provide higher instruction throughput using floating point instructions even when relying on double-precision. The algorithmic complexity and inner-most code is in fact unsuitable to reach high performance levels. OpenMP based parallelization or VNNI instructions are only meant to improve emulating high-precision.

When built with [LIBXSTREAM](https://github.com/hfp/libxstream) support (sibling `libxstream` repository detected at build time), an optional OpenCL/GPU path is available for Ozaki-1 and Ozaki-2 as well (see `OZAKI` environment variable). The GPU bridge (`ozaki_ocl.c`) wraps LIBXSTREAM behind an opaque handle so that the CPU code has no OpenCL dependency. At runtime, the GPU path is enabled by default (`OZAKI_OCL=1`) and can be disabled with `OZAKI_OCL=0` to fall back to the CPU implementation. The CPU-only code performs the low-precision conversion on-the-fly and only requires a reasonable stack size to buffer small matrix blocks.

## Scheme 1 — Mantissa Slicing

Scheme 1 (`OZAKI=1`, the default) decomposes each IEEE-754 mantissa into 7-bit int8 slices and accumulates all pairwise slice products via low-precision GEMM. The number of slices aka "splits" determines achievable accuracy and can be set at runtime via `OZAKI_N`. The default and maximum vary by precision (double: default 8, max 16; float: default 4, max 8). The size of the matrices employed by potential "matrix cores" is set at compile-time with `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`; grouping of consecutive K-panels is controlled by `BATCH_K`. The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

## Complex GEMM (3M Method)

Intercepted ZGEMM (double-complex) and CGEMM (single-complex) calls are implemented via the Karatsuba 3M method. Each complex GEMM is decomposed into three real GEMM calls:

- P1 = Re(A) · Re(B)
- P2 = Im(A) · Im(B)
- P3 = (Re(A) + Im(A)) · (Re(B) + Im(B))

The real and imaginary parts of the product are recovered as Re(A·B) = P1 − P2 and Im(A·B) = P3 − P1 − P2. Complex alpha/beta scaling is applied in a final pass. The three real GEMM calls flow through the same wrapper, so they are optionally accelerated by the Ozaki scheme as well.

## Scheme 1 Flags (`OZAKI_FLAGS`) and Diagonal Trim (`OZAKI_TRIM`)

The Scheme&#160;1 loop is controlled by two runtime knobs:

### Flags

The environment variable `OZAKI_FLAGS` is an integer bitmask:

| Bit | Value | Flag | Description |
|:---:|:-----:|------|-------------|
| 0 | 1 | Triangular | Only iterate slice pairs (sa,&#160;sb) with sb&#160;>=&#160;sa (upper triangle). |
| 1 | 2 | Symmetrize | For each off-diagonal pair (sa,&#160;sb) in the upper triangle, also compute the mirror dot product D(sb,&#160;sa) at negligible extra cost (one additional int8 dot product per pair). |

The default value is **3** (both flags). Setting `OZAKI_FLAGS=0` runs the full S^2 square of slice pairs.

### Diagonal Trim

The environment variable `OZAKI_TRIM` drops the T least significant diagonals from the slice-pair iteration. Pairs with sa&#160;+&#160;sb&#160;>&#160;2*(S-1)&#160;-&#160;T are skipped. Pair significance scales as 2^(low_bit[sa]&#160;+&#160;low_bit[sb]), so sa&#160;+&#160;sb directly determines significance — smaller sums are more significant. Each dropped diagonal loses approximately 7&#160;bits of relative precision.

The default value is **0** (exact: all pairs). The value is clamped so that at least diagonal&#160;0 is always computed.

**Cost overview** for *S*&#160;slices with TRIANGULAR&#160;+&#160;SYMMETRIZE (default flags):

| `OZAKI_TRIM` | Dot products | Pairs covered | Relative bits lost |
|:---:|:---:|:---:|:---:|
| 0 (default) | S*(S+1)/2 | all S^2 | 0 (exact) |
| S-1 | ~S^2/4 | (S+1)*S/2 | ~7*(S-1) least significant |
| 3S/2 | ~S^2/8 | ~S^2/4 | ~7*3S/2 least significant |
| 2*(S-1) | 1 | 1 (only diagonal) | ~7*2*(S-1) |

Because SYMMETRIZE computes both D(sa,sb) and D(sb,sa), the number of dot products with TRIANGULAR equals S*(S+1)/2 for cutoff&#160;=&#160;2*(S-1) but covers all S^2 contributions.

Examples:

```bash
./dgemm-wrap.x 256                      # exact (default: flags=3, trim=0)
OZAKI_TRIM=4 ./dgemm-wrap.x 256        # drop 4 least significant diagonals
OZAKI_FLAGS=0 ./dgemm-wrap.x 256       # full S^2 square, no symmetrize
```

## Scheme 2 — Chinese Remainder Theorem

Scheme 2 (`OZAKI=2`) uses modular arithmetic instead of mantissa slicing. Each matrix element is reduced modulo a set of small pairwise coprime moduli (primes and prime powers ≤ 128) so that residues fit in int8 and dot products use VNNI int8 instructions when available. GEMM is performed independently modulo each modulus, and the exact integer result is recovered via the Chinese Remainder Theorem (Garner's algorithm with grouped uint64 Horner evaluation). The Horner reconstruction partitions mixed-radix digits into groups of up to 9, evaluates each group exactly in uint64 arithmetic, and combines groups with a minimal number of FP64 operations — avoiding double-precision throughput bottlenecks on hardware where integer arithmetic is faster. Because the work is linear in the number of moduli — versus quadratic in the number of slices for Scheme 1 — Scheme 2 can be more efficient when many moduli/slices are needed.

Residues are signed int8 (−127..+127) with the element sign folded directly into the residue. This maps naturally to VNNI's VPDPBUSD encoding (unsigned × signed with bias correction). An unsigned variant (uint8 residues, moduli ≤ 256) is theoretically possible but would require a separate sign array and scalar dot-product accumulation, negating the VNNI advantage.

The number of moduli can be set at runtime via `OZAKI_N`. The default and maximum are: double: 17 / 18; float: 8 / 10.

Example:

```bash
OZAKI=2 ./dgemm-wrap.x 256                        # use CRT scheme
```

## Scheme 3 — BF16 Dekker Split

Scheme 3 (`OZAKI=3`) decomposes each matrix element into a sequence of BF16 slices using a Dekker-style error-free split: `slice[0] = round_bf16(x); residual = x - slice[0]; slice[1] = round_bf16(residual); …` Each BF16 value carries its own 8-bit exponent, so no shared per-row or per-column exponent is needed — unlike Schemes&#160;1 and&#160;2, there is no exponent alignment, mantissa extraction, or `pow2` reconstruction. Preprocessing simply splits and scatters; accumulation is `C[i,j] += alpha * dot_bf16(A_sa, B_sb)` with no scale factors.

Each BF16 slice captures roughly 8&#160;significant bits. For double precision (52&#160;mantissa bits), ~7 slices suffice (8&#215;7&#160;=&#160;56&#160;>&#160;52); for single precision (23&#160;bits), ~3 slices suffice. The dot products are computed via `VDPBF16PS` (AVX-512-BF16) when available, otherwise a scalar fallback. FP32 accumulation of BF16&#215;BF16 products is **not** exact in general (unlike the integer-exact Scheme&#160;1), so Scheme&#160;3 is inherently approximate — the error depends on the number of slices and the FP32 rounding during accumulation. In practice the relative error is very small (typically &lt;&#160;1e-9 for double precision with default settings).

Scheme&#160;3 shares the same runtime knobs as Scheme&#160;1: `OZAKI_N` (number of slices), `OZAKI_FLAGS` (triangular + symmetrize), and `OZAKI_TRIM` (diagonal trim). Because no per-row/per-column exponent panels are needed, the metadata overhead is halved relative to Scheme&#160;1.

Example:

```bash
OZAKI=3 ./dgemm-wrap.x 256                        # use BF16 scheme
```

## Compile-Time Parameters

The block and batch sizes can be overridden at compile time via `-D`:

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `BLOCK_M` | 16 | Tile rows (A and C). |
| `BLOCK_N` | 16 | Tile columns (B and C). |
| `BLOCK_K` | 16 | Tile depth: the K-dimension of each low-precision matrix multiply. Maps to a single SIMD register width for VNNI (128-bit at BLOCK_K=16, 256-bit at 32, 512-bit at 64) and similarly for BF16 (256-bit at BLOCK_K=16, 512-bit at 32, 2&#215;512-bit at 64). |
| `BATCH_K` | 4 | Number of BLOCK_K panels grouped into one batch. The effective K-step per batch is BATCH_K&#160;×&#160;BLOCK_K. Larger values reduce barrier overhead and improve C-tile reuse at the cost of increased panel memory. |

Example:

```bash
make ECFLAGS="-DBLOCK_K=32 -DBATCH_K=2" dgemm-wrap.x
```

## Environment Variables

| Variable | Default | Description |
|----------|:-------:|-------------|
| `OZAKI` | 1 | Scheme selector: 0 = bypass (call original BLAS directly), 1 = Scheme 1 (mantissa slicing, int8), 2 = Scheme 2 (CRT), 3 = Scheme 3 (BF16 Dekker split), 4 = Scheme 4 (CRT + BF16, no OpenCL). |
| `OZAKI_N` | *per scheme* | Number of decomposition units: slices for Scheme 1 (double: 1..16, default 8; float: 1..8, default 4) or moduli for Scheme 2 (see Scheme 2 section for per-precision defaults). |
| `OZAKI_OCL` | 1 | Enable (1) or disable (0) the OpenCL/GPU path at runtime. Only effective when built with LIBXSTREAM support (`__LIBXSTREAM`). When disabled, falls back to the CPU Ozaki scheme. |
| `OZAKI_FLAGS` | 3 | Scheme 1 bitmask: Triangular (1), Symmetrize (2); see above. |
| `OZAKI_TRIM` | 0 | Scheme 1 diagonal trim: 0 = exact, T = drop T least significant diagonals (~7 bits each). |
| `OZAKI_EPS` | inf | Dump A/B matrices as MHD-files when the epsilon error exceeds the given threshold (implies `OZAKI_VERBOSE=1` if unset). |
| `OZAKI_VERBOSE` | 0 | 0&#160;=&#160;silent; 1&#160;=&#160;print accumulated statistic at exit; *N*&#160;=&#160;print every *N*th GEMM call. |
| `OZAKI_STAT` | 0 | Track C-matrix (0), A-matrix representation (1), or B-matrix representation (2). |
| `OZAKI_EXIT` | 1 | Exit with failure after dumping matrices on accuracy violation (eps/rsq threshold exceeded). Set to 0 to continue execution. |
| `OZAKI_RSQ` | 0 | Dump A/B matrices as MHD-files when RSQ drops below the given threshold; the threshold is updated after each dump (implies `OZAKI_VERBOSE=1` if unset). |
| `CHECK` | 0 | Accuracy validation against BLAS reference: 0&#160;=&#160;disabled, negative&#160;=&#160;auto-threshold (1e-10 for double, 1e-3 for float), positive&#160;=&#160;use value as threshold. Prints `CHECK: eps=…` to stderr and exits with failure if exceeded. |
| `NREPEAT` | 1 | Number of GEMM calls; when >&#160;1 the first call is warmup and excluded from timing. |

## Test Driver

The test driver (`gemm.c`) accepts positional arguments:

```text
dgemm-wrap.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
sgemm-wrap.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
dgemm-blas.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
sgemm-blas.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
```

TA and TB select transposition: 0&#160;means&#160;'N' (no transpose), non-zero means&#160;'T' (transpose). The `dgemm-*` and `sgemm-*` drivers call DGEMM or SGEMM respectively, built with the matching `GEMM_REAL_TYPE` (`double` or `float`). The `GEMM_INT_TYPE` (default `int`) can also be overridden at build time.

## Source Layout

| File | Purpose |
|------|----------|
| `ozaki.h` | Shared header: block sizes (`BLOCK_M`/`BLOCK_N`/`BLOCK_K`/`BATCH_K`), slice and prime constants, IEEE-754 decomposition helpers, flag definitions (`OZ1_TRIANGULAR`, `OZ1_SYMMETRIZE`), and inline utility functions used by both schemes. |
| `gemm.h` | Common header: type macros (`GEMM_ARGDECL`/`GEMM_ARGPASS`), precision-specific name redirects, function prototypes for all four GEMM flavors. |
| `ozaki.c` | Wrapper/orchestration for real GEMM (`GEMM_WRAP`): initialization, environment handling, fallback dispatch, and global state management. Compiled twice (double + float). |
| `ozaki1_int8.c` | Ozaki Scheme-1 computational kernel (`gemm_oz1`): decomposes IEEE-754 mantissa into 7-bit int8 slices for low-precision dot products. Uses function-pointer dispatch for VNNI vs scalar int8 dot product. Compiled twice (double + float). |
| `ozaki2_int8.c` | Ozaki Scheme-2 computational kernel (`gemm_oz2`): CRT-based modular arithmetic using small pairwise coprime moduli. Barrett reduction for fast modular arithmetic, Garner's algorithm with batched reconstruction. Residues fit in int8 (sign folded in) and dot products use VNNI. Compiled twice (double + float). |
| `ozaki1_bf16.c` | Ozaki Scheme-3 computational kernel (`gemm_oz3`): Dekker-style error-free split into BF16 slices — each carries its own exponent, so no shared-exponent alignment is needed. Dot products via `VDPBF16PS` (scalar fallback). Inherently approximate (FP32 accumulation rounding). Compiled twice (double + float). |
| `ozaki2_bf16.c` | Ozaki Scheme-4 computational kernel (`gemm_oz4`): CRT-based modular arithmetic (same decomposition as Scheme 2) with BF16 dot products via `VDPBF16PS` instead of VNNI int8. Residues (0–127) are exactly representable in BF16; FP32 accumulation is exact for BLOCK_K ≤ 64. Select with `OZAKI=4`. Compiled twice (double + float). |
| `zgemm3m.c` | Complex GEMM 3M wrapper (`ZGEMM_WRAP`): deinterleaves complex matrices, issues 3 real GEMM calls (Karatsuba), recombines. Uses `libxs_malloc` for workspace. Compiled twice (double + float). |
| `ozaki_ocl.c` | OpenCL bridge: wraps LIBXSTREAM behind an opaque handle (`ozaki_ocl_create`/`release`/`dgemm`/`finalize`). Compiled only when LIBXSTREAM is detected; isolates all OpenCL includes from the rest of the code. |
| `wrap.c` | Entry points (`GEMM`, `ZGEMM`) and dlsym fallbacks (`GEMM_REAL`, `ZGEMM_REAL`) via `GEMM_DEFINE_DLSYM` macro. Used only in the LD_PRELOAD path; excluded from the static archive to keep `__real_` resolution correct. |
| `gemm.c` | Test driver. Compiled as `dgemm-{wrap,blas}.x` (double) and `sgemm-{wrap,blas}.x` (float). |
| `gemm-print.c` | `print_gemm` and `print_diff` utilities. |
| `test-wrap.sh` | Integration test: exercises STATIC WRAP, ORIGINAL BLAS, and LD_PRELOAD paths. Auto-discovers built `*-wrap.x` / `*-blas.x` executables; accepts optional test-name prefix as first argument. |
| `test-check.sh` | Correctness test: runs all four Ozaki schemes with `CHECK` validation. Auto-discovers built `*gemm-wrap.x` executables; accepts optional test-name prefix. |
| `test-mhd.sh` | MHD file test: runs GEMM on all A/B MHD-file pairs in a directory. Accepts optional test-name prefix and directory arguments. |

If the driver is called with MHD-files, accuracy issues can be analyzed outside of an application.
