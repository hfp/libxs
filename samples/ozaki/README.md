# Ozaki-Scheme Low-Precision GEMM

## Intercepted GEMM

This code sample intercepts all four standard BLAS GEMM routines — DGEMM, SGEMM, ZGEMM, and CGEMM — and only relies on the LAPACK/BLAS interface. Real GEMM calls (DGEMM/SGEMM) are executed via one of two Ozaki low-precision schemes (mantissa slicing or CRT); complex GEMM calls (ZGEMM/CGEMM) are implemented via the 3M (Karatsuba) method using three real GEMM calls each. The wrapper sources are compiled twice (once for `double`, once for `float`), so all four symbols coexist in a single binary.

Two link-time variants are built: (1)&#160;code which is dynamically linked against LAPACK/BLAS (`gemm-blas.x`), (2)&#160;code which is linked using `--wrap=`*symbol* supported by GNU&#160;GCC compatible tool chains (`gemm-wrap.x`). Running `wrap-test.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux).

The static wrapper library is built by default (`make`), and suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_`). To build and use the shared wrapper library:

```bash
make BLAS_STATIC=0

LD_PRELOAD=/path/to/libwrap.so ./application
```

Note: LIBXS has to be built upfront for the sample code to link.

## Scheme 1 — Mantissa Slicing

Scheme 1 (`GEMM_OZAKI=1`, the default) decomposes each IEEE-754 mantissa into 7-bit int8 slices and accumulates all pairwise slice products via low-precision GEMM. The number of slices aka "splits" determines achievable accuracy and can be set at runtime via `GEMM_OZN`. The default and maximum vary by precision (double: default 8, max 16; float: default 4, max 8). The size of the matrices employed by potential "matrix cores" is set at compile-time with `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`. The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

## Complex GEMM (3M Method)

Intercepted ZGEMM (double-complex) and CGEMM (single-complex) calls are implemented via the Karatsuba 3M method. Each complex GEMM is decomposed into three real GEMM calls:

- P1 = Re(A) · Re(B)
- P2 = Im(A) · Im(B)
- P3 = (Re(A) + Im(A)) · (Re(B) + Im(B))

The real and imaginary parts of the product are recovered as Re(A·B) = P1 − P2 and Im(A·B) = P3 − P1 − P2. Complex alpha/beta scaling is applied in a final pass. The three real GEMM calls flow through the same wrapper, so they are optionally accelerated by the Ozaki scheme as well.

## Scheme 1 Flags (`GEMM_OZFLAGS`) and Diagonal Trim (`GEMM_OZTRIM`)

The Scheme&#160;1 loop is controlled by two runtime knobs:

### Flags

The environment variable `GEMM_OZFLAGS` is an integer bitmask:

| Bit | Value | Flag | Description |
|:---:|:-----:|------|-------------|
| 0 | 1 | Triangular | Only iterate slice pairs (sa,&#160;sb) with sb&#160;>=&#160;sa (upper triangle). |
| 1 | 2 | Symmetrize | For each off-diagonal pair (sa,&#160;sb) in the upper triangle, also compute the mirror dot product D(sb,&#160;sa) at negligible extra cost (one additional int8 dot product per pair). |

The default value is **3** (both flags). Setting `GEMM_OZFLAGS=0` runs the full S^2 square of slice pairs.

### Diagonal Trim

The environment variable `GEMM_OZTRIM` drops the T least significant diagonals from the slice-pair iteration. Pairs with sa&#160;+&#160;sb&#160;>&#160;2*(S-1)&#160;-&#160;T are skipped. Pair significance scales as 2^(low_bit[sa]&#160;+&#160;low_bit[sb]), so sa&#160;+&#160;sb directly determines significance — smaller sums are more significant. Each dropped diagonal loses approximately 7&#160;bits of relative precision.

The default value is **0** (exact: all pairs). The value is clamped so that at least diagonal&#160;0 is always computed.

**Cost overview** for *S*&#160;slices with TRIANGULAR&#160;+&#160;SYMMETRIZE (default flags):

| `GEMM_OZTRIM` | Dot products | Pairs covered | Relative bits lost |
|:---:|:---:|:---:|:---:|
| 0 (default) | S*(S+1)/2 | all S^2 | 0 (exact) |
| S-1 | ~S^2/4 | (S+1)*S/2 | ~7*(S-1) least significant |
| 3S/2 | ~S^2/8 | ~S^2/4 | ~7*3S/2 least significant |
| 2*(S-1) | 1 | 1 (only diagonal) | ~7*2*(S-1) |

Because SYMMETRIZE computes both D(sa,sb) and D(sb,sa), the number of dot products with TRIANGULAR equals S*(S+1)/2 for cutoff&#160;=&#160;2*(S-1) but covers all S^2 contributions.

Examples:

```bash
./gemm-wrap.x 256                      # exact (default: flags=3, trim=0)
GEMM_OZTRIM=4 ./gemm-wrap.x 256        # drop 4 least significant diagonals
GEMM_OZFLAGS=0 ./gemm-wrap.x 256       # full S^2 square, no symmetrize
```

## Scheme 2 — Chinese Remainder Theorem

Scheme 2 (`GEMM_OZAKI=2`) uses modular arithmetic instead of mantissa slicing. Each matrix element is reduced modulo a set of small primes (all < 256) so that residues fit in uint8 and pairwise products fit in uint32. GEMM is performed independently modulo each prime, and the exact integer result is recovered via the Chinese Remainder Theorem (Garner's algorithm). Because the work is linear in the number of primes — versus quadratic in the number of slices for Scheme 1 — Scheme 2 can be more efficient when many primes/slices are needed.

The number of primes can be set at runtime via `GEMM_OZN`. The default and maximum vary by precision (double: default 15, max 16; float: default 7, max 10). The product of all selected primes must exceed the maximum possible dot-product magnitude to avoid aliasing.

Example:

```bash
GEMM_OZAKI=2 ./gemm-wrap.x 256    # use CRT scheme
```

## Environment Variables

| Variable | Default | Description |
|----------|:-------:|-------------|
| `GEMM_OZAKI` | 1 | Scheme selector: 0 = bypass (call original BLAS directly), 1 = Scheme 1 (mantissa slicing), 2 = Scheme 2 (CRT). |
| `GEMM_OZN` | *per scheme* | Number of decomposition units: slices for Scheme 1 (double: 1..16, default 8; float: 1..8, default 4) or primes for Scheme 2 (double: 1..16, default 15; float: 1..10, default 7). |
| `GEMM_OZFLAGS` | 3 | Scheme 1 bitmask: Triangular (1), Symmetrize (2); see above. |
| `GEMM_OZTRIM` | 0 | Scheme 1 diagonal trim: 0 = exact, T = drop T least significant diagonals (~7 bits each). |
| `GEMM_EPS` | inf | Dump A/B matrices as MHD-files when the epsilon error exceeds the given threshold (implies `GEMM_VERBOSE=1` if unset). |
| `GEMM_VERBOSE` | 0 | 0&#160;=&#160;silent; 1&#160;=&#160;print accumulated statistic at exit; *N*&#160;=&#160;print every *N*th GEMM call. |
| `GEMM_STAT` | 0 | Track C-matrix (0), A-matrix representation (1), or B-matrix representation (2). |
| `GEMM_EXIT` | 1 | Exit with failure after dumping matrices on accuracy violation (eps/rsq threshold exceeded). Set to 0 to continue execution. |
| `GEMM_RSQ` | 0 | Dump A/B matrices as MHD-files when RSQ drops below the given threshold; the threshold is updated after each dump (implies `GEMM_VERBOSE=1` if unset). |
| `NREPEAT` | 1 | Number of GEMM calls; when >&#160;1 the first call is warmup and excluded from timing. |

## Test Driver

The test driver (`gemm.c`) accepts positional arguments:

```text
gemm-wrap.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
gemm-blas.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
```

TA and TB select transposition: 0&#160;means&#160;'N' (no transpose), non-zero means&#160;'T' (transpose). The driver calls DGEMM by default; precision and integer width can be selected at build time via `GEMM_REAL_TYPE` (default `double`) and `GEMM_INT_TYPE` (default `int`).

## Source Layout

| File | Purpose |
|------|----------|
| `gemm.h` | Common header: type macros (`GEMM_ARGDECL`/`GEMM_ARGPASS`), precision-specific name redirects, function prototypes for all four GEMM flavors. |
| `ozaki.c` | Wrapper/orchestration for real GEMM (`GEMM_WRAP`): initialization, environment handling, fallback dispatch, and global state management. Compiled twice (double + float). |
| `ozaki1.c` | Ozaki Scheme-1 computational kernel (`gemm_oz1`): decomposes IEEE-754 mantissa into 7-bit int8 slices for low-precision dot products. Compiled twice (double + float). |
| `ozaki2.c` | Ozaki Scheme-2 computational kernel (`gemm_oz2`): CRT-based modular arithmetic using small primes (< 256). Linear in number of primes vs quadratic slices. Compiled twice (double + float). |
| `zgemm3m.c` | Complex GEMM 3M wrapper (`ZGEMM_WRAP`): deinterleaves complex matrices, issues 3 real GEMM calls (Karatsuba), recombines. Uses `libxs_malloc` for workspace. Compiled twice (double + float). |
| `wrap.c` | Entry points (`GEMM`, `ZGEMM`) and dlsym fallbacks (`GEMM_REAL`, `ZGEMM_REAL`) via `GEMM_DEFINE_DLSYM` macro. Used only in the LD_PRELOAD path; excluded from the static archive to keep `__real_` resolution correct. |
| `gemm.c` | Test driver. |
| `gemm-print.c` | `print_gemm` and `print_diff` utilities. |

If the driver is called with MHD-files, accuracy issues can be analyzed outside of an application.

