# Ozaki Scheme-1 based Low-Precision GEMM

## Intercepted GEMM

This code sample intercepts all four standard BLAS GEMM routines — DGEMM, SGEMM, ZGEMM, and CGEMM — and only relies on the LAPACK/BLAS interface. Real GEMM calls (DGEMM/SGEMM) are executed via the Ozaki low-precision scheme; complex GEMM calls (ZGEMM/CGEMM) are implemented via the 3M (Karatsuba) method using three real GEMM calls each. The wrapper sources are compiled twice (once for `double`, once for `float`), so all four symbols coexist in a single binary.

Two link-time variants are built: (1)&#160;code which is dynamically linked against LAPACK/BLAS (`gemm-blas.x`), (2)&#160;code which is linked using `--wrap=`*symbol* supported by GNU&#160;GCC compatible tool chains (`gemm-wrap.x`). Running `wrap-test.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux).

The static wrapper library is built by default (`make`), and suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_`). To build and use the shared wrapper library:

```bash
make BLAS_STATIC=0

LD_PRELOAD=/path/to/libwrap.so ./application
```

Note: LIBXS has to be built upfront for the sample code to link.

## Low-Precision GEMM

The wrapper library intercepts an application's calls to DGEMM/SGEMM and executes Ozaki Scheme-1 to emulate higher accuracy based on low-precision matrix multiplication. The number of slices aka "splits" determines achievable accuracy and can be set at runtime via `GEMM_OZ1N`. The default and maximum vary by precision (double: default&#160;8, max&#160;16; float: default&#160;4, max&#160;8). The size of the matrices employed by potential "matrix cores" is set at compile-time with `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`. The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

## Complex GEMM (3M Method)

Intercepted ZGEMM (double-complex) and CGEMM (single-complex) calls are implemented via the Karatsuba 3M method. Each complex GEMM is decomposed into three real GEMM calls:

- P1 = Re(A) · Re(B)
- P2 = Im(A) · Im(B)
- P3 = (Re(A) + Im(A)) · (Re(B) + Im(B))

The real and imaginary parts of the product are recovered as Re(A·B) = P1 − P2 and Im(A·B) = P3 − P1 − P2. Complex alpha/beta scaling is applied in a final pass. The three real GEMM calls flow through the same wrapper, so they are optionally accelerated by the Ozaki scheme as well.

## Scheme Flags (`GEMM_OZ1`)

The multiplication scheme is controlled at runtime by the environment variable `GEMM_OZ1`, an integer interpreted as a bitmask of four orthogonal flags:

| Bit | Value | Flag | Description |
|:---:|:-----:|------|-------------|
| 0 | 1 | Triangular | Drop symmetric contributions, only compute the upper triangle of slice pairs (speed for accuracy). |
| 1 | 2 | Symmetrize | Double off-diagonal upper-triangle terms to approximate the dropped lower-triangle contributions (zero extra cost). |
| 2 | 4 | Reverse&#160;pass | Explicitly recover the most significant lower-triangle terms (slice&#160;a&#160;>=&#160;S/2) at ~S^2/4 additional cost. |
| 3 | 8 | Trim&#160;forward | Limit the forward pass to slice&#160;a&#160;<&#160;S/2 so that, combined with the reverse pass, total cost equals the original triangular (S^2/2) but with better coverage. |

The default value is **15** (all flags enabled). Setting `GEMM_OZ1=0` runs the full square of slice pairs.

**Cost overview** for *S* slices:

| `GEMM_OZ1` | Flags | Forward | Reverse | Total | Notes |
|:---:|---|:---:|:---:|:---:|---|
| 1 | Triangular | S^2/2 | 0 | S^2/2 | Upper triangle only |
| 3 | +&#160;Symmetrize | S^2/2 | 0 | S^2/2 | +&#160;doubling approximation |
| 7 | +&#160;Reverse&#160;pass | S^2/2 | S^2/4 | 3S^2/4 | Upper&#160;+ lower significant |
| **15** | +&#160;Trim&#160;forward | S^2/4 | S^2/4 | **S^2/2** | Symmetric coverage at original cost **(default)** |
| 0 | *(none)* | S^2 | 0 | S^2 | All pairs, full square |

Example:

```bash
GEMM_OZ1=3 ./gemm-wrap.x 256    # triangular + symmetrize, no reverse pass
```

## Environment Variables

| Variable | Default | Description |
|----------|:-------:|-------------|
| `GEMM_OZ1N` | 8 (double), 4 (float) | Number of slices (double: 1..16, float: 1..8); sensible range is 5..11 for double, 2..6 for float. |
| `GEMM_OZAKI` | 1 | Set to 0 to bypass LP-GEMM and the 3M complex wrapper, calling the original BLAS directly (DGEMM/SGEMM/ZGEMM/CGEMM). |
| `GEMM_EPS` | inf | Dump A/B matrices as MHD-files when the epsilon error exceeds the given threshold (implies `GEMM_VERBOSE=1` if unset). |
| `GEMM_VERBOSE` | 0 | 0&#160;=&#160;silent; 1&#160;=&#160;print accumulated statistic at exit; *N*&#160;=&#160;print every *N*th GEMM call. |
| `GEMM_DIFF` | 0 | Track C-matrix (0), A-matrix representation (1), or B-matrix representation (2). |
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
| `ozaki.c` | Ozaki Scheme-1 implementation (`GEMM_WRAP`): decomposes IEEE-754 mantissa into 7-bit int8 slices for low-precision dot products. Compiled twice (double + float). |
| `zgemm3m.c` | Complex GEMM 3M wrapper (`ZGEMM_WRAP`): deinterleaves complex matrices, issues 3 real GEMM calls (Karatsuba), recombines. Uses `libxs_malloc` for workspace. Compiled twice (double + float). |
| `wrap.c` | Entry points (`GEMM`, `ZGEMM`) and dlsym fallbacks (`GEMM_REAL`, `ZGEMM_REAL`) via `GEMM_DEFINE_DLSYM` macro. Used only in the LD_PRELOAD path; excluded from the static archive to keep `__real_` resolution correct. |
| `gemm.c` | Test driver. |
| `gemm-print.c` | `print_gemm` and `print_diff` utilities. |

If the driver is called with MHD-files, accuracy issues can be analyzed outside of an application.

