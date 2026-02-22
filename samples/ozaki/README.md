# Ozaki Scheme-1 based Low-Precision GEMM

## Intercepted GEMM

This code sample is calling DGEMM (with `-DGEMM_REAL_TYPE=float`, the SGEMM does not implement the Ozaki scheme) and only relies on LAPACK/BLAS interface. Two variants can be linked when building the source code: (1)&#160;code which is dynamically linked against LAPACK/BLAS (`gemm-blas.x`), (2)&#160;code which is linked using `--wrap=`*symbol* supported by GNU&#160;GCC compatible tool chains (`gemm-wrap.x`). Running `wrap-test.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux).

The static wrapper library is built by default (`make`), and suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_`). To build and use the shared wrapper library:

```bash
make BLAS_STATIC=0

LD_PRELOAD=/path/to/libwrap.so ./application
```

## Low-Precision GEMM

The wrapper library intercepts an application's calls to DGEMM, and executes Ozaki Scheme-1 to emulate FP64 accuracy based on low-precision matrix multiplication. The number of slices aka "splits" determines achievable accuracy and can be set at runtime via `GEMM_OZ1N` (default&#160;8, maximum&#160;16). The size of the matrices employed by potential "matrix cores" is set at compile-time with `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`. The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

### Scheme Flags (`GEMM_OZ1`)

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

### Other Environment Variables

| Variable | Default | Description |
|----------|:-------:|-------------|
| `GEMM_OZ1N` | 8 | Number of slices (1..16); for double-precision 5..11 is sensible, up to 16 for single-precision. |
| `GEMM_OZAKI` | 1 | Set to 0 to bypass LP-GEMM and call the original DGEMM directly. |
| `GEMM_VERBOSE` | 0 | 0&#160;=&#160;silent; 1&#160;=&#160;print accumulated statistic at exit; *N*&#160;=&#160;print every *N*th DGEMM call. |
| `GEMM_DIFF` | 0 | Track C-matrix (0), A-matrix representation (1), or B-matrix representation (2). |
| `GEMM_RSQ` | 0 | Dump A/B matrices as MHD-files when RSQ drops below the given threshold; the threshold is updated after each dump. |
| `NREPEAT` | 1 | Number of GEMM calls; when >&#160;1 the first call is warmup and excluded from timing. |

### Test Driver

The test driver (`gemm.c`) accepts positional arguments:

```text
gemm-wrap.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
gemm-blas.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
```

TA and TB select transposition: 0&#160;means&#160;'N' (no transpose), non-zero means&#160;'T' (transpose). The code can be built with `GEMM_REAL_TYPE` (default `double`) and `GEMM_INT_TYPE` (default `int`) to select precision and integer width.

If the driver is called with MHD-files, accuracy issues can be analyzed outside of an application.

