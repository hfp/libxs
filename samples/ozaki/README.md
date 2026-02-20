# Ozaki Scheme-1 based Low-Precision GEMM

## Intercepted GEMM

This code sample is calling DGEMM (with `-DGEMM_REAL_TYPE=float`, the SGEMM does not implement the Ozaki scheme) and only relies on LAPACK/BLAS interface. Two variants can be linked when building the source code: (1) code which is dynamically linked against LAPACK/BLAS (`gemm-blas.x`), (2) code which is linked using `--wrap=`*symbol* supported by GNU&#160;GCC compatible tool chains (`gemm-wrap.x`). Running `wrap-test.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux).

The static wrapper library is built by default (`make`), and suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_`). To build and use the shared wrapper library:

```bash
make BLAS_STATIC=0

LD_PRELOAD=/path/to/libwrap.so ./application
```

## Low-Precision GEMM

The wrapper library intercepts an application's calls to DGEMM, and executes Ozaki Scheme-1 to emulate FP64 accuracy based on low-precision matrix multiplication. The code supports compile-time configuration of the multiplication scheme (`TRIANGULAR`, `SYMMETRIZE`, `REVERSE_PASS`, and `TRIM_FORWARD`), the number of slices aka "splits" used to achieve certain accuracy (`NSLICES`), the size of the matrices employed by potential "matrix cores" (`BLOCK_M`, `BLOCK_N`, and `BLOCK_K`). The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

| Config | Forward | Reverse | Total | Notes |
|---|---|---|---|---|
| `TRIANGULAR` only | S^2 / 2 | 0 | S^2 / 2 | Upper triangle only |
| + `SYMMETRIZE` | S^2 / 2 | 0 | S^2 / 2 | + doubling approximation |
| + `REVERSE_PASS` (no trim) | S^2 / 2 | S^2 / 4 | 3S^2 / 4 | Upper + lower significant |
| + `REVERSE_PASS` + `TRIM_FORWARD` | S^2 / 4 | S^2 / 4 | **S^2 / 2** | Symmetric coverage at original cost (default) |
| Full (no `TRIANGULAR`) | S^2 | 0 | S^2 | All pairs |

At runtime, the wrapper can be instructed to print a running statistic of the observed accuracy. The default is `GEMM_VERBOSE=0` whereas `GEMM_VERBOSE=1` environment variable prints the accumulated statistic when the application terminates, and `GEMM_VERBOSE=N` prints every *N*th call of DGEMM. The environment variable `GEMM_DIFF` can select tracking the A-matrix (1), the B-matrix (2) representations, and by default tracks the C-matrix against the original DGEMM. For convenience, `GEMM_OZAKI=0` calls the original DGEMM right away (no LP-GEMM is involved). If a multiplication yields a smaller RSQ than what is given by the environment variable `GEMM_RSQ`, the related GEMM arguments are printed and the A-matrix and B-matrix are dumped as MHD-files along with transposition, leading dimension, and alpha or beta argument. The RSQ-value is zero by default (no prints and dumps), and it is updated after every print-dump to only print-dump again if the value is undercut again. The environment variable `NREPEAT` controls the number of GEMM calls (default is 1). When `NREPEAT` is greater than 1, the first call serves as warmup and is excluded from timing.

The test driver (`gemm.c`) accepts positional arguments:

```text
gemm-wrap.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
gemm-blas.x [A.mhd|M [B.mhd|N] [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]
```

TA and TB select transposition: 0 means 'N' (no transpose), non-zero means 'T' (transpose). The code can be built with `GEMM_REAL_TYPE` (default `double`) and `GEMM_INT_TYPE` (default `int`) to select precision and integer width.

If the driver is called with MHD-files, accuracy issues can be analyzed outside of an application.

