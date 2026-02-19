# Ozaki Scheme-1 based Low-Precision GEMM

## Intercepted GEMM

This code sample is calling DGEMM and only relies on LAPACK/BLAS interface. Two variants can be linked when building the source code: (1) code which is dynamically linked against LAPACK/BLAS (`gemm-blas.x`), (2) code which is linked using `--wrap=`*symbol* supported by GNU&#160;GCC compatible tool chains (`gemm-wrap.x`). Running `wrap-test.sh` exercises three flavors: the two build variants and additionally the first variant using the LD_PRELOAD mechanism (available under Linux).

The static wrapper library is built by default (`make`), and suitable for applications with static linkage against a LAPACK/BLAS library (`-Wl,--wrap=dgemm_`). To build and use the shared wrapper library:

```bash
make BLAS_STATIC=0

LD_PRELOAD=/path/to/libwrap.so ./application
```

## Low-Precision GEMM

The wrapper library intercepts an application's calls to DGEMM, and executes Ozaki Scheme-1 to emulate FP64 accuracy based on low-precision matrix multiplication. The code supports compile-time configuration of the multiplication scheme (`TRIANGULAR`), the number of slices aka "splits" used to achieve certain accuracy (`NSLICES`), the size of the matrices employed by potential "matrix cores" (`BLOCK_M`, `BLOCK_N`, and `BLOCK_K`). The term "slices" is preferred over "splits" since the latter suggests *N* splits would yield *N+1* slices.

At runtime, the wrapper can be instructed to print a running statistic of the observed accuracy. The default is `GEMM_VERBOSE=0` whereas `GEMM_VERBOSE=1` environment variable prints the accumulated statistic when the application terminates, and `GEMM_VERBOSE=N` prints every *N*th call of DGEMM. The environment variable `GEMM_DIFF` can select tracking the A-matrix (1), the B-matrix (2) representations, and by default tracks the C-matrix against the original DGEMM. For convenience, `GEMM_OZAKI=0` calls the original DGEMM right away (no LP-GEMM is involved).

The test driver (`gemm.c`) accepts positional arguments:

```text
gemm-wrap.x [M [N [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]]
gemm-blas.x [M [N [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]]
```

