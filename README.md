# LIBXS
LIBXS is a library for small dense and small sparse matrix-matrix multiplications targeting Intel Architecture (x86). The library is generating code for the following instruction set extensions: Intel SSE3, Intel AVX, Intel AVX2, IMCI (KNCni) for Intel Xeon Phi coprocessors ("KNC"), and Intel AVX-512 as found in the Intel Xeon Phi processor family ("KNL") and future Intel Xeon processors. Historically the library was solely targeting the Intel Many Integrated Core Architecture "MIC") using intrinsic functions, meanwhile optimized assembly code is targeting all aforementioned instruction set extensions (static code generation), and Just-In-Time (JIT) code generation is targeting Intel AVX and beyond. [[pdf](https://github.com/hfp/libxs/raw/master/documentation/libxs.pdf)] [[src](https://github.com/hfp/libxs/archive/1.0.1.zip)] [![status](https://travis-ci.org/hfp/libxs.svg?branch=master "Master branch build status")](https://github.com/hfp/libxs/archive/master.zip)

**What is a small matrix-matrix multiplication?** When characterizing the problem size using the M, N, and K parameters, a problem size suitable for LIBXS falls approximately within (M N K)<sup>1/3</sup> \<= 80 (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). However the code generator only generates code up to the specified [threshold](#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M and K dimension), but also miss to implement a tiling scheme to effectively utilize the L2 cache. For problem sizes above the configurable threshold, LIBXS is falling back to BLAS.

**How to determine whether an application can benefit from using LIBXS or not?** Given the application uses BLAS to carry out matrix multiplications, one may link against Intel MKL 11.2 (or higher), set the environment variable MKL_VERBOSE=1, and run the application using a representative workload (env MKL_VERBOSE=1 ./workload > verbose.txt). The collected output is the starting point for evaluating the problem sizes as imposed by the workload (grep -a "MKL_VERBOSE DGEMM" verbose.txt | cut -d, -f3-5).

## Interface
The interface of the library is *generated* according to the [Build Instructions](#build-instructions), and is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxs/blob/master/src/libxs.template.h) and [FORTRAN](https://github.com/hfp/libxs/blob/master/src/libxs.template.f).

In order to initialize the dispatch-table or other internal resources, one may call an explicit initialization routine in order to avoid lazy initialization overhead when calling LIBXS for the first time. The library deallocates internal resources automatically, but also provides a companion to the aforementioned initialization (finalize).

```C
/** Initialize the library; pay for setup cost at a specific point. */
void libxs_init();
/** Uninitialize the library and free internal memory (optional). */
void libxs_finalize();
```

To perform the dense matrix-matrix multiplication *C<sub>m&thinsp;x&thinsp;n</sub> = alpha &middot; A<sub>m&thinsp;x&thinsp;k</sub> &middot; B<sub>k&thinsp;x&thinsp;n</sub> + beta &middot; C<sub>m&thinsp;x&thinsp;n</sub>*, the full-blown GEMM/BLAS interface can be treated with "default arguments":

```C
/** Calling the automatically dispatched dense matrix multiplication (single/double-precision, C code). */
libxs_?gemm(NULL/*transa*/, NULL/*transb*/, &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
/** Calling the automatically dispatched dense matrix multiplication (C++ code). */
libxs_gemm(NULL/*transa*/, NULL/*transb*/, m/*required*/, n/*required*/, k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

For the C interface (with type prefix 's' or 'd'), all arguments and in particular m, n, and k are passed by pointer. This is needed for binary compatibility with the original GEMM/BLAS interface. In contrast, the C++ interface is supplying overloaded versions which allow to passing m, n, and k by-value (which makes it more clear that m, n, and k are non-optional arguments).

The Fortran interface supports optional arguments (without affecting the binary compatibility with the original LAPACK/BLAS interface) by allowing to omit arguments (where the C/C++ interface is allowing NULL to be passed). For convenience, a similar BLAS-based dense matrix multiplication (libxs_blas_gemm instead of libxs_gemm) is provided for all supported languages which is simply re-exposing the underlying GEMM/BLAS implementation. However, the re-exposed functions perform argument twiddling to account for ROW_MAJOR storage order (if enabled). The BLAS-based GEMM might be useful for validation/benchmark purposes, and more important as a fallback implementation when building an application-specific dispatch mechanism.

```Fortran
! Calling the automatically dispatched dense matrix multiplication (single/double-precision).
CALL libxs_?gemm(m=m, n=n, k=k, a=a, b=b, c=c)
! Calling the automatically dispatched dense matrix multiplication (generic interface).
CALL libxs_gemm(m=m, n=n, k=k, a=a, b=b, c=c)
```

Successively calling a particular kernel (i.e., multiple times) allows for amortizing the cost of the code dispatch. Moreover in order to customize the dispatch mechanism, one can rely on the following interface.

```C
/** If non-zero function pointer is returned, call (*function_ptr)(a, b, c). */
libxs_smmfunction libxs_smmdispatch(int m, int n, int k,
                                    int lda, int ldb, int ldc,
                                    /* supply NULL as a default for alpha or beta */
                                    const float* alpha, const float* beta);
/** If non-zero function pointer is returned, call (*function_ptr)(a, b, c). */
libxs_dmmfunction libxs_dmmdispatch(int m, int n, int k,
                                    int lda, int ldb, int ldc,
                                    /* supply NULL as a default for alpha or beta */
                                    const double* alpha, const double* beta);
```

A variety of overloaded function signatures is provided allowing to omit arguments not deviating from the configured defaults. Moreover, in C++ a type 'libxs_mmfunction<*type*>' can be used to instantiate a functor rather than making a distinction for the numeric type in 'libxs_?mmdispatch'. Similarly in Fortran, when calling the generic interface (libxs_mmdispatch) the given LIBXS_?FUNCTION is dispatched such that libxs_call can be used to actually perform the function call using the PROCEDURE POINTER wrapped by LIBXS_?FUNCTION. Beside of dispatching code, one can also call a specific kernel (e.g., 'libxs_dmm_4_4_4') using the prototype functions included for statically generated kernels.

# Performance
### Tuning
By default all supported host code paths are generated (with the compiler picking the one according to the feature bits of the host). Specifying a particular code path will not only save some time when generating the static code ("printing"), but also enable cross-compilation for a target that is different from the compiler's host. The build system allows to conveniently select the target system when invoking 'make': SSE=3 (in fact SSE!=0), AVX=1, AVX=2 (with FMA), and AVX=3 are supported. The latter is targeting the Intel Knights Landing processor family ("KNL") and future Intel Xeon processors using foundational Intel AVX-512 instructions (AVX-512F):

```
make AVX=3
```

An extended interface can generated which allows to perform software prefetches. Prefetching data might be helpful when processing batches of matrix multiplications where the next operands are farther away or otherwise unpredictable in their memory location. The prefetch strategy can be specified similar as shown in the section [Directly invoking the generator backend](#directly-invoking-the-generator-backend) i.e., by either using the number of the shown enumeration, or by exactly using the name of the prefetch strategy. The only exception is PREFETCH=1 which is enabling a default strategy ("AL2_BL2viaC" rather than "nopf"). The following example is requesting the "AL2jpst" strategy:

```
make PREFETCH=8
```

The interface which is supporting software prefetches extends the signature of all kernels by three arguments (pa, pb, and pc) allowing the call-side to specify where to prefetch the operands of the "next" multiplication from (a, b, and c). There are [macros](https://github.com/hfp/libxs/blob/master/src/libxs_prefetch.h) available (C/C++ only) allowing to call the matrix multiplication functions in a prefetch-agnostic fashion (see [cp2k](https://github.com/hfp/libxs/blob/master/samples/cp2k/cp2k.cpp) or [smm](https://github.com/hfp/libxs/tree/master/samples/smm samples) code samples). Further, the generated interface of the library also encodes the parameters the library was built for (static information). This helps optimizing client code related to the library's functionality. For example, the LIBXS_MAX_* and LIBXS_AVG_* information can be used with the LIBXS_PRAGMA_LOOP_COUNT macro in order to hint loop trip counts when handling matrices related to the problem domain of LIBXS.

### Auto-dispatch
The function 'libxs_?mmdispatch' helps amortizing the cost of the dispatch when multiple calls with the same M, N, and K are needed. The automatic code dispatch is orchestrating two levels:

1. Specialized routine (implemented in assembly code),
3. LAPACK/BLAS library call (fallback).

Both levels are accessible directly (see [Interface](#interface)) allowing to customize the code dispatch. The fallback level may be supplied by the Intel Math Kernel Library (Intel MKL) 11.2 DIRECT CALL feature. 

Further, a preprocessor symbol denotes the largest problem size (*M* x *N* x *K*) that belongs to the first level, and therefore determines if a matrix multiplication falls back to calling into the LAPACK/BLAS library alongside of LIBXS. The problem size threshold can be configured by using for example:

```
make THRESHOLD=$((60 * 60 * 60))
```

The maximum of the given threshold and the largest requested specialization refines the value of the threshold. If a problem size is below the threshold, dispatching the code requires to figure out whether a specialized routine exists or not.

## Directly invoking the generator backend
In rare situations it might be useful to directly incorporate generated C code (with inline assembly regions). This is accomplished by invoking a driver program (with certain command line arguments). The driver program is built as part of LIBXS's build process (when requesting static code generation), but also available via a separate build target:

```
make generator
bin/libxs_generator
```

The code generator driver program accepts the following arguments:

1. dense/dense_asm/sparse (dense creates C code, dense_asm creates ASM)
2. Filename of a file to append to
3. Routine name to be created
4. M parameter
5. N parameter
6. K parameter
7. LDA (0 when 1. is "sparse" indicates A is sparse)
8. LDB (0 when 1. is "sparse" indicates B is sparse)
9. LDC parameter
10. alpha (-1 or 1)
11. beta (0 or 1)
12. Alignment override for A (1 auto, 0 no alignment)
13. Alignment override for C (1 auto, 0 no alignment)
14. Architecture (noarch, wsm, snb, hsw, knc, knl)
15. Prefetch strategy, see below enumeration (dense/dense_asm only)
16. single precision (SP), or double precision (DP)
17. CSC file (just required when 1. is "sparse"). Matrix market format.

The prefetch strategy can be:

1. "nopf": no prefetching at all, just 3 inputs (\*A, \*B, \*C)
2. "pfsigonly": just prefetching signature, 6 inputs (\*A, \*B, \*C, \*A’, \*B’, \*C’)
3. "BL2viaC": uses accesses to \*C to prefetch \*B’
4. "AL2": uses accesses to \*A to prefetch \*A’
5. "curAL2": prefetches current \*A ahead in the kernel
6. "AL2_BL2viaC": combines AL2 and BL2viaC
7. "curAL2_BL2viaC": combines curAL2 and BL2viaC
8. "AL2jpst": aggressive \*A’ prefetch of first rows without any structure
9. "AL2jpst_BL2viaC": combines AL2jpst and BL2viaC

Here are some examples of invoking the driver program:

```
bin/libxs_generator dense foo.c foo 16 16 16 32 32 32 1 1 1 1 hsw nopf DP
bin/libxs_generator dense_asm foo.c foo 16 16 16 32 32 32 1 1 1 1 knl AL2_BL2viaC DP
bin/libxs_generator sparse foo.c foo 16 16 16 32 0 32 1 1 1 1 hsw nopf DP bar.csc
```

Please note, there are additional examples given in samples/generator and samples/seissol.

## Implementation
## Applications and References
**\[1] [http://cp2k.org/](http://cp2k.org/)**: Open Source Molecular Dynamics with its DBCSR component generating batches of small matrix multiplications ("matrix stacks") out of a problem-specific distributed block-sparse matrix. The idea and the interface of LIBXS is sharing some origin with CP2K's "libsmm" library which can be optionally substituted by LIBXS (see https://github.com/hfp/libxs/raw/master/documentation/cp2k.pdf).

**\[2] [https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/)**: SeisSol is one of the leading codes for earthquake scenarios, in particular for simulating dynamic rupture processes. LIBXS provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see https://github.com/TUM-I5/seissol_kernels/).

**\[3] [http://software.intel.com/xeonphicatalog](http://software.intel.com/xeonphicatalog)**: Intel Xeon Phi Applications and Solutions Catalog.

**\[4] [http://goo.gl/qsnOOf](https://software.intel.com/en-us/articles/intel-and-third-party-tools-and-libraries-available-with-support-for-intelr-xeon-phitm)**: Intel 3rd Party Tools and Libraries.
