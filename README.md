# LIBXS
LIBXS is a library for small matrix-matrix multiplications targeting Intel Architecture (x86). The library is generating code for the following instruction set extensions: Intel SSE3, Intel AVX, Intel AVX2, IMCI (KNCni) for Intel Xeon Phi coprocessors ("KNC"), and Intel AVX-512 as found in the Intel Xeon Phi processor family ("KNL") and future Intel Xeon processors. Historically the library was solely targeting the Intel Many Integrated Core Architecture "MIC") using intrinsic functions, meanwhile optimized assembly code is targeting all aforementioned instruction set extensions (static code generation), and Just-In-Time (JIT) code generation is targeting Intel AVX and beyond. [[pdf](https://github.com/hfp/libxs/raw/master/documentation/libxs.pdf)] [[src](https://github.com/hfp/libxs/archive/1.0.1.zip)] [![status](https://travis-ci.org/hfp/libxs.svg?branch=master "Master branch build status")](https://github.com/hfp/libxs/archive/master.zip)

**What is a small matrix-matrix multiplication?** When characterizing the problem size using the M, N, and K parameters, a problem size suitable for LIBXS falls approximately within (M N K)^(1/3) \<= 80 (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). However the code generator only generates code up to the specified [threshold](#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M and K dimension), but also miss to implement a tiling scheme to effectively utilize the L2 cache. For problem sizes above the configurable threshold, LIBXS is falling back to BLAS.

**How to determine whether an application can benefit from using LIBXS or not?** Given the application uses BLAS to carry out matrix multiplications, one may link against Intel MKL 11.2 (or higher), set the environment variable MKL_VERBOSE=1, and run the application using a representative workload (env MKL_VERBOSE=1 ./workload > verbose.txt). The collected output is the starting point for evaluating the problem sizes as imposed by the workload (grep -a "MKL_VERBOSE DGEMM" verbose.txt | cut -d, -f3-5).

## Interface
The interface of the library is *generated* according to the [Build Instructions](#build-instructions), and is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxs/blob/master/src/libxs.template.h) and [FORTRAN](https://github.com/hfp/libxs/blob/master/src/libxs.template.f90). To perform the matrix-matrix multiplication *c*<sub>*m* x *n*</sub> = *c*<sub>*m* x *n*</sub> + *a*<sub>*m* x *k*</sub> \* *b*<sub>*k* x *n*</sub>, the following interfaces can be used:

```C
/** Initialization function to set up LIBXS's dispatching table. One may
    call this routine to avoid lazy initialization overhead in the first
    call to a LIBXS kernel routine */
void libxs_init();
/** If non-zero function pointer is returned, call (*function)(M, N, K). */
libxs_smm_function libxs_smm_dispatch(int m, int n, int k);
libxs_dmm_function libxs_dmm_dispatch(int m, int n, int k);
/** Automatically dispatched matrix-matrix multiplication. */
void libxs_smm(int m, int n, int k, const float* a, const float* b, float* c);
void libxs_dmm(int m, int n, int k, const double* a, const double* b, double* c);
/** Non-dispatched matrix-matrix multiplication using inline code. */
void libxs_simm(int m, int n, int k, const float* a, const float* b, float* c);
void libxs_dimm(int m, int n, int k, const double* a, const double* b, double* c);
/** Matrix-matrix multiplication using BLAS. */
void libxs_sblasmm(int m, int n, int k, const float* a, const float* b, float* c);
void libxs_dblasmm(int m, int n, int k, const double* a, const double* b, double* c);
```

With C++ and FORTRAN function overloading, the library allows to omit the 's' and 'd' prefixes denoting the numeric type in the above C interface. Further, in C++ a type 'libxs_mm_dispatch<*type*>' can be used to instantiate a functor rather than making a distinction for the numeric type in 'libxs_?mm_dispatch'.

Note: Function overloading in FORTRAN is only recommended when using automatically dispatched calls. When querying function pointers, using a type-specific query function avoids to rely on using C_LOC for arrays (needed by the polymorphic version) which GNU Fortran (gfortran) refuses to digest (as it is not specified in the FORTRAN standard).

# Performance
### Tuning
By default all supported host code paths are generated (with the compiler picking the one according to the feature bits of the host). Specifying a particular code path will not only save some time when generating the static code ("printing"), but also enable cross-compilation for a target that is different from the compiler's host. The build system allows to conveniently select the target system when invoking 'make': SSE=3 (in fact SSE!=0), AVX=1, AVX=2 (with FMA), and AVX=3 are supported. The latter is targeting the Intel Knights Landing processor family ("KNL") and future Intel Xeon processors using foundational Intel AVX-512 instructions (AVX-512F):

```
make AVX=3
```

The library supports generating code using an "implicitly aligned leading dimension" for the destination matrix of a multiplication. The latter is enabling aligned store instructions, and also hints the inlinable C/C++ code accordingly. The client code may be arranged accordingly by checking the build parameters of the library (at compile-time using the preprocessor). Aligned store instructions imply a leading dimension which is a multiple of the default alignment:

```
make ALIGNED_STORES=1
```

The general alignment (ALIGNMENT=64) as well as an alignment specific for the store instructions (ALIGNED_STORES=n) can be specified when invoking 'make' (by default ALIGNED_STORES inherits ALIGNMENT when "enabled" using ALIGNED_STORES=1). The "implicitly aligned leading dimension" optimization is not expected to have a big impact due to the relatively low amount of store instructions in the mix. In contrast, supporting an "implicitly aligned leading dimension" for loading the input matrices is supposed to make a bigger impact, however this is not exposed by the build system because: (1) aligning a batch of input matrices implies usually larger code changes for the client code whereas accumulating into a local temporary destination matrix is a relatively minor change, and (2) today's Advanced Vector Extensions including the AVX-512 capable hardware supports unaligned load/store instructions.

An extended interface can generated which allows to perform software prefetches. Prefetching data might be helpful when processing batches of matrix multiplications where the next operands are farther away or otherwise unpredictable in their memory location. The prefetch strategy can be specified similar as shown in the section [Directly invoking the generator backend](#directly-invoking-the-generator-backend) i.e., by either using the number of the shown enumeration, or by exactly using the name of the prefetch strategy. The only exception is PREFETCH=1 which is enabling a default strategy ("AL2_BL2viaC" rather than "nopf"). The following example is requesting the "AL2jpst" strategy:

```
make PREFETCH=8
```

The interface which is supporting software prefetches extends the signature of all kernels by three arguments (pa, pb, and pc) allowing the call-side to specify where to prefetch the operands of the "next" multiplication from (a, b, and c). There are [macros](https://github.com/hfp/libxs/blob/master/src/libxs_prefetch.h) available (C/C++ only) allowing to call the matrix multiplication functions in a prefetch-agnostic fashion (see [cp2k](https://github.com/hfp/libxs/blob/master/samples/cp2k/cp2k.cpp) or [smm](https://github.com/hfp/libxs/tree/master/samples/smm samples) code samples).

Further, the generated interface of the library also encodes the parameters the library was built for (static information). This helps optimizing client code related to the library's functionality. For example, the LIBXS_MAX_* and LIBXS_AVG_* information can be used with the LIBXS_PRAGMA_LOOP_COUNT macro in order to hint loop trip counts when handling matrices related to the problem domain of LIBXS.

### Auto-dispatch
The function 'libxs_?mm_dispatch' helps amortizing the cost of the dispatch when multiple calls with the same M, N, and K are needed. In contrast, the automatic code dispatch is orchestrating three levels:

1. Specialized routine (implemented in assembly code),
2. Inlinable C/C++ code or optimized FORTRAN code, and
3. BLAS library call.

All three levels are accessible directly (see [Interface](#interface)) allowing to customize the code dispatch. The level 2 and 3 may be supplied by the Intel Math Kernel Library (Intel MKL) 11.2 DIRECT CALL feature. Beside of the generic interface, one can also call a specific kernel e.g., 'libxs_dmm_4_4_4'. For statically generated kernels, the interface includes prototypes of the specialized functions.

Further, a preprocessor symbol denotes the largest problem size (*M* x *N* x *K*) that belongs to level (1) and (2), and therefore determines if a matrix multiplication falls back to level (3) calling into the LAPACK/BLAS library alongside of LIBXS. The problem size threshold can be configured by using for example:

```
make THRESHOLD=$((60 * 60 * 60))
```

The maximum of the given threshold and the largest requested specialization refines the value of the threshold. If a problem size is below the threshold, dispatching the code requires to figure out whether a specialized routine exists or not.

## Directly invoking the generator backend
In rare situations it might be useful to directly incorporate generated C code (with inline assembly regions). This is accomplished by invoking a driver program (with certain command line arguments). The driver program is built as part of LIBXS's build process (when requesting static code generation), but also available via a separate build target:

```
make generator
bin/generator
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
10. alpha (currently only 1)
11. beta (0 or 1)
12. Alignment override for A (1 auto, 0 no alignment)
13. Alignment override for C (1 auto, 0 no alignment)
14. Prefetch strategy, see below enumeration (dense/dense_asm only)
15. single precision (SP), or double precision (DP)
16. CSC file (just required when 1. is "sparse"). Matrix market format.

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
bin/generator dense foo.c foo 16 16 16 32 32 32 1 1 1 1 hsw nopf DP
bin/generator dense_asm foo.c foo 16 16 16 32 32 32 1 1 1 1 knl AL2_BL2viaC DP
bin/generator sparse foo.c foo 16 16 16 32 0 32 1 1 1 1 hsw nopf DP bar.csc
```

Please note, there are additional examples given in samples/generator and samples/seissol.

## Implementation
## Roadmap
Although the library is under development, the published interface is rather stable and may only be extended in future revisions. The following issues are being addressed in upcoming revisions:

* Full xGEMM interface, and extended code dispatcher
* API supporting sparse matrices and other cases

## Applications and References
**\[1] [http://cp2k.org/](http://cp2k.org/)**: Open Source Molecular Dynamics with its DBCSR component generating batches of small matrix multiplications ("matrix stacks") out of a problem-specific distributed block-sparse matrix. The idea and the interface of LIBXS is sharing some origin with CP2K's "libsmm" library which can be optionally substituted by LIBXS (see https://github.com/hfp/libxs/raw/master/documentation/cp2k.pdf).

**\[2] [https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/)**: SeisSol is one of the leading codes for earthquake scenarios, in particular for simulating dynamic rupture processes. LIBXS provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see https://github.com/TUM-I5/seissol_kernels/).

**\[3] [http://software.intel.com/xeonphicatalog](http://software.intel.com/xeonphicatalog)**: Intel Xeon Phi Applications and Solutions Catalog.

**\[4] [http://goo.gl/qsnOOf](https://software.intel.com/en-us/articles/intel-and-third-party-tools-and-libraries-available-with-support-for-intelr-xeon-phitm)**: Intel 3rd Party Tools and Libraries.
