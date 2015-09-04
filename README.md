# LIBXS
Library for small matrix-matrix multiplications targeting Intel Architecture (x86). The library generates code for the following instruction set extensions: Intel SSE3, Intel AVX, Intel AVX2, IMCI (KNCni) for Intel Xeon Phi coprocessors ("KNC"), and Intel AVX-512 as found in the Intel Xeon Phi processor family ("KNL") and future Intel Xeon processors. Historically the library was solely targeting the Intel Many Integrated Core Architecture "MIC") using intrinsic functions, meanwhile optimized assembly code is targeting all fore mentioned instruction set extensions. [[pdf](https://github.com/hfp/libxs/raw/master/documentation/libxs.pdf)] [[src](https://github.com/hfp/libxs/archive/0.9.1.zip)]

**What is a small matrix-matrix multiplication?** When characterizing the problem size using the M, N, and K parameters, a problem size suitable for LIBXS falls approximately within (M N K)^(1/3) <= 60 (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). However the code generator only generates code up to the specified [threshold](#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling), but also miss to implement a tiling scheme to effectively utilize the L2 cache. For problem sizes above the configurable threshold, LIBXS is falling back to BLAS.

**How to determine whether an application can benefit from using LIBXS or not?** Given the application uses BLAS to carry out matrix multiplications, one may link against Intel MKL 11.2 (or higher), set the environment variable MKL_VERBOSE=1, and run the application using a representative workload (env MKL_VERBOSE=1 ./workload > verbose.txt). The collected output is the starting point for evaluating the problem sizes as imposed by the workload (grep -a "MKL_VERBOSE DGEMM" verbose.txt | cut -d, -f3-5).

## Interface
The interface of the library is *generated* according to the [Build Instructions](#build-instructions), and is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxs/blob/master/src/libxs.template.h) and [FORTRAN](https://github.com/hfp/libxs/blob/master/src/libxs.template.f90). To perform the matrix-matrix multiplication *c*<sub>*m* x *n*</sub> = *c*<sub>*m* x *n*</sub> + *a*<sub>*m* x *k*</sub> \* *b*<sub>*k* x *n*</sub>, the following interfaces can be used:

```C
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

# Performance
## Auto-dispatch
The function 'libxs_?mm_dispatch' helps amortizing the cost of the dispatch when multiple calls with the same M, N, and K are needed. In contrast, the automatic code dispatch is orchestrating three levels:

1. Specialized routine (implemented in assembly code),
2. Inlinable C/C++ code or optimized FORTRAN code, and
3. BLAS library call.

All three levels are accessible directly (see [Interface](#interface)) allowing to customize the code dispatch. The level 2 and 3 may be supplied by the Intel Math Kernel Library (Intel MKL) 11.2 DIRECT CALL feature. Beside of the generic interface, one can also call a specific kernel e.g., 'libxs_dmm_4_4_4' multiplying 4x4 matrices. For the latter, the code generator generates prototypes for all specialized functions (interface).

Further, a preprocessor symbol denotes the largest problem size (*M* x *N* x *K*) that belongs to level (1) and (2), and therefore determines if a matrix multiplication falls back to level (3) of calling the LAPACK/BLAS library alongside of LIBXS. This threshold can be configured using for example:

```
make THRESHOLD=$((24 * 24 * 24))
```

The maximum of the given threshold and the largest requested specialization refines the value of the threshold. If a problem size falls below the threshold, dispatching the code requires to figure out whether a specialized routine exists or not. This can be implemented by bisecting a table of all specialized functions (binary search). At the expense of storing function pointers for the entire problem space below the threshold, a direct lookup can be used instead. The actual behavior can be configured using for example:

```
make SPARSITY=2
```

A binary search is implemented when a sparsity (calculated at construction time of the library) is above the given SPARSITY value. Raising the given value prevents generating a binary search (and generates a direct lookup) whereas a value below or equal one is generating the binary search. Furthermore, the size of the direct lookup table is limited to 512 KB (currently hardcoded). The overhead of auto-dispatched multiplications based on the binary search becomes negligible with reasonable problem sizes (above ~20x20 matrices), but may be significant for very small auto-dispatched matrix-matrix multiplication.

## Results
The library does not claim to be "optimal" or "best-performing", and the presented results are modeling certain applications which are not representative "in general". Instead, information on how to reproduce the results are given for each of the reported cases.

![This performance plot for a dual-socket Intel Xeon E5-2699v3 ("Haswell") shows a "compact selection" (to make the plot visually more appealing) out of 328 specializations as useful for CP2K Open Source Molecular Dynamics [1]. The code has been generated and built by running "./[make.sh](https://github.com/hfp/libxs/blob/master/make.sh) -cp2k AVX=2 -j". This and below plots were generated by running "cd samples/cp2k ; ./[cp2k-plot.sh](https://github.com/hfp/libxs/blob/master/samples/cp2k/cp2k-plot.sh) specialized cp2k-specialized.png -1". Please note that larger problem sizes (MNK) carry a higher arithmetic intensity which usually leads to higher performance (less bottlenecked by memory bandwidth).](https://raw.githubusercontent.com/hfp/libxs/master/samples/cp2k/results/Xeon_E5-2699v3/1-cp2k-specialized.png)
> This performance plot for a dual-socket Intel Xeon E5-2699v3 ("Haswell") shows a "compact selection" (to make the plot visually more appealing) out of 328 specializations as useful for CP2K Open Source Molecular Dynamics [1]. The code has been generated and built by running "./[make.sh](https://github.com/hfp/libxs/blob/master/make.sh) -cp2k AVX=2 -j". This and below plots were generated by running "cd samples/cp2k ; ./[cp2k-plot.sh](https://github.com/hfp/libxs/blob/master/samples/cp2k/cp2k-plot.sh) specialized cp2k-specialized.png -1". Please note that larger problem sizes (MNK) carry a higher arithmetic intensity which usually leads to higher performance (less bottlenecked by memory bandwidth).

![This plot summarizes the performance of the generated kernels by averaging the results over K (and therefore the bar on the right hand side may not show the same maximum when compared to other plots). The performance is well-tuned across the parameter space with no "cold islands", and the lower left "cold" corner is fairly limited. Please refer to the first figure on how to reproduce the results.](https://raw.githubusercontent.com/hfp/libxs/master/samples/cp2k/results/Xeon_E5-2699v3/2-cp2k-specialized.png)
> This plot summarizes the performance of the generated kernels by averaging the results over K (and therefore the bar on the right hand side may not show the same maximum when compared to other plots). The performance is well-tuned across the parameter space with no "cold islands", and the lower left "cold" corner is fairly limited. Please refer to the first figure on how to reproduce the results.

![This plot shows the arithmetic average (non-sliding) of the performance with respect to groups of problem sizes (MNK). The problem sizes are binned into three groups according to the shown intervals: "Small", "Medium", and "Larger" (notice that "larger" may still not be a large problem size). Please refer to the first figure on how to reproduce the results.](https://raw.githubusercontent.com/hfp/libxs/master/samples/cp2k/results/Xeon_E5-2699v3/3-cp2k-specialized.png)
> This plot shows the arithmetic average (non-sliding) of the performance with respect to groups of problem sizes (MNK). The problem sizes are binned into three groups according to the shown intervals: "Small", "Medium", and "Larger" (notice that "larger" may still not be a large problem size). Please refer to the first figure on how to reproduce the results.

![In order to further summarize the previous plots, this graph shows the cumulative distribution function (CDF) of the performance across all cases. Similar to the median value at 50%, one can read for example that 100% of the cases are yielding less or equal the largest discovered value. The value highlighted by the arrows is usually the median value, the [plot script](https://github.com/hfp/libxs/blob/master/samples/cp2k/cp2k-perf.plt) however attempts to highlight a single "fair performance value" representing all cases by linearly fitting the CDF, projecting onto the x-axis, and taking the midpoint of the projection (usually at 50%). Please note, that this diagram shows a statistical distribution and does not allow to identify any particular kernel. Moreover at any point of the x-axis ("Probability"), the "Compute Performance" and the "Memory Bandwidth" graph do not necessarily belong to the same kernel! Please refer to the first figure on how to reproduce the results.](https://raw.githubusercontent.com/hfp/libxs/master/samples/cp2k/results/Xeon_E5-2699v3/4-cp2k-specialized.png)
> In order to further summarize the previous plots, this graph shows the cumulative distribution function (CDF) of the performance across all cases. Similar to the median value at 50%, one can read for example that 100% of the cases are yielding less or equal the largest discovered value. The value highlighted by the arrows is usually the median value, the [plot script](https://github.com/hfp/libxs/blob/master/samples/cp2k/cp2k-perf.plt) however attempts to highlight a single "fair performance value" representing all cases by linearly fitting the CDF, projecting onto the x-axis, and taking the midpoint of the projection (usually at 50%). Please note, that this diagram shows a statistical distribution and does not allow to identify any particular kernel. Moreover at any point of the x-axis ("Probability"), the "Compute Performance" and the "Memory Bandwidth" graph do not necessarily belong to the same kernel! Please refer to the first figure on how to reproduce the results.

# Limitations
Beside of the inlinable C or optimized FORTRAN code path, the library is currently limited to a single code path which is selected at build time of the library. Without a specific flag (SSE=1, AVX=1|2|3), the assembly code generator emits code for all supported instruction set extensions. However, the compiler is picking only one of the generated code paths according to its code generation flags (or according to what is native with respect to the compiler-host). A future version of the library may be including all code paths at build time and allow for runtime-dynamic dispatch of the most suitable code path.

A future version of the library may support an auto-tuning stage when generating the code (to find M,N,K-combinations for specialized routines which are beneficial compared to the code generated from the inlinable C or optimized FORTRAN code path). Auto-tuning the compiler code generation using a profile-guided optimization may be another option to be incorporated into the build system (Makefile). However, auto-tuning also imposes a reasonable burden (similar to profile guided optimization) and the current library is supposed to generate extremely efficient code for everything that does not rely on L2 cache blocking (below THRESHOLD), and except for very small problem sizes where a dedicated kernel may beat our implementation (e.g., MNK=4x4x4).

## Applications and References
**\[1] [http://cp2k.org/](http://cp2k.org/)**: Open Source Molecular Dynamics which (optionally) uses LIBXS. The application is generating batches of small matrix-matrix multiplications ("matrix stack") out of a problem-specific distributed block-sparse matrix (see https://github.com/hfp/libxs/raw/master/documentation/cp2k.pdf).

**\[2] [https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/)**: SeisSol is one of the leading codes for earthquake scenarios, in particular for simulating dynamic rupture processes. LIBXS provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see https://github.com/TUM-I5/seissol_kernels/).

**\[3] [http://software.intel.com/xeonphicatalog](http://software.intel.com/xeonphicatalog)**: Intel Xeon Phi Applications and Solutions Catalog.

**\[4] [http://goo.gl/qsnOOf](https://software.intel.com/en-us/articles/intel-and-third-party-tools-and-libraries-available-with-support-for-intelr-xeon-phitm)**: Intel 3rd Party Tools and Libraries.
