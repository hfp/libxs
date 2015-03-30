# LIBXS
Library for small matrix-matrix multiplications targeting Intel Architecture (x86). The initial version of the library was targeting the Intel Xeon Phi coprocessor (an instance of the Intel Many Integrated Core Architecture "MIC") particularly by using KNC intrinsic functions (called KNCni or IMCI). Today, the library reaches the Many Integrated Core Architecture as well as other hardware which is capable of executing Intel Advanced Vector Extensions 512 (Intel AVX-512). Please also have a look at the collection of upcoming [enhancements](https://github.com/hfp/libxs/labels/enhancement). [[pdf](https://github.com/hfp/libxs/raw/master/documentation/libxs.pdf)] [[src](https://github.com/hfp/libxs/archive/0.8.3.zip)]

The library provides a sophisticated dispatch mechanism (see [More Details](#more-details)) which is also targeting other instruction sets (beside of the Intrinsic code path). The library can be also compiled to "MIC native code" which is able to run self-hosted as well as in an offloaded code region (via a FORTRAN directive or via C/C++ preprocessor pragma). The prerequisite for offloading the code is to compile it to position-independent (PIC) code even when building a static library.

Performance: the presented code is by no means "optimal" or "best-performing" - it just uses Intrinsics. In fact, a well-optimizing compiler may arrange better code compared to what is laid out via the library's Python scripts. The latter can be exploited by just relying on the "inlinable code" and by not generating specialized functions.

## Interface
The interface of the library is *generated* according to the [Build Instructions](#build-instructions) (therefore the header file 'include/libxs.h' is **not** stored in the code repository). The generated interface also defines certain preprocessor symbols to store the properties the library was built for.

To perform the matrix-matrix multiplication *c*<sub>*m* x *n*</sub> = *c*<sub>*m* x *n*</sub> + *a*<sub>*m* x *k*</sub> \* *b*<sub>*k* x *n*</sub>, one of the following interfaces can be used:

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

With C++ function overloading, the library allows to omit the 's' and 'd' denoting the numeric type in the above C interface. Further, a type 'libxs_mm_dispatch<*type*>' can be used to instantiate a functor rather than making a distinction for the numeric type in 'libxs_?mm_dispatch'.

# Performance
## Auto-dispatch
The function 'libxs_?mm_dispatch' helps amortizing the cost of the dispatch when multiple calls with the same M, N, and K are needed. In contrast, the automatic code dispatch uses three levels:

1. Specialized routine,
2. Inlined code, and
3. BLAS library call.

All three levels are accessible directly (see [Interface](#interface)) in order to allow a customized code dispatch. The level 2 and 3 may be supplied by the Intel Math Kernel Library (Intel MKL) 11.2 DIRECT CALL feature. Beside of the generic interface, one can call a specific kernel e.g., 'libxs_dmm_4_4_4' multiplying 4x4 matrices.

Further, a preprocessor symbol denotes the largest problem size (*M* x *N* x *K*) that belongs to level (1) and (2), and therefore determines if a matrix-matrix multiplication falls back to level (3) of calling the BLAS library linked with the library. This threshold can be configured using for example:

```
make THRESHOLD=$((24 * 24 * 24))
```

The maximum of the given threshold and the largest requested specialization refines the value of the threshold. If a problem size falls below the threshold, dispatching the code requires to figure out whether a specialized routine exists or not. This is implemented searching a table of the specialized functions in a binary manner. At the expense of storing function pointers for the entire problem space below the threshold, a direct lookup can be implemented as well. This can be configured using for example:

```
make SPARSITY=2
```

A sparsity is calculated at construction time of the library determining whether to use binary search (value above given SPARSITY or in case of SPARSITY <= 1) or direct lookup (raising the given SPARSITY allows to prevent the binary search). However, the overhead of the binary search is negligible which still holds (~2%) when using a relatively slower clocked single core of an Intel Xeon Phi coprocessor.

## Applications and References
**\[1] http://cp2k.org/**: Open Source Molecular Dynamics application. Beside of CP2K's own SMM module, LIBXS aims to provide highly optimized assembly kernels.

**\[2] https://github.com/TUM-I5/GemmCodeGenerator**: Code generator for matrix-matrix multiplications. Due to LIBXS's roadmap, this is a related project.

**\[3] http://software.intel.com/xeonphicatalog**: Intel Xeon Phi Applications and Solutions Catalog.

**\[4] [http://goo.gl/qsnOOf](https://software.intel.com/en-us/articles/intel-and-third-party-tools-and-libraries-available-with-support-for-intelr-xeon-phitm)**: Intel 3rd Party Tools and Libraries.
