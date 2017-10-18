# Overview<a name="small-matrix-multiplication-smm"></a>

To perform the dense matrix-matrix multiplication *C<sub>m&#8239;x&#8239;n</sub> = alpha &middot; A<sub>m&#8239;x&#8239;k</sub> &middot; B<sub>k&#8239;x&#8239;n</sub> + beta &middot; C<sub>m&#8239;x&#8239;n</sub>*, the full-blown GEMM interface can be treated with "default arguments" (which is deviating from the BLAS standard, however without compromising the binary compatibility).

```C
/** Automatically dispatched dense matrix multiplication (single/double-precision, C code). */
libxs_?gemm(NULL/*transa*/, NULL/*transb*/, &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
/** Automatically dispatched dense matrix multiplication (C++ code). */
libxs_gemm(NULL/*transa*/, NULL/*transb*/, m/*required*/, n/*required*/, k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

For the C interface (with type prefix 's' or 'd'), all arguments including m, n, and k are passed by pointer. This is needed for binary compatibility with the original GEMM/BLAS interface. The C++ interface is also supplying overloaded versions where m, n, and k can be passed by&#8209;value (making it clearer that m, n, and k are non-optional arguments).

The FORTRAN interface supports optional arguments (without affecting the binary compatibility with the original BLAS interface) by allowing to omit arguments where the C/C++ interface allows for NULL to be passed.

```FORTRAN
! Automatically dispatched dense matrix multiplication (single/double-precision).
CALL libxs_?gemm(m=m, n=n, k=k, a=a, b=b, c=c)
! Automatically dispatched dense matrix multiplication (generic interface).
CALL libxs_gemm(m=m, n=n, k=k, a=a, b=b, c=c)
```

For convenience, a BLAS-based dense matrix multiplication (`libxs_blas_gemm`) is provided for all supported languages which is simply re-exposing the underlying GEMM/BLAS implementation. The BLAS-based GEMM might be useful for validation/benchmark purposes, and more important as a fallback when building an application-specific dispatch mechanism.

```C
/** Automatically dispatched dense matrix multiplication (single/double-precision). */
libxs_blas_?gemm(NULL/*transa*/, NULL/*transb*/, &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

A more recently added variant of matrix multiplication is parallelized based on the OpenMP standard. These routines will open an internal parallel region and rely on "classic" thread-based OpenMP. If these routines are called from inside of a parallel region, the parallelism will be based on tasks (OpenMP&#160;3.0). Please note that all OpenMP-based routines are hosted by the extension library (libxsext), which keeps the main library agnostic with respect to a threading runtime.

```C
/** OpenMP parallelized dense matrix multiplication (single/double-precision). */
libxs_?gemm_omp(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
```

## Batched Multiplication

In case of batched SMMs, it can be beneficial to supply "next locations" such that the upcoming operands are prefetched ahead of time. Such a location would be the address of the next matrix to be multiplied (and not any of the floating-point elements within the "current" matrix-operand). The "prefetch strategy" is requested at dispatch-time of a kernel. A [strategy](libxs_be.md#prefetch-strategy) other than `LIBXS_PREFETCH_NONE` turns the signature of a JIT'ted kernel into a function with six arguments (`a,b,c, pa,pb,pc` instead of `a,b,c`). To defer the decision about the strategy to a CPUID-based mechanism, one can choose `LIBXS_PREFETCH_AUTO`.

```C
int prefetch = LIBXS_PREFETCH_AUTO;
int flags = 0; /* LIBXS_FLAGS */
libxs_dmmfunction xmm = NULL;
double alpha = 1, beta = 0;
xmm = libxs_dmmdispatch(23/*m*/, 23/*n*/, 23/*k*/,
  NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
  &alpha, &beta, &flags, &prefetch);
```

Above, pointer-arguments of `libxs_dmmdispatch` can be NULL (or OPTIONAL in FORTRAN): for LDx this means a "tight" leading dimension, alpha, beta, and flags are given by a [default value](https://github.com/hfp/libxs/blob/master/src/template/libxs_config.h) (which is selected at compile-time), and for the prefetch strategy a NULL-argument refers to "no prefetch" (which is equivalent to an explicit `LIBXS_PREFETCH_NONE`).

```C
/** Process a series of matrix multiplications (explicit data representation). */
int libxs_mmbatch(const libxs_gemm_descriptor* descriptor,
  const void* a_matrix, const void* b_matrix, void* c_matrix,
  int index_base, int index_stride, const unsigned int a_stride[],
  const unsigned int b_stride[], const unsigned int c_stride[],
  unsigned int batchsize);
```

To further simplify the multiplication of matrices in a batch, the above interface can help if an explicit data representation is available. A lower level form (`libxs_mmbatch_thread`) can employ a user-defined threading runtime, which is already the case for OpenMP (`libxs_mmbatch_omp`) and hosted by the extension library (libxsext). Please note that an explicit data presentation is not necessary to prefetch a series of matrix multiplications. A "chain" of multiplications can be programmatically described without the need to build arrays of operands or indexes.

### Call Wrapper

Since the library is binary compatible with existing GEMM calls (BLAS), these calls can be replaced at link-time or intercepted at runtime of an application such that LIBXS is used instead of the original BLAS library. There are two cases to consider:

1. An application which is linked statically against BLAS requires to wrap the 'sgemm_' and the 'dgemm_' symbol (an alternative is to wrap only 'dgemm_'), and a special build of the libxs(ext) library is required (`make WRAP=1` to wrap SGEMM and DGEMM, or `make WRAP=2` to wrap only DGEMM):  
`gcc [...] -Wl,--wrap=sgemm_,--wrap=dgemm_ /path/to/libxsext.a /path/to/libxs.a /path/to/your_regular_blas.a`
    * Relinking the application as shown above can often be accomplished by copying, pasting, modifying the linker command, and then re-invoking the modified link step. This linker command may appear as console output of the application's "make" command (or a similar build system).
    * The static link-time wrapper technique may only work with a GCC tool chain (GNU Binutils: `ld`, or `ld` via compiler-driver), and it has been tested with GNU GCC, Intel&#160;Compiler, and Clang. However, this does not work under Microsoft Windows (even when using the GNU tool chain), and it may not work under OS&#160;X (Compiler&#160;6.1 or earlier, later versions have not been tested).
2. An application that is dynamically linked against BLAS allows to intercept the GEMM calls at startup time (runtime) of the unmodified executable by using the LD_PRELOAD mechanism. The shared library of LIBXS (`make STATIC=0`) can be used to intercept GEMM calls:  
`LD_PRELOAD=/path/to/libxsext.so ./myapplication`

**NOTE**: Using the same multiplication kernel in a consecutive fashion (batch-processing) allows to extract higher performance, when using LIBXS's native programming interface.

