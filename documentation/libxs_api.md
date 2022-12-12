# Overview

LIBXS splits each call into `dispatch`, which JIT compiles the kernel by selecting the optimal placement of instructions for a particular set of matrices and returns a function pointer, and `invoke`, which calls that function pointer with the matrices' pointers as arguments.

The basic pattern is:
```
// Get the shape of the operands to build the right op
op_shape = libxs_create_<op>_shape(matrix dimensions, leading dimensions, data types);

// Define a new kernel
libxs_xsmmfunction kernel = {NULL};

// Dispatch (JIT compile or retrieve the previously compiled function)
kernel.<op> = libxs_dispatch_<op>_v2(op_shape, FLAGS);

// Prepare the execution parameters
libxs_<op>_param op_param;
op_param.?.? = op parameters

// Call the kernel
kernel.<op>(&op_param);
```

The `<op>` pattern is different, depending on the operation group you want to call.

There are four operation groups:
 * `meltw_unary`: For element-wise operations with a single input and a single output (ex. ReLU).
 * `melw_binary`: For element-wise operations with two inputs and a single output (ex. Add, Sub).
 * `meltw_ternary`: For element-wise operations with three inputs and a single output (ex. ??).
 * `gemm` or `brgemm`: For matrix multiplication operations with three inputs and a single output (ex. GEMM, BRGEMM).

In any of those operations:
 * The output can alias with one of the inputs (ex. accumulation, in-place operations).
 * If one input has a lower rank than the other, and the dimensions are compatible, a broadcast is performed before the operation.
 * If one input has a smaller type (of the same family) than the other, a (safe) type promotion is performed before the operation.

## Binary Operations

```
  LIBXS_API libxs_meltw_binary_shape libxs_create_meltw_binary_shape( const libxs_blasint m, const libxs_blasint n,
                                                                            const libxs_blasint ldi, const libxs_blasint ldi2, const libxs_blasint ldo,
                                                                            const libxs_datatype in0_type, const libxs_datatype in1_type, const libxs_datatype out_type, const libxs_datatype comp_type );
  LIBXS_API libxs_meltwfunction_binary libxs_dispatch_meltw_binary_v2( const libxs_meltw_binary_type binary_type, const libxs_meltw_binary_shape binary_shape, const libxs_bitfield binary_flags );
```

## GEMM Operations

```
LIBXS_API libxs_gemm_shape libxs_create_gemm_shape( const libxs_blasint m, const libxs_blasint n, const libxs_blasint k,
                                                            const libxs_blasint lda, const libxs_blasint ldb, const libxs_blasint ldc,
                                                            const libxs_datatype a_in_type, const libxs_datatype b_in_type, const libxs_datatype out_type, const libxs_datatype comp_type );

  /** Query or JIT-generate SMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
  LIBXS_API libxs_gemmfunction libxs_dispatch_gemm_v2( const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags,
                                                          const libxs_bitfield prefetch_flags );
  /** Query or JIT-generate BRGEMM-kernel general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
  LIBXS_API libxs_gemmfunction libxs_dispatch_brgemm_v2( const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags,
                                                            const libxs_bitfield prefetch_flags, const libxs_gemm_batch_reduce_config brgemm_config );
  /** Query or JIT-generate BRGEMM-kernel with fusion, general mixed precision options and batch reduce; returns NULL if it does not exist or if JIT is not supported */
  LIBXS_API libxs_gemmfunction_ext libxs_dispatch_brgemm_ext_v2( const libxs_gemm_shape gemm_shape, const libxs_bitfield gemm_flags,
                                                                    const libxs_bitfield prefetch_flags, const libxs_gemm_batch_reduce_config brgemm_config,
                                                                    const libxs_gemm_ext_unary_argops unary_argops, const libxs_gemm_ext_binary_postops binary_postops );
```
