!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      !> GEMM dispatch with auto-detected backends (MKL JIT, BLAS).
      !> USE LIBXS_JIT instead of LIBXS to get backend-aware
      !> dispatch without explicit function-pointer arguments.
      !> Requires BLAS at link time (dgemm_, sgemm_).
      MODULE LIBXS_JIT
        INCLUDE 'libxs_spec.fi'

#if defined(__MKL)
! MKL JIT kernel creation and retrieval
        INTERFACE
          FUNCTION mkl_cblas_jit_create_dgemm(jitter,                   &
     &    layout, transa, transb, m, n, k,                              &
     &    alpha, lda, ldb, beta, ldc)                                   &
     &    RESULT(status) BIND(C)
            IMPORT :: C_PTR, C_INT, C_DOUBLE
            INTEGER(C_INT) :: status
            TYPE(C_PTR) :: jitter
            INTEGER(C_INT), VALUE :: layout, transa, transb
            INTEGER(C_INT), VALUE :: m, n, k, lda, ldb, ldc
            REAL(C_DOUBLE), VALUE :: alpha, beta
          END FUNCTION
          FUNCTION mkl_jit_get_dgemm_ptr(jitter)                        &
     &    RESULT(ptr) BIND(C)
            IMPORT :: C_FUNPTR, C_PTR
            TYPE(C_FUNPTR) :: ptr
            TYPE(C_PTR), INTENT(IN), VALUE :: jitter
          END FUNCTION
          FUNCTION mkl_cblas_jit_create_sgemm(jitter,                   &
     &    layout, transa, transb, m, n, k,                              &
     &    alpha, lda, ldb, beta, ldc)                                   &
     &    RESULT(status) BIND(C)
            IMPORT :: C_PTR, C_INT, C_FLOAT
            INTEGER(C_INT) :: status
            TYPE(C_PTR) :: jitter
            INTEGER(C_INT), VALUE :: layout, transa, transb
            INTEGER(C_INT), VALUE :: m, n, k, lda, ldb, ldc
            REAL(C_FLOAT), VALUE :: alpha, beta
          END FUNCTION
          FUNCTION mkl_jit_get_sgemm_ptr(jitter)                        &
     &    RESULT(ptr) BIND(C)
            IMPORT :: C_FUNPTR, C_PTR
            TYPE(C_FUNPTR) :: ptr
            TYPE(C_PTR), INTENT(IN), VALUE :: jitter
          END FUNCTION
        END INTERFACE
#endif
#if defined(__LIBXSMM)
! LIBXSMM dispatch
        INTERFACE
          FUNCTION libxsmm_dispatch_gemm(shape, flags, prefetch)        &
     &    RESULT(fn) BIND(C)
            IMPORT :: C_FUNPTR, C_INT
            INTEGER(C_INT), INTENT(IN) :: shape(10)
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
            TYPE(C_FUNPTR) :: fn
          END FUNCTION
        END INTERFACE
#endif

! BLAS GEMM (dgemm_, sgemm_)
        INTERFACE
          SUBROUTINE dgemm_blas(transa, transb, m, n, k,                &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="dgemm_")
            IMPORT :: C_DOUBLE, C_INT, C_CHAR
            CHARACTER(1, C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta, a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
          END SUBROUTINE
          SUBROUTINE sgemm_blas(transa, transb, m, n, k,                &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="sgemm_")
            IMPORT :: C_FLOAT, C_INT, C_CHAR
            CHARACTER(1, C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta, a(*), b(*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(*)
          END SUBROUTINE
        END INTERFACE

        PUBLIC :: libxs_gemm_dispatch
        PUBLIC :: libxs_syrk_dispatch
        PUBLIC :: libxs_syr2k_dispatch

      CONTAINS
        INCLUDE 'libxs_procs.fi'

        !> Dispatch a GEMM kernel with auto-detected backends.
        FUNCTION libxs_gemm_dispatch(config,                            &
     &  datatype, transa, transb,                                       &
     &  m, n, k, lda, ldb, ldc, alpha, beta, registry)
          TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
          INTEGER(C_INT), INTENT(IN) :: datatype
          CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
          INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          INTEGER(C_INT) :: libxs_gemm_dispatch
          libxs_gemm_dispatch = libxs_gemm_dispatch_base(config,        &
     &      datatype, transa, transb,                                   &
     &      m, n, k, lda, ldb, ldc, alpha, beta,                        &
#if defined(__MKL)
     &      jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),      &
     &      jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),              &
     &      jit_create_sgemm=C_FUNLOC(mkl_cblas_jit_create_sgemm),      &
     &      jit_get_sgemm=C_FUNLOC(mkl_jit_get_sgemm_ptr),              &
#endif
#if defined(__LIBXSMM)
     &      xgemm_dispatch=C_FUNLOC(libxsmm_dispatch_gemm),             &
#endif
     &      dgemm_blas=C_FUNLOC(dgemm_blas),                            &
     &      sgemm_blas=C_FUNLOC(sgemm_blas),                            &
     &      registry=registry)
        END FUNCTION


        !> Dispatch a SYR2K config with auto-detected backends.
        FUNCTION libxs_syr2k_dispatch(datatype,                         &
     &  n, k, lda, ldb, ldc, registry)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k
          INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          TYPE(C_PTR) :: libxs_syr2k_dispatch
          libxs_syr2k_dispatch = libxs_syr2k_dispatch_base(datatype,    &
     &      n, k, lda, ldb, ldc,                                        &
#if defined(__MKL)
     &      jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),      &
     &      jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),              &
     &      jit_create_sgemm=C_FUNLOC(mkl_cblas_jit_create_sgemm),      &
     &      jit_get_sgemm=C_FUNLOC(mkl_jit_get_sgemm_ptr),              &
#endif
#if defined(__LIBXSMM)
     &      xgemm_dispatch=C_FUNLOC(libxsmm_dispatch_gemm),             &
#endif
     &      dgemm_blas=C_FUNLOC(dgemm_blas),                            &
     &      sgemm_blas=C_FUNLOC(sgemm_blas),                            &
     &      registry=registry)
        END FUNCTION


        !> Dispatch a SYRK config with auto-detected backends.
        FUNCTION libxs_syrk_dispatch(datatype,                          &
     &  n, k, lda, ldc, registry)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k
          INTEGER(C_INT), INTENT(IN) :: lda, ldc
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          TYPE(C_PTR) :: libxs_syrk_dispatch
          libxs_syrk_dispatch = libxs_syr2k_dispatch(datatype,          &
     &      n, k, lda, lda, ldc, registry)
        END FUNCTION
      END MODULE
