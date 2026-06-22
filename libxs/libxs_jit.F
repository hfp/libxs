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
#if defined(__LIBXS) || defined(LIBXS_BUILD)
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
          CHARACTER(C_CHAR), INTENT(IN), VALUE :: transa, transb
          INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          INTEGER(C_INT) :: libxs_gemm_dispatch
          TYPE(C_FUNPTR) :: dgemm_fp, sgemm_fp
#if defined(__MKL)
          TYPE(C_FUNPTR) :: jcd_fp, jgd_fp, jcs_fp, jgs_fp
#endif
#if defined(__LIBXSMM)
          TYPE(C_FUNPTR) :: xdisp_fp
#endif
          dgemm_fp = C_FUNLOC(dgemm_blas)
          sgemm_fp = C_FUNLOC(sgemm_blas)
#if defined(__MKL)
          jcd_fp = C_FUNLOC(mkl_cblas_jit_create_dgemm)
          jgd_fp = C_FUNLOC(mkl_jit_get_dgemm_ptr)
          jcs_fp = C_FUNLOC(mkl_cblas_jit_create_sgemm)
          jgs_fp = C_FUNLOC(mkl_jit_get_sgemm_ptr)
#endif
#if defined(__LIBXSMM)
          xdisp_fp = C_FUNLOC(libxsmm_dispatch_gemm)
#endif
          libxs_gemm_dispatch = libxs_gemm_dispatch_base(config,        &
     &      datatype, transa, transb,                                   &
     &      m, n, k, lda, ldb, ldc, alpha, beta,                        &
#if defined(__MKL)
     &      jit_create_dgemm=jcd_fp,                                    &
     &      jit_get_dgemm=jgd_fp,                                       &
     &      jit_create_sgemm=jcs_fp,                                    &
     &      jit_get_sgemm=jgs_fp,                                       &
#endif
#if defined(__LIBXSMM)
     &      xgemm_dispatch=xdisp_fp,                                    &
#endif
     &      dgemm_blas=dgemm_fp,                                        &
     &      sgemm_blas=sgemm_fp,                                        &
     &      registry=registry)
        END FUNCTION


        !> Dispatch a SYR2K config with auto-detected backends.
        FUNCTION libxs_syr2k_dispatch(datatype,                         &
     &  n, k, lda, ldb, ldc, registry)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k
          INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          TYPE(C_PTR) :: libxs_syr2k_dispatch
          TYPE(C_FUNPTR) :: dgemm_fp, sgemm_fp
#if defined(__MKL)
          TYPE(C_FUNPTR) :: jcd_fp, jgd_fp, jcs_fp, jgs_fp
#endif
#if defined(__LIBXSMM)
          TYPE(C_FUNPTR) :: xdisp_fp
#endif
          dgemm_fp = C_FUNLOC(dgemm_blas)
          sgemm_fp = C_FUNLOC(sgemm_blas)
#if defined(__MKL)
          jcd_fp = C_FUNLOC(mkl_cblas_jit_create_dgemm)
          jgd_fp = C_FUNLOC(mkl_jit_get_dgemm_ptr)
          jcs_fp = C_FUNLOC(mkl_cblas_jit_create_sgemm)
          jgs_fp = C_FUNLOC(mkl_jit_get_sgemm_ptr)
#endif
#if defined(__LIBXSMM)
          xdisp_fp = C_FUNLOC(libxsmm_dispatch_gemm)
#endif
          libxs_syr2k_dispatch = libxs_syr2k_dispatch_base(datatype,    &
     &      n, k, lda, ldb, ldc,                                        &
#if defined(__MKL)
     &      jit_create_dgemm=jcd_fp,                                    &
     &      jit_get_dgemm=jgd_fp,                                       &
     &      jit_create_sgemm=jcs_fp,                                    &
     &      jit_get_sgemm=jgs_fp,                                       &
#endif
#if defined(__LIBXSMM)
     &      xgemm_dispatch=xdisp_fp,                                    &
#endif
     &      dgemm_blas=dgemm_fp,                                        &
     &      sgemm_blas=sgemm_fp,                                        &
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

#else
! Fallback: BLAS-only (no LIBXS library required)
        INCLUDE 'libxs_jit_spec.fi'

      CONTAINS
        SUBROUTINE libxs_init()
        END SUBROUTINE

        SUBROUTINE libxs_finalize()
        END SUBROUTINE

        SUBROUTINE libxs_gemm_call_config(config, a, b, c)
          TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b
          TYPE(C_PTR), VALUE :: c
          REAL(C_DOUBLE), POINTER :: pa(:), pb(:), pc(:)
          CALL C_F_POINTER(a, pa,                                       &
     &      [config%shape%lda * config%shape%k])
          CALL C_F_POINTER(b, pb,                                       &
     &      [config%shape%ldb * config%shape%k])
          CALL C_F_POINTER(c, pc,                                       &
     &      [config%shape%ldc * config%shape%n])
          CALL DGEMM(config%shape%transa, config%shape%transb,          &
     &      config%shape%m, config%shape%n, config%shape%k,             &
     &      config%shape%alpha, pa, config%shape%lda,                   &
     &      pb, config%shape%ldb,                                       &
     &      config%shape%beta, pc, config%shape%ldc)
        END SUBROUTINE

        SUBROUTINE libxs_gemm_call_direct(datatype,                     &
     &  transa, transb, m, n, k, lda, ldb, ldc,                         &
     &  alpha, beta, a, b, c)
          INTEGER(C_INT), INTENT(IN) :: datatype
          CHARACTER, INTENT(IN), VALUE :: transa, transb
          INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b
          TYPE(C_PTR), VALUE :: c
          REAL(C_DOUBLE), POINTER :: pa(:), pb(:), pc(:)
          CALL C_F_POINTER(a, pa, [lda * k])
          CALL C_F_POINTER(b, pb, [ldb * k])
          CALL C_F_POINTER(c, pc, [ldc * n])
          CALL DGEMM(transa, transb, m, n, k,                           &
     &      alpha, pa, lda, pb, ldb, beta, pc, ldc)
        END SUBROUTINE

        SUBROUTINE libxs_syr2k(datatype, n, k, lda, ldb, ldc,           &
     &  uplo, alpha, beta, a, b, c)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k, lda, ldb, ldc
          CHARACTER, INTENT(IN), VALUE :: uplo
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b
          TYPE(C_PTR), VALUE :: c
          REAL(C_DOUBLE), POINTER :: pa(:), pb(:), pc(:)
          CALL C_F_POINTER(a, pa, [lda * k])
          CALL C_F_POINTER(b, pb, [ldb * k])
          CALL C_F_POINTER(c, pc, [ldc * n])
          CALL DSYR2K(uplo, 'N', n, k,                                  &
     &      alpha, pa, lda, pb, ldb, beta, pc, ldc)
        END SUBROUTINE

        SUBROUTINE libxs_syrk(datatype, n, k, lda, ldc,                 &
     &  uplo, alpha, beta, a, c)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k, lda, ldc
          CHARACTER, INTENT(IN), VALUE :: uplo
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN), VALUE :: a
          TYPE(C_PTR), VALUE :: c
          REAL(C_DOUBLE), POINTER :: pa(:), pc(:)
          CALL C_F_POINTER(a, pa, [lda * k])
          CALL C_F_POINTER(c, pc, [ldc * n])
          CALL DSYRK(uplo, 'N', n, k,                                   &
     &      alpha, pa, lda, beta, pc, ldc)
        END SUBROUTINE
#endif
      END MODULE
