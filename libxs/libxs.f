!=======================================================================!
! Copyright (c) 2009-2026 Hans Pabst                                    !
! Copyright (c) 2009-2026 Intel Corporation                             !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      MODULE LIBXS
        INCLUDE 'libxs_spec.fi'
        PUBLIC :: libxs_gemm_dispatch, libxs_gemm_dispatch_ptr
        PUBLIC :: libxs_gemm_backend_kind
        PUBLIC :: libxs_syrk_dispatch
        PUBLIC :: libxs_tas_dispatch
        INTERFACE libxs_gemm_dispatch
          MODULE PROCEDURE libxs_gemm_dispatch_base
          MODULE PROCEDURE libxs_gemm_dispatch_ptr_base
        END INTERFACE
        INTERFACE libxs_gemm_dispatch_ptr
          MODULE PROCEDURE libxs_gemm_dispatch_ptr_base
        END INTERFACE
        INTERFACE libxs_gemm_backend_kind
          MODULE PROCEDURE libxs_gemm_backend_kind_shape
          MODULE PROCEDURE libxs_gemm_backend_kind_any
        END INTERFACE
        INTERFACE libxs_syrk_dispatch
          MODULE PROCEDURE libxs_syrk_dispatch_base
        END INTERFACE
        INTERFACE libxs_syrk
          MODULE PROCEDURE libxs_syrk_plan
          MODULE PROCEDURE libxs_syr2k_plan
          MODULE PROCEDURE libxs_syrk_direct
          MODULE PROCEDURE libxs_syr2k_direct
        END INTERFACE
        INTERFACE libxs_tas_dispatch
          MODULE PROCEDURE libxs_tas_dispatch_base
        END INTERFACE
      CONTAINS
        INCLUDE 'libxs_procs.fi'

        !> SYRK in one step, the arguments in the order of BLAS.
        SUBROUTINE libxs_syrk_direct(datatype, uplo, trans,             &
     &  n, k, alpha, a, lda, beta, c, ldc, info)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k, lda, ldc
          CHARACTER(C_CHAR), INTENT(IN), VALUE :: uplo, trans
          REAL(C_DOUBLE), INTENT(IN), VALUE :: alpha, beta
          TYPE(C_PTR), INTENT(IN), VALUE :: a
          TYPE(C_PTR), VALUE :: c
          INTEGER(C_INT), INTENT(OUT), OPTIONAL :: info
          TYPE(libxs_syrk_t) :: plan
          IF (0 .NE. libxs_syrk_dispatch_base(plan, datatype,           &
     &      uplo, trans, n, k, lda, 0_C_INT, ldc)) THEN
            CALL internal_syrk_run(plan,                                &
     &        alpha, a, C_NULL_PTR, beta, c, info)
          ELSE
            CALL internal_syrk_status(1_C_INT, info)
          END IF
        END SUBROUTINE

        !> SYR2K in one step, the arguments in the order of BLAS.
        SUBROUTINE libxs_syr2k_direct(datatype, uplo, trans,            &
     &  n, k, alpha, a, lda, b, ldb, beta, c, ldc, info)
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k, lda, ldb, ldc
          CHARACTER(C_CHAR), INTENT(IN), VALUE :: uplo, trans
          REAL(C_DOUBLE), INTENT(IN), VALUE :: alpha, beta
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b
          TYPE(C_PTR), VALUE :: c
          INTEGER(C_INT), INTENT(OUT), OPTIONAL :: info
          TYPE(libxs_syrk_t) :: plan
          ! ldb=0 would select SYRK, BLAS however demands at least one
          IF (0 .LT. ldb) THEN
            IF (0 .NE. libxs_syrk_dispatch_base(plan, datatype,         &
     &        uplo, trans, n, k, lda, ldb, ldc)) THEN
              CALL internal_syrk_run(plan, alpha, a, b, beta, c, info)
            ELSE
              CALL internal_syrk_status(1_C_INT, info)
            END IF
          ELSE
            CALL internal_syrk_status(1_C_INT, info)
          END IF
        END SUBROUTINE
      END MODULE
