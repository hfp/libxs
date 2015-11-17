!*****************************************************************************!
!* Copyright (c) 2013-2015, Intel Corporation                                *!
!* All rights reserved.                                                      *!
!*                                                                           *!
!* Redistribution and use in source and binary forms, with or without        *!
!* modification, are permitted provided that the following conditions        *!
!* are met:                                                                  *!
!* 1. Redistributions of source code must retain the above copyright         *!
!*    notice, this list of conditions and the following disclaimer.          *!
!* 2. Redistributions in binary form must reproduce the above copyright      *!
!*    notice, this list of conditions and the following disclaimer in the    *!
!*    documentation and/or other materials provided with the distribution.   *!
!* 3. Neither the name of the copyright holder nor the names of its          *!
!*    contributors may be used to endorse or promote products derived        *!
!*    from this software without specific prior written permission.          *!
!*                                                                           *!
!* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *!
!* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         *!
!* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     *!
!* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      *!
!* HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    *!
!* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  *!
!* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    *!
!* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *!
!* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *!
!* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *!
!* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *!
!*****************************************************************************!
!* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.)                *!
!*****************************************************************************!

      MODULE LIBXS
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &                      C_F_PROCPOINTER, C_FUNPTR, C_LOC, C_PTR,    &
     &                      C_INT, C_FLOAT, C_DOUBLE, C_LONG_LONG
        IMPLICIT NONE

        ! Kind of types used to parameterize the implementation.
        INTEGER, PARAMETER ::                                           &
        ! Single-precision (FLS_KIND), and double-precision (FLD_KIND)
     &    LIBXS_FLS_KIND = SELECTED_REAL_KIND( 6,  30),               &
     &    LIBXS_FLD_KIND = SELECTED_REAL_KIND(14, 200),               &
        ! LP64: 32-bit with integers, and ILP64: 64-bit integers
     &    LIBXS_INT_KIND = $INTEGER_TYPE

        ! Parameters the library and static kernels were built for.
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_ALIGNMENT = $ALIGNMENT
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_ROW_MAJOR = $ROW_MAJOR
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_COL_MAJOR = $COL_MAJOR
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_PREFETCH = $PREFETCH
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_MAX_MNK = $MAX_MNK
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_MAX_M = $MAX_M
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_MAX_N = $MAX_N
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_MAX_K = $MAX_K
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_AVG_M = $AVG_M
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_AVG_N = $AVG_N
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_AVG_K = $AVG_K
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_FLAGS = $FLAGS
        INTEGER(LIBXS_INT_KIND), PARAMETER :: LIBXS_JIT = $JIT

        ! Parameters representing the GEMM performed by the simplified interface.
        REAL(LIBXS_FLD_KIND), PARAMETER ::                            &
     &    LIBXS_ALPHA = $ALPHA, LIBXS_BETA = $BETA

        ! Flag enumeration which can be IORed.
        INTEGER(LIBXS_INT_KIND), PARAMETER ::                         &
     &    LIBXS_GEMM_FLAG_TRANS_A = 1,                                &
     &    LIBXS_GEMM_FLAG_TRANS_B = 2,                                &
     &    LIBXS_GEMM_FLAG_ALIGN_A = 4,                                &
     &    LIBXS_GEMM_FLAG_ALIGN_C = 8

        ! Enumeration of the available prefetch strategies which can be IORed.
        !   LIBXS_PREFETCH_NONE:      No prefetching and no prefetch fn. signature.
        !   LIBXS_PREFETCH_SIGNATURE: Only function prefetch signature.
        !   LIBXS_PREFETCH_AL2:       Prefetch PA using accesses to A.
        !   LIBXS_PREFETCH_AL2_JPST:  Prefetch PA (aggressive).
        !   LIBXS_PREFETCH_BL2_VIA_C: Prefetch PB using accesses to C.
        !   LIBXS_PREFETCH_AL2_AHEAD: Prefetch A ahead.
        INTEGER(LIBXS_INT_KIND), PARAMETER ::                         &
     &    LIBXS_PREFETCH_NONE       = 0,                              &
     &    LIBXS_PREFETCH_SIGNATURE  = 1,                              &
     &    LIBXS_PREFETCH_AL2        = 2,                              &
     &    LIBXS_PREFETCH_AL2_JPST   = 4,                              &
     &    LIBXS_PREFETCH_BL2_VIA_C  = 8,                              &
     &    LIBXS_PREFETCH_AL2_AHEAD  = 16,                             &
          ! Composed prefetch strategies.
     &    LIBXS_PREFETCH_AL2BL2_VIA_C = IOR(                          &
     &        LIBXS_PREFETCH_BL2_VIA_C, LIBXS_PREFETCH_AL2),        &
     &    LIBXS_PREFETCH_AL2BL2_VIA_C_JPST = IOR(                     &
     &        LIBXS_PREFETCH_BL2_VIA_C, LIBXS_PREFETCH_AL2_JPST),   &
     &    LIBXS_PREFETCH_AL2BL2_VIA_C_AHEAD = IOR(                    &
     &        LIBXS_PREFETCH_BL2_VIA_C, LIBXS_PREFETCH_AL2_AHEAD)

        ! Type of a function specialized for a given parameter set.
        ABSTRACT INTERFACE
          ! Specialized function with fused alpha and beta arguments.
          PURE SUBROUTINE LIBXS_FUNCTION(a, b, c) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments.
          PURE SUBROUTINE LIBXS_XFUNCTION(a, b, c,                    &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXS_SMM_FUNCTION
          PROCEDURE(LIBXS_FUNCTION), NOPASS, POINTER ::               &
     &      fn0 => NULL()
          PROCEDURE(LIBXS_XFUNCTION), NOPASS, POINTER ::              &
     &      fn1 => NULL()
        END TYPE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXS_DMM_FUNCTION
          PROCEDURE(LIBXS_FUNCTION), NOPASS, POINTER ::               &
     &      fn0 => NULL()
          PROCEDURE(LIBXS_XFUNCTION), NOPASS, POINTER ::              &
     &      fn1 => NULL()
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxs_sdispatch
          MODULE PROCEDURE                                              &
     &      libxs_sfunction0, libxs_sfunction1
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxs_ddispatch
          MODULE PROCEDURE                                              &
     &      libxs_dfunction0, libxs_dfunction1
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxs_dispatch
          MODULE PROCEDURE                                              &
     &      libxs_sdispatch_mnk, libxs_ddispatch_mnk,               &
     &      libxs_sdispatch_ldx, libxs_ddispatch_ldx,               &
     &      libxs_sdispatch_abf, libxs_ddispatch_abf,               &
     &      libxs_sdispatch_all, libxs_ddispatch_all
        END INTERFACE

        ! Check if a function (LIBXS_?MM_FUNCTION_TYPE) is available.
        INTERFACE libxs_available
          MODULE PROCEDURE libxs_savailable, libxs_davailable
        END INTERFACE

        ! Call a specialized function (single-precision).
        INTERFACE libxs_scall
          MODULE PROCEDURE                                              &
     &      libxs_scall_abx, libxs_scall_abc,                       &
     &      libxs_scall_prx, libxs_scall_prf
        END INTERFACE

        ! Call a specialized function (double-precision).
        INTERFACE libxs_dcall
          MODULE PROCEDURE                                              &
     &      libxs_dcall_abx, libxs_dcall_abc,                       &
     &      libxs_dcall_prx, libxs_dcall_prf
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxs_call
          MODULE PROCEDURE                                              &
     &      libxs_scall_abx, libxs_scall_abc,                       &
     &      libxs_scall_prx, libxs_scall_prf,                       &
     &      libxs_dcall_abx, libxs_dcall_abc,                       &
     &      libxs_dcall_prx, libxs_dcall_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines (single precision).
        INTERFACE libxs_smm
          MODULE PROCEDURE libxs_smm_abc, libxs_smm_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines (double precision).
        INTERFACE libxs_dmm
          MODULE PROCEDURE libxs_dmm_abc, libxs_dmm_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines.
        INTERFACE libxs_mm
          MODULE PROCEDURE                                              &
     &      libxs_smm_abc, libxs_smm_prf,                           &
     &      libxs_dmm_abc, libxs_dmm_prf
        END INTERFACE

        ! Overloaded BLAS routines (single/double precision).
        INTERFACE libxs_blasmm
          MODULE PROCEDURE                                              &
     &      libxs_sblasmm, libxs_dblasmm,                           &
     &      libxs_sblasmm_abf, libxs_dblasmm_abf
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_init
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch0, libxs_ddispatch0
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch1, libxs_ddispatch1
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_timer_tick, libxs_timer_duration
        INTERFACE
          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxs_init() BIND(C)
          END SUBROUTINE

          SUBROUTINE libxs_finalize() BIND(C)
          END SUBROUTINE

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxs_sdispatch0(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta)                   &
     &    BIND(C, NAME="libxs_sdispatch")
            IMPORT :: C_FUNPTR, C_INT, C_FLOAT
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxs_ddispatch0(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta)                   &
     &    BIND(C, NAME="libxs_ddispatch")
            IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxs_sdispatch1(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)         &
     &    BIND(C, NAME="libxs_sxdispatch")
            IMPORT :: C_FUNPTR, C_INT, C_FLOAT
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
            INTEGER(C_INT), INTENT(IN), VALUE :: prefetch
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxs_ddispatch1(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)         &
     &    BIND(C, NAME="libxs_dxdispatch")
            IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
            INTEGER(C_INT), INTENT(IN), VALUE :: prefetch
          END FUNCTION

          ! Non-pure function returning the current clock tick
          ! using a platform-specific resolution.
          INTEGER(C_LONG_LONG) FUNCTION libxs_timer_tick() BIND(C)
            IMPORT :: C_LONG_LONG
          END FUNCTION
          ! Non-pure function (timer freq. may vary) returning
          ! the duration between two ticks (seconds).
          REAL(C_DOUBLE) FUNCTION libxs_timer_duration(               &
     &    tick0, tick1) BIND(C)
            IMPORT :: C_LONG_LONG, C_DOUBLE
            INTEGER(C_LONG_LONG), INTENT(IN), VALUE :: tick0, tick1
          END FUNCTION
        END INTERFACE$MNK_INTERFACE_LIST

      CONTAINS
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sfunction0
        TYPE(LIBXS_SMM_FUNCTION) FUNCTION libxs_sfunction0(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          PROCEDURE(LIBXS_FUNCTION), POINTER :: function
          CALL C_F_PROCPOINTER(                                         &
     &      libxs_sdispatch0(flags, m, n, k,                          &
     &          MERGE(0, lda, .NOT.PRESENT(lda)),                       &
     &          MERGE(0, ldb, .NOT.PRESENT(ldb)),                       &
     &          MERGE(0, ldc, .NOT.PRESENT(ldc)),                       &
     &          MERGE(REAL(LIBXS_ALPHA, T), alpha,                    &
     &            .NOT.PRESENT(alpha)),                                 &
     &          MERGE(REAL(LIBXS_BETA, T), beta,                      &
     &            .NOT.PRESENT(beta))),                                 &
     &      function)
          libxs_sfunction0%fn0 => function
          libxs_sfunction0%fn1 => NULL()
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dfunction0
        TYPE(LIBXS_DMM_FUNCTION) FUNCTION libxs_dfunction0(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          PROCEDURE(LIBXS_FUNCTION), POINTER :: function
          CALL C_F_PROCPOINTER(                                         &
     &      libxs_ddispatch0(flags, m, n, k,                          &
     &          MERGE(0, lda, .NOT.PRESENT(lda)),                       &
     &          MERGE(0, ldb, .NOT.PRESENT(ldb)),                       &
     &          MERGE(0, ldc, .NOT.PRESENT(ldc)),                       &
     &          MERGE(REAL(LIBXS_ALPHA, T), alpha,                    &
     &            .NOT.PRESENT(alpha)),                                 &
     &          MERGE(REAL(LIBXS_BETA, T), beta,                      &
     &            .NOT.PRESENT(beta))),                                 &
     &      function)
          libxs_dfunction0%fn0 => function
          libxs_dfunction0%fn1 => NULL()
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sfunction1
        TYPE(LIBXS_SMM_FUNCTION) FUNCTION libxs_sfunction1(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: prefetch
          PROCEDURE(LIBXS_XFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXS_FUNCTION), POINTER :: fn0
          IF (LIBXS_PREFETCH_NONE.NE.prefetch) THEN
            CALL C_F_PROCPOINTER(libxs_sdispatch1(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch),    &
     &        fn1)
            libxs_sfunction1%fn1 => fn1
            libxs_sfunction1%fn0 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(libxs_sdispatch0(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta),              &
     &        fn0)
            libxs_sfunction1%fn0 => fn0
            libxs_sfunction1%fn1 => NULL()
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dfunction1
        TYPE(LIBXS_DMM_FUNCTION) FUNCTION libxs_dfunction1(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: prefetch
          PROCEDURE(LIBXS_XFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXS_FUNCTION), POINTER :: fn0
          IF (LIBXS_PREFETCH_NONE.NE.prefetch) THEN
            CALL C_F_PROCPOINTER(libxs_ddispatch1(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch),    &
     &        fn1)
            libxs_dfunction1%fn1 => fn1
            libxs_dfunction1%fn0 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(libxs_ddispatch0(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta),              &
     &        fn0)
            libxs_dfunction1%fn0 => fn0
            libxs_dfunction1%fn1 => NULL()
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch_mnk
        SUBROUTINE libxs_sdispatch_mnk(function,                      &
     &  m, n, k, alpha, beta, flags)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXS_FLS_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxs_sfunction0(                                &
     &      MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, alpha = alpha, beta = beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ddispatch_mnk
        SUBROUTINE libxs_ddispatch_mnk(function,                      &
     &  m, n, k, alpha, beta, flags)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXS_FLD_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxs_dfunction0(                                &
     &      MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, alpha = alpha, beta = beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch_ldx
        SUBROUTINE libxs_sdispatch_ldx(function,                      &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLS_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxs_sfunction0(                                &
     &      MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, lda, ldb, ldc, alpha, beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ddispatch_ldx
        SUBROUTINE libxs_ddispatch_ldx(function,                      &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLD_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxs_dfunction0(                                &
     &      MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, lda, ldb, ldc, alpha, beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch_abf
        SUBROUTINE libxs_sdispatch_abf(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLS_KIND), INTENT(IN) :: ralpha, rbeta
          function = libxs_sfunction0(                                &
     &      flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ddispatch_abf
        SUBROUTINE libxs_ddispatch_abf(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLD_KIND), INTENT(IN) :: ralpha, rbeta
          function = libxs_dfunction0(                                &
     &      flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch_all
        SUBROUTINE libxs_sdispatch_all(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: prefetch
          function = libxs_sfunction1(                                &
     &      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ddispatch_all
        SUBROUTINE libxs_ddispatch_all(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXS_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: prefetch
          function = libxs_dfunction1(                                &
     &      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_savailable
        LOGICAL PURE FUNCTION libxs_savailable(fn)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          libxs_savailable =                                          &
     &      ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_davailable
        LOGICAL PURE FUNCTION libxs_davailable(fn)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          libxs_davailable =                                          &
     &      ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_scall_abx
        PURE SUBROUTINE libxs_scall_abx(fn, a, b, c)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          CALL fn%fn0(a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall_abx
        PURE SUBROUTINE libxs_dcall_abx(fn, a, b, c)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          CALL fn%fn0(a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_scall_prx
        PURE SUBROUTINE libxs_scall_prx(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          CALL fn%fn1(a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall_prx
        PURE SUBROUTINE libxs_dcall_prx(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          CALL fn%fn1(a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_scall_abc
        SUBROUTINE libxs_scall_abc(fn, a, b, c)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          CALL libxs_scall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall_abc
        SUBROUTINE libxs_dcall_abc(fn, a, b, c)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          CALL libxs_dcall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_scall_prf
        SUBROUTINE libxs_scall_prf(fn, a, b, c, pa, pb, pc)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          REAL(T), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxs_scall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),      &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall_prf
        SUBROUTINE libxs_dcall_prf(fn, a, b, c, pa, pb, pc)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          REAL(T), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxs_dcall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),      &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ld
        INTEGER(LIBXS_INT_KIND) PURE FUNCTION libxs_ld(m, n)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n
          libxs_ld = MERGE(m, n, 0.NE.LIBXS_COL_MAJOR)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_align_value
        INTEGER(LIBXS_INT_KIND) PURE FUNCTION libxs_align_value(    &
     &    n, typesize, alignment)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: n, typesize
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: alignment
          libxs_align_value = (((n * typesize + alignment - 1) /      &
     &      alignment) * alignment) / typesize
        END FUNCTION

        ! Non-dispatched matrix multiplication using BLAS (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sblasmm
        SUBROUTINE libxs_sblasmm(m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXS_INT_KIND) :: f
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: sgemm
          INTERFACE
            SUBROUTINE sgemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
              IMPORT LIBXS_INT_KIND, LIBXS_FLS_KIND
              CHARACTER(1), INTENT(IN) :: transa, transb
              INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(LIBXS_FLS_KIND), INTENT(IN) :: alpha, beta
              REAL(LIBXS_FLS_KIND), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(LIBXS_FLS_KIND), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          f = MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags))
          IF (0.NE.LIBXS_COL_MAJOR) THEN
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_B, f)),             &
     &        m, n, k,                                                  &
     &        MERGE(REAL(LIBXS_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        a, MAX(SIZE(a, 1), m), b, MAX(SIZE(b, 1), k),             &
     &        MERGE(REAL(LIBXS_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), m))
          ELSE
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_B, f)),             &
     &        n, m, k,                                                  &
     &        MERGE(REAL(LIBXS_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        b, MAX(SIZE(b, 2), n), a, MAX(SIZE(a, 2), k),             &
     &        MERGE(REAL(LIBXS_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), n))
          END IF
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dblasmm
        SUBROUTINE libxs_dblasmm(m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXS_INT_KIND) :: f
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: dgemm
          INTERFACE
            SUBROUTINE dgemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
              IMPORT LIBXS_INT_KIND, LIBXS_FLD_KIND
              CHARACTER(1), INTENT(IN) :: transa, transb
              INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXS_INT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(LIBXS_FLD_KIND), INTENT(IN) :: alpha, beta
              REAL(LIBXS_FLD_KIND), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(LIBXS_FLD_KIND), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          f = MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags))
          IF (0.NE.LIBXS_COL_MAJOR) THEN
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_B, f)),             &
     &        m, n, k,                                                  &
     &        MERGE(REAL(LIBXS_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        a, MAX(SIZE(a, 1), m), b, MAX(SIZE(b, 1), k),             &
     &        MERGE(REAL(LIBXS_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), m))
          ELSE
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXS_GEMM_FLAG_TRANS_B, f)),             &
     &        n, m, k,                                                  &
     &        MERGE(REAL(LIBXS_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        b, MAX(SIZE(b, 2), n), a, MAX(SIZE(a, 2), k),             &
     &        MERGE(REAL(LIBXS_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), n))
          END IF
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sblasmm_abf
        SUBROUTINE libxs_sblasmm_abf(                                 &
     &  m, n, k, a, b, c, salpha, sbeta, flags)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: salpha, sbeta
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          CALL libxs_sblasmm(m, n, k, a, b, c,                        &
     &      MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      salpha, sbeta)
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dblasmm_abf
        SUBROUTINE libxs_dblasmm_abf(                                 &
     &  m, n, k, a, b, c, dalpha, dbeta, flags)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: dalpha, dbeta
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          CALL libxs_dblasmm(m, n, k, a, b, c,                        &
     &      MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      dalpha, dbeta)
        END SUBROUTINE

        ! Dispatched matrix multiplication (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_smm_abc
        SUBROUTINE libxs_smm_abc(                                     &
     &  m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXS_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXS_BETA, T)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXS_SMM_FUNCTION) :: function
          INTEGER(LIBXS_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
            function = libxs_sfunction0(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta)
            IF (ASSOCIATED(function%fn0)) THEN
              CALL libxs_scall_abc(function, a, b, c)
            ELSE
              CALL libxs_sblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxs_sblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dmm_abc
        SUBROUTINE libxs_dmm_abc(                                     &
     &  m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXS_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXS_BETA, T)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXS_DMM_FUNCTION) :: function
          INTEGER(LIBXS_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
            function = libxs_dfunction0(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta)
            IF (ASSOCIATED(function%fn0)) THEN
              CALL libxs_dcall_abc(function, a, b, c)
            ELSE
              CALL libxs_dblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxs_dblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication with prefetches (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_smm_prf
        SUBROUTINE libxs_smm_prf(                                     &
     &  m, n, k, a, b, c, pa, pb, pc, flags, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLS_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXS_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXS_BETA, T)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: pa(*), pb(*), pc(*)
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXS_SMM_FUNCTION) :: function
          INTEGER(LIBXS_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
            function = libxs_sfunction1(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta,                  &
     &        MERGE(LIBXS_PREFETCH, LIBXS_PREFETCH_SIGNATURE,       &
     &            LIBXS_PREFETCH_NONE.NE.LIBXS_PREFETCH))
            IF (ASSOCIATED(function%fn1)) THEN
              CALL libxs_scall_prf(function, a, b, c, pa, pb, pc)
            ELSE
              CALL libxs_sblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxs_sblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication with prefetches (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dmm_prf
        SUBROUTINE libxs_dmm_prf(                                     &
     &  m, n, k, a, b, c, pa, pb, pc, flags, alpha, beta)
          INTEGER(LIBXS_INT_KIND), PARAMETER :: T = LIBXS_FLD_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXS_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXS_BETA, T)
          INTEGER(LIBXS_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: pa(*), pb(*), pc(*)
          INTEGER(LIBXS_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXS_DMM_FUNCTION) :: function
          INTEGER(LIBXS_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXS_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
            function = libxs_dfunction1(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta,                  &
     &        MERGE(LIBXS_PREFETCH, LIBXS_PREFETCH_SIGNATURE,       &
     &            LIBXS_PREFETCH_NONE.NE.LIBXS_PREFETCH))
            IF (ASSOCIATED(function%fn1)) THEN
              CALL libxs_dcall_prf(function, a, b, c, pa, pb, pc)
            ELSE
              CALL libxs_dblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxs_dblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE
      END MODULE
