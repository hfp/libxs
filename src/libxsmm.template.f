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
     &                                    C_FUNPTR, C_F_PROCPOINTER,    &
     &                                    C_PTR, C_NULL_PTR, C_LOC,     &
     &                                    C_INT, C_FLOAT, C_DOUBLE,     &
     &                                    C_LONG_LONG, C_CHAR
        IMPLICIT NONE
        PRIVATE :: libxs_srealptr, libxs_drealptr

        CHARACTER(*), PARAMETER :: LIBXS_VERSION = "$VERSION"
        CHARACTER(*), PARAMETER :: LIBXS_BRANCH  = "$BRANCH"
        INTEGER(C_INT), PARAMETER :: LIBXS_VERSION_MAJOR  = $MAJOR
        INTEGER(C_INT), PARAMETER :: LIBXS_VERSION_MINOR  = $MINOR
        INTEGER(C_INT), PARAMETER :: LIBXS_VERSION_UPDATE = $UPDATE
        INTEGER(C_INT), PARAMETER :: LIBXS_VERSION_PATCH  = $PATCH

        ! Parameters the library and static kernels were built for.
        INTEGER(C_INT), PARAMETER :: LIBXS_ALIGNMENT = $ALIGNMENT
        INTEGER(C_INT), PARAMETER :: LIBXS_ROW_MAJOR = $ROW_MAJOR
        INTEGER(C_INT), PARAMETER :: LIBXS_COL_MAJOR = $COL_MAJOR
        INTEGER(C_INT), PARAMETER :: LIBXS_PREFETCH = $PREFETCH
        INTEGER(C_INT), PARAMETER :: LIBXS_MAX_MNK = $MAX_MNK
        INTEGER(C_INT), PARAMETER :: LIBXS_MAX_M = $MAX_M
        INTEGER(C_INT), PARAMETER :: LIBXS_MAX_N = $MAX_N
        INTEGER(C_INT), PARAMETER :: LIBXS_MAX_K = $MAX_K
        INTEGER(C_INT), PARAMETER :: LIBXS_AVG_M = $AVG_M
        INTEGER(C_INT), PARAMETER :: LIBXS_AVG_N = $AVG_N
        INTEGER(C_INT), PARAMETER :: LIBXS_AVG_K = $AVG_K
        INTEGER(C_INT), PARAMETER :: LIBXS_FLAGS = $FLAGS
        INTEGER(C_INT), PARAMETER :: LIBXS_ILP64 = $ILP64
        INTEGER(C_INT), PARAMETER :: LIBXS_JIT = $JIT

        ! Integer type impacting the BLAS interface (LP64: 32-bit, and ILP64: 64-bit).
        INTEGER(C_INT), PARAMETER :: LIBXS_BLASINT_KIND =             &
     &    MERGE(C_INT, C_LONG_LONG, 0.EQ.LIBXS_ILP64)

        ! Parameters representing the GEMM performed by the simplified interface.
        REAL(C_DOUBLE), PARAMETER :: LIBXS_ALPHA = $ALPHA
        REAL(C_DOUBLE), PARAMETER :: LIBXS_BETA = $BETA

        ! Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXS_GEMM_FLAG_TRANS_A = 1,                                &
     &    LIBXS_GEMM_FLAG_TRANS_B = 2,                                &
     &    LIBXS_GEMM_FLAG_ALIGN_A = 4,                                &
     &    LIBXS_GEMM_FLAG_ALIGN_C = 8

        ! Enumeration of the available prefetch strategies which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! No prefetching and no prefetch function signature.
     &    LIBXS_PREFETCH_NONE       = 0,                              &
          ! Only function prefetch signature.
     &    LIBXS_PREFETCH_SIGNATURE  = 1,                              &
          ! Prefetch PA using accesses to A.
     &    LIBXS_PREFETCH_AL2        = 2,                              &
          ! Prefetch PA (aggressive).
     &    LIBXS_PREFETCH_AL2_JPST   = 4,                              &
          ! Prefetch PB using accesses to C.
     &    LIBXS_PREFETCH_BL2_VIA_C  = 8,                              &
          ! Prefetch A ahead.
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
          PURE SUBROUTINE LIBXS_FUNCTION0(a, b, c) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments.
          PURE SUBROUTINE LIBXS_FUNCTION1(a, b, c,                    &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function (single-precision).
        TYPE :: LIBXS_SMM_FUNCTION
          PRIVATE
            PROCEDURE(LIBXS_FUNCTION0), NOPASS, POINTER :: fn0
            PROCEDURE(LIBXS_FUNCTION1), NOPASS, POINTER :: fn1
        END TYPE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function (double-precision).
        TYPE :: LIBXS_DMM_FUNCTION
          PRIVATE
            PROCEDURE(LIBXS_FUNCTION0), NOPASS, POINTER :: fn0
            PROCEDURE(LIBXS_FUNCTION1), NOPASS, POINTER :: fn1
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxs_dispatch
          MODULE PROCEDURE libxs_sdispatch, libxs_ddispatch
        END INTERFACE

        ! Check if a function (LIBXS_?MM_FUNCTION_TYPE) is available.
        INTERFACE libxs_available
          MODULE PROCEDURE libxs_savailable, libxs_davailable
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxs_call
          MODULE PROCEDURE libxs_scall_abx, libxs_scall_prx
          MODULE PROCEDURE libxs_scall_abc, libxs_scall_prf
          MODULE PROCEDURE libxs_dcall_abx, libxs_dcall_prx
          MODULE PROCEDURE libxs_dcall_abc, libxs_dcall_prf
        END INTERFACE

        ! Overloaded GEMM routines (single/double precision).
        INTERFACE libxs_gemm
          MODULE PROCEDURE libxs_sgemm, libxs_dgemm
        END INTERFACE

        ! Overloaded BLAS GEMM routines (single/double precision).
        INTERFACE libxs_blas_gemm
          MODULE PROCEDURE libxs_blas_sgemm, libxs_blas_dgemm
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_init, libxs_finalize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_timer_tick
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_timer_duration
        INTERFACE
          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxs_init() BIND(C)
          END SUBROUTINE

          SUBROUTINE libxs_finalize() BIND(C)
          END SUBROUTINE

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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_srealptr
        FUNCTION libxs_srealptr(a) RESULT(p)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:)
          REAL(C_FLOAT), POINTER :: p
          p => a(LBOUND(a,1),LBOUND(a,2))
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_drealptr
        FUNCTION libxs_drealptr(a) RESULT(p)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:)
          REAL(C_DOUBLE), POINTER :: p
          p => a(LBOUND(a,1),LBOUND(a,2))
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ld
        INTEGER(C_INT) PURE FUNCTION libxs_ld(m, n)
          INTEGER(C_INT), INTENT(IN) :: m, n
          libxs_ld = MERGE(m, n, 0.NE.LIBXS_COL_MAJOR)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_align_value
        INTEGER(C_INT) PURE FUNCTION libxs_align_value(               &
     &    n, typesize, alignment)
          INTEGER(C_INT), INTENT(IN) :: n, typesize
          INTEGER(C_INT), INTENT(IN) :: alignment
          libxs_align_value = (((n * typesize + alignment - 1) /      &
     &      alignment) * alignment) / typesize
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sfunction
        TYPE(LIBXS_SMM_FUNCTION) FUNCTION libxs_sfunction(          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          PROCEDURE(LIBXS_FUNCTION0), POINTER :: fn0
          PROCEDURE(LIBXS_FUNCTION1), POINTER :: fn1
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: sdispatch
          INTERFACE
            TYPE(C_FUNPTR) PURE FUNCTION sdispatch(                     &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)       &
     &      BIND(C, NAME="libxs_sdispatch")
              IMPORT :: C_FUNPTR, C_INT, C_FLOAT
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
              INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              INTEGER(C_INT), INTENT(IN) :: flags, prefetch
            END FUNCTION
          END INTERFACE
          IF (.NOT.PRESENT(prefetch)) THEN
            CALL C_F_PROCPOINTER(sdispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn0)
            libxs_sfunction%fn0 => fn0
            libxs_sfunction%fn1 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(sdispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn1)
            libxs_sfunction%fn0 => NULL()
            libxs_sfunction%fn1 => fn1
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dfunction
        TYPE(LIBXS_DMM_FUNCTION) FUNCTION libxs_dfunction(          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          PROCEDURE(LIBXS_FUNCTION0), POINTER :: fn0
          PROCEDURE(LIBXS_FUNCTION1), POINTER :: fn1
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: ddispatch
          INTERFACE
            TYPE(C_FUNPTR) PURE FUNCTION ddispatch(                     &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)       &
     &      BIND(C, NAME="libxs_ddispatch")
              IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
              INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              INTEGER(C_INT), INTENT(IN) :: flags, prefetch
            END FUNCTION
          END INTERFACE
          IF (.NOT.PRESENT(prefetch)) THEN
            CALL C_F_PROCPOINTER(ddispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn0)
            libxs_dfunction%fn0 => fn0
            libxs_dfunction%fn1 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(ddispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn1)
            libxs_dfunction%fn0 => NULL()
            libxs_dfunction%fn1 => fn1
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sdispatch
        SUBROUTINE libxs_sdispatch(fn,                                &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          fn = libxs_sfunction(                                       &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ddispatch
        SUBROUTINE libxs_ddispatch(fn,                                &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          fn = libxs_dfunction(                                       &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_savailable
        LOGICAL PURE FUNCTION libxs_savailable(fn)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          libxs_savailable = ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_davailable
        LOGICAL PURE FUNCTION libxs_davailable(fn)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          libxs_davailable = ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
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
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          CALL libxs_scall_abx(fn,                                    &
     &      C_LOC(libxs_srealptr(a)),                                 &
     &      C_LOC(libxs_srealptr(b)),                                 &
     &      C_LOC(libxs_srealptr(c)))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall_abc
        SUBROUTINE libxs_dcall_abc(fn, a, b, c)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          CALL libxs_dcall_abx(fn,                                    &
     &      C_LOC(libxs_drealptr(a)),                                 &
     &      C_LOC(libxs_drealptr(b)),                                 &
     &      C_LOC(libxs_drealptr(c)))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_scall_prf
        SUBROUTINE libxs_scall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          REAL(C_FLOAT), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxs_scall_prx(fn,                                    &
     &      C_LOC(libxs_srealptr(a)),                                 &
     &      C_LOC(libxs_srealptr(b)),                                 &
     &      C_LOC(libxs_srealptr(c)),                                 &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall_prf
        SUBROUTINE libxs_dcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxs_dcall_prx(fn,                                    &
     &      C_LOC(libxs_drealptr(a)),                                 &
     &      C_LOC(libxs_drealptr(b)),                                 &
     &      C_LOC(libxs_drealptr(c)),                                 &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_scall
        SUBROUTINE libxs_scall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXS_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pa(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pb(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pc(*)
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            CALL libxs_scall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),    &
     &        MERGE(C_NULL_PTR, C_LOC(pa), .NOT.PRESENT(pa)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pb), .NOT.PRESENT(pb)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pc), .NOT.PRESENT(pc)))
          ELSE
            CALL libxs_scall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dcall
        SUBROUTINE libxs_dcall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXS_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pc(*)
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            CALL libxs_dcall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),    &
     &        MERGE(C_NULL_PTR, C_LOC(pa), .NOT.PRESENT(pa)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pb), .NOT.PRESENT(pb)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pc), .NOT.PRESENT(pc)))
          ELSE
            CALL libxs_dcall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sgemm
        SUBROUTINE libxs_sgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxs_sgemm")
              IMPORT LIBXS_BLASINT_KIND, C_CHAR, C_FLOAT
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dgemm
        SUBROUTINE libxs_dgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxs_dgemm")
              IMPORT LIBXS_BLASINT_KIND, C_CHAR, C_DOUBLE
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_blas_sgemm
        SUBROUTINE libxs_blas_sgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxs_blas_sgemm")
              IMPORT LIBXS_BLASINT_KIND, C_CHAR, C_FLOAT
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_blas_dgemm
        SUBROUTINE libxs_blas_dgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXS_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxs_blas_dgemm")
              IMPORT LIBXS_BLASINT_KIND, C_CHAR, C_DOUBLE
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXS_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE
      END MODULE
