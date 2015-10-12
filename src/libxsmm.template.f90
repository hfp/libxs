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
  USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE

  ! Kind of types used to parameterize the implementation.
  INTEGER, PARAMETER :: LIBXS_SINGLE_PRECISION  = 4 !KIND(1.0)
  INTEGER, PARAMETER :: LIBXS_DOUBLE_PRECISION  = 8 !KIND(1D0)
  INTEGER, PARAMETER :: LIBXS_INTEGER_TYPE      = KIND(1)

  ! Parameters the library was built for.
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_ALIGNMENT       = $ALIGNMENT
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_ALIGNED_STORES  = $ALIGNED_STORES
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_ALIGNED_LOADS   = $ALIGNED_LOADS
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_ALIGNED_MAX     = $ALIGNED_MAX
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_PREFETCH        = $PREFETCH
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_ROW_MAJOR       = $ROW_MAJOR
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_COL_MAJOR       = $COL_MAJOR
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_MAX_MNK         = $MAX_MNK
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_MAX_M           = $MAX_M
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_MAX_N           = $MAX_N
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_MAX_K           = $MAX_K
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_AVG_M           = $AVG_M
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_AVG_N           = $AVG_N
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_AVG_K           = $AVG_K
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_BETA            = $BETA
  INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: LIBXS_JIT             = $JIT

  ! Overloaded BLAS routines (single/double precision)
  INTERFACE libxs_blasmm
    MODULE PROCEDURE libxs_sblasmm, libxs_dblasmm
  END INTERFACE

  ! Overloaded optimized routines (single/double precision)
  INTERFACE libxs_imm
    MODULE PROCEDURE libxs_simm, libxs_dimm
  END INTERFACE

  ! Overloaded auto-dispatch routines (single/double precision)
  INTERFACE libxs_mm
    MODULE PROCEDURE libxs_smm, libxs_dmm
  END INTERFACE

  ! Type of a function generated for a specific M, N, and K
  ABSTRACT INTERFACE
    ! This interface requires the use of C_LOC in the user code 
    ! on arrays which is not support in all FORTRAN compilers
    PURE SUBROUTINE LIBXS_XMM_FUNCTION(a, b, c) BIND(C)
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE, INTENT(IN) :: a, b, c
    END SUBROUTINE

    PURE SUBROUTINE LIBXS_SMM_FUNCTION(a, b, c) BIND(C)
      IMPORT :: C_FLOAT
      REAL(KIND=C_FLOAT), DIMENSION(*), INTENT(in)  :: a
      REAL(KIND=C_FLOAT), DIMENSION(*), INTENT(in)  :: b
      REAL(KIND=C_FLOAT), DIMENSION(*), INTENT(out) :: c
    END SUBROUTINE

    PURE SUBROUTINE LIBXS_DMM_FUNCTION(a, b, c) BIND(C)
      IMPORT :: C_DOUBLE
      REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(in)  :: a
      REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(in)  :: b
      REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(out) :: c
    END SUBROUTINE
  END INTERFACE

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: sgemm, dgemm, libxs_smm_dispatch, libxs_dmm_dispatch
  INTERFACE
    SUBROUTINE sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      IMPORT LIBXS_INTEGER_TYPE, LIBXS_SINGLE_PRECISION
      CHARACTER(1), INTENT(IN) :: transa, transb
      INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k, lda, ldb, ldc
      REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a(lda,*), b(ldb,*), alpha, beta
      REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c(ldc,*)
    END SUBROUTINE
    SUBROUTINE dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      IMPORT LIBXS_INTEGER_TYPE, LIBXS_DOUBLE_PRECISION
      CHARACTER(1), INTENT(IN) :: transa, transb
      INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k, lda, ldb, ldc
      REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a(lda,*), b(ldb,*), alpha, beta
      REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c(ldc,*)
    END SUBROUTINE

    ! Init the library
    PURE SUBROUTINE libxs_build_static() BIND(C)
    END SUBROUTINE
 
    ! Build explicitly a kernel, do not rely on automatic JIT in dispatch
    PURE SUBROUTINE libxs_build_jit(single_precision, m, n, k) BIND(C)
      IMPORT :: C_FUNPTR, C_INT
      INTEGER(C_INT), VALUE, INTENT(IN) :: single_precision, m, n, k
    END SUBROUTINE

    ! Query the pointer of a generated function; zero if it does not exist, single-precision.
    TYPE(C_FUNPTR) PURE FUNCTION libxs_smm_dispatch(m, n, k) BIND(C)
      IMPORT :: C_FUNPTR, C_INT
      INTEGER(C_INT), VALUE, INTENT(IN) :: m, n, k
    END FUNCTION
    ! Query the pointer of a generated function; zero if it does not exist; double-precision.
    TYPE(C_FUNPTR) PURE FUNCTION libxs_dmm_dispatch(m, n, k) BIND(C)
      IMPORT :: C_FUNPTR, C_INT
      INTEGER(C_INT), VALUE, INTENT(IN) :: m, n, k
    END FUNCTION$MNK_INTERFACE_LIST
  END INTERFACE

CONTAINS
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_up
  !DIR$ ATTRIBUTES INLINE :: libxs_up
  PURE FUNCTION libxs_up(n, up) RESULT(nup)
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: n, up
    INTEGER(LIBXS_INTEGER_TYPE) :: nup
    nup = ((n + up - 1) / up) * up
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_align_value
  !DIR$ ATTRIBUTES INLINE :: libxs_align_value
  PURE FUNCTION libxs_align_value(n, typesize, alignment) RESULT(na)
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: n, typesize, alignment
    INTEGER(LIBXS_INTEGER_TYPE) :: na
    na = libxs_up(n * typesize, alignment) / typesize
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_ld
  !DIR$ ATTRIBUTES INLINE :: libxs_ld
  PURE FUNCTION libxs_ld(m, n) RESULT(ld)
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n
    INTEGER(LIBXS_INTEGER_TYPE) :: ld
    ld = MERGE(m, n, 0.NE.LIBXS_COL_MAJOR)
  END FUNCTION

  ! Non-dispatched matrix-matrix multiplication using BLAS; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sblasmm
  !DIR$ ATTRIBUTES INLINE :: libxs_sblasmm
  SUBROUTINE libxs_sblasmm(m, n, k, a, b, c)
    INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: T = LIBXS_SINGLE_PRECISION
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c(:,:)
    REAL(T), PARAMETER :: alpha = 1, beta = LIBXS_BETA
    IF (0.NE.LIBXS_COL_MAJOR) THEN
      CALL sgemm('N', 'N', m, n, k, alpha, a, m, b, k, beta, c, SIZE(c, 1))
    ELSE
      CALL sgemm('N', 'N', n, m, k, alpha, b, n, a, k, beta, c, SIZE(c, 1))
    ENDIF
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using BLAS; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dblasmm
  !DIR$ ATTRIBUTES INLINE :: libxs_dblasmm
  SUBROUTINE libxs_dblasmm(m, n, k, a, b, c)
    INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: T = LIBXS_DOUBLE_PRECISION
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c(:,:)
    REAL(T), PARAMETER :: alpha = 1, beta = LIBXS_BETA
    IF (0.NE.LIBXS_COL_MAJOR) THEN
      CALL dgemm('N', 'N', m, n, k, alpha, a, m, b, k, beta, c, SIZE(c, 1))
    ELSE
      CALL dgemm('N', 'N', n, m, k, alpha, b, n, a, k, beta, c, SIZE(c, 1))
    ENDIF
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using optimized code; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_simm
  !DIR$ ATTRIBUTES INLINE :: libxs_simm
  SUBROUTINE libxs_simm(m, n, k, a, b, c)
    INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: T = LIBXS_SINGLE_PRECISION
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k
    INTEGER(LIBXS_INTEGER_TYPE) :: i, j
    REAL(T), INTENT(IN), TARGET, CONTIGUOUS :: a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c($SHAPE_C1,$SHAPE_C2)
    REAL(T), POINTER :: x(:,:), y(:,:)
    IF (0.NE.LIBXS_COL_MAJOR.AND.LIBXS_BETA.NE.0) THEN
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = LBOUND(b, 2), LBOUND(b, 2) + n - 1
        !DIR$ LOOP COUNT(1, LIBXS_MAX_M, LIBXS_AVG_M)
        DO i = LBOUND(a, 1), LBOUND(a, 1) + m - 1
          c(i,j) = c(i,j) + DOT_PRODUCT(a(i,:), b(:,j))
        END DO
      END DO
    ELSE IF (LIBXS_BETA.NE.0) THEN
      x(1:$SHAPE_AT1,1:$SHAPE_AT2) => b(:,:)
      y(1:$SHAPE_BT1,1:$SHAPE_BT2) => a(:,:)
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = 1, m
        !DIR$ LOOP COUNT(1, LIBXS_MAX_N, LIBXS_AVG_N)
        DO i = 1, n
          c(i,j) = c(i,j) + DOT_PRODUCT(x(i,:), y(:,j))
        END DO
      END DO
    ELSE IF (0.NE.LIBXS_COL_MAJOR) THEN
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = LBOUND(b, 2), LBOUND(b, 2) + n - 1
        !DIR$ LOOP COUNT(1, LIBXS_MAX_M, LIBXS_AVG_M)
        DO i = LBOUND(a, 1), LBOUND(a, 1) + m - 1
          c(i,j) = DOT_PRODUCT(a(i,:), b(:,j))
        END DO
      END DO
    ELSE
      x(1:$SHAPE_AT1,1:$SHAPE_AT2) => b(:,:)
      y(1:$SHAPE_BT1,1:$SHAPE_BT2) => a(:,:)
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = 1, m
        !DIR$ LOOP COUNT(1, LIBXS_MAX_N, LIBXS_AVG_N)
        DO i = 1, n
          c(i,j) = DOT_PRODUCT(x(i,:), y(:,j))
        END DO
      END DO
    ENDIF
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using optimized code; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dimm
  !DIR$ ATTRIBUTES INLINE :: libxs_dimm
  SUBROUTINE libxs_dimm(m, n, k, a, b, c)
    INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: T = LIBXS_DOUBLE_PRECISION
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k
    INTEGER(LIBXS_INTEGER_TYPE) :: i, j
    REAL(T), INTENT(IN), TARGET, CONTIGUOUS :: a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c($SHAPE_C1,$SHAPE_C2)
    REAL(T), POINTER :: x(:,:), y(:,:)
    IF (0.NE.LIBXS_COL_MAJOR.AND.LIBXS_BETA.NE.0) THEN
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = LBOUND(b, 2), LBOUND(b, 2) + n - 1
        !DIR$ LOOP COUNT(1, LIBXS_MAX_M, LIBXS_AVG_M)
        DO i = LBOUND(a, 1), LBOUND(a, 1) + m - 1
          c(i,j) = c(i,j) + DOT_PRODUCT(a(i,:), b(:,j))
        END DO
      END DO
    ELSE IF (LIBXS_BETA.NE.0) THEN
      x(1:$SHAPE_AT1,1:$SHAPE_AT2) => b(:,:)
      y(1:$SHAPE_BT1,1:$SHAPE_BT2) => a(:,:)
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = 1, m
        !DIR$ LOOP COUNT(1, LIBXS_MAX_N, LIBXS_AVG_N)
        DO i = 1, n
          c(i,j) = c(i,j) + DOT_PRODUCT(x(i,:), y(:,j))
        END DO
      END DO
    ELSE IF (0.NE.LIBXS_COL_MAJOR) THEN
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = LBOUND(b, 2), LBOUND(b, 2) + n - 1
        !DIR$ LOOP COUNT(1, LIBXS_MAX_M, LIBXS_AVG_M)
        DO i = LBOUND(a, 1), LBOUND(a, 1) + m - 1
          c(i,j) = DOT_PRODUCT(a(i,:), b(:,j))
        END DO
      END DO
    ELSE
      x(1:$SHAPE_AT1,1:$SHAPE_AT2) => b(:,:)
      y(1:$SHAPE_BT1,1:$SHAPE_BT2) => a(:,:)
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = 1, m
        !DIR$ LOOP COUNT(1, LIBXS_MAX_N, LIBXS_AVG_N)
        DO i = 1, n
          c(i,j) = DOT_PRODUCT(x(i,:), y(:,j))
        END DO
      END DO
    ENDIF
  END SUBROUTINE

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_mm_dispatch
  !DIR$ ATTRIBUTES INLINE :: libxs_mm_dispatch
  FUNCTION libxs_mm_dispatch(m, n, k, type) RESULT(f)
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k, type
    !PROCEDURE(LIBXS_XMM_FUNCTION), POINTER :: f
    TYPE(C_FUNPTR) :: f
    f = MERGE( &
      libxs_dmm_dispatch(m, n, k), &
      libxs_smm_dispatch(m, n, k), &
      LIBXS_DOUBLE_PRECISION.EQ.type)
  END FUNCTION

  ! Dispatched matrix-matrix multiplication; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_smm
  !DIR$ ATTRIBUTES INLINE :: libxs_smm
  SUBROUTINE libxs_smm(m, n, k, a, b, c)
    INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: T = LIBXS_SINGLE_PRECISION
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), TARGET, INTENT(IN) :: a(:,:), b(:,:)
    REAL(T), TARGET, INTENT(INOUT) :: c(:,:)
    !DIR$ ATTRIBUTES OFFLOAD:MIC :: smm
    PROCEDURE(LIBXS_SMM_FUNCTION), POINTER :: smm
    TYPE(C_FUNPTR) :: f
    IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
      f = libxs_smm_dispatch(m, n, k)
      IF (C_ASSOCIATED(f)) THEN
        CALL C_F_PROCPOINTER(f, smm)
        CALL smm(a, b, c)
      ELSE
        CALL libxs_simm(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxs_sblasmm(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE

  ! Dispatched matrix-matrix multiplication; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dmm
  !DIR$ ATTRIBUTES INLINE :: libxs_dmm
  SUBROUTINE libxs_dmm(m, n, k, a, b, c)
    INTEGER(LIBXS_INTEGER_TYPE), PARAMETER :: T = LIBXS_DOUBLE_PRECISION
    INTEGER(LIBXS_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), TARGET, INTENT(IN) :: a(:,:), b(:,:)
    REAL(T), TARGET, INTENT(INOUT) :: c(:,:)
    !DIR$ ATTRIBUTES OFFLOAD:MIC :: dmm
    PROCEDURE(LIBXS_DMM_FUNCTION), POINTER :: dmm
    TYPE(C_FUNPTR) :: f
    IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
      f = libxs_dmm_dispatch(m, n, k)
      IF (C_ASSOCIATED(f)) THEN
        CALL C_F_PROCPOINTER(f, dmm)
        CALL dmm(a, b, c)
      ELSE
        CALL libxs_dimm(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxs_dblasmm(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE
END MODULE
