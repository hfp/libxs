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
!* Hans Pabst (Intel Corp.)                                                  *!
!*****************************************************************************!

MODULE LIBXS
  IMPLICIT NONE

  ! Kind of types used to parameterize the implementation.
  INTEGER, PARAMETER :: LIBXS_SINGLE_PRECISION  = KIND(1.0)
  INTEGER, PARAMETER :: LIBXS_DOUBLE_PRECISION  = KIND(1D0)
  INTEGER, PARAMETER :: LIBXS_INTEGER_TYPE      = KIND(1)

  ! Parameters the library was built for.
  INTEGER, PARAMETER :: LIBXS_ALIGNMENT       = $ALIGNMENT
  INTEGER, PARAMETER :: LIBXS_ALIGNED_STORES  = $ALIGNED_STORES
  INTEGER, PARAMETER :: LIBXS_ALIGNED_LOADS   = $ALIGNED_LOADS
  INTEGER, PARAMETER :: LIBXS_ALIGNED_MAX     = $ALIGNED_MAX
  INTEGER, PARAMETER :: LIBXS_ROW_MAJOR       = $ROW_MAJOR
  INTEGER, PARAMETER :: LIBXS_COL_MAJOR       = $COL_MAJOR
  INTEGER, PARAMETER :: LIBXS_MAX_MNK         = $MAX_MNK
  INTEGER, PARAMETER :: LIBXS_MAX_M           = $MAX_M
  INTEGER, PARAMETER :: LIBXS_MAX_N           = $MAX_N
  INTEGER, PARAMETER :: LIBXS_MAX_K           = $MAX_K
  INTEGER, PARAMETER :: LIBXS_AVG_M           = $AVG_M
  INTEGER, PARAMETER :: LIBXS_AVG_N           = $AVG_N
  INTEGER, PARAMETER :: LIBXS_AVG_K           = $AVG_K

  INTERFACE LIBXS_BLASMM
    MODULE PROCEDURE LIBXS_SBLASMM, LIBXS_DBLASMM
  END INTERFACE

  INTERFACE LIBXS_IMM
    MODULE PROCEDURE LIBXS_SIMM, LIBXS_DIMM
  END INTERFACE

  INTERFACE LIBXS_MM
    MODULE PROCEDURE LIBXS_SMM, LIBXS_DMM
  END INTERFACE

  ABSTRACT INTERFACE
    ! Type of a function generated for a specific M, N, and K.
    PURE SUBROUTINE LIBXS_SMM_FUNCTION(a, b, c)
      USE, INTRINSIC :: ISO_C_BINDING
      REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
      REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
    END SUBROUTINE

    ! Type of a function generated for a specific M, N, and K.
    PURE SUBROUTINE LIBXS_DMM_FUNCTION(a, b, c)
      USE, INTRINSIC :: ISO_C_BINDING
      REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
      REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
    END SUBROUTINE
  END INTERFACE$MNK_INTERFACE_LIST

CONTAINS
  ! Non-dispatched matrix-matrix multiplication using BLAS; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_SBLASMM
  !DIR$ ATTRIBUTES INLINE :: LIBXS_SBLASMM
  PURE SUBROUTINE LIBXS_SBLASMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_BLASMM(LIBXS_SINGLE_PRECISION, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using BLAS; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_DBLASMM
  !DIR$ ATTRIBUTES INLINE :: LIBXS_DBLASMM
  PURE SUBROUTINE LIBXS_DBLASMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_BLASMM(LIBXS_DOUBLE_PRECISION, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_SIMM
  !DIR$ ATTRIBUTES INLINE :: LIBXS_SIMM
  PURE SUBROUTINE LIBXS_SIMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_IMM(LIBXS_SINGLE_PRECISION, LIBXS_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_DIMM
  !DIR$ ATTRIBUTES INLINE :: LIBXS_DIMM
  PURE SUBROUTINE LIBXS_DIMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_IMM(LIBXS_DOUBLE_PRECISION, LIBXS_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_SMM_DISPATCH
  !DIR$ ATTRIBUTES INLINE :: LIBXS_SMM_DISPATCH
  FUNCTION LIBXS_SMM_DISPATCH(m, n, k) RESULT(f)
    PROCEDURE(LIBXS_SMM_FUNCTION), POINTER :: f
    INTEGER, INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_SMM_DISPATCH_AUX
      TYPE(C_FUNPTR) PURE FUNCTION LIBXS_SMM_DISPATCH_AUX(m, n, k) BIND(C, NAME="libxs_smm_dispatch")
        USE, INTRINSIC :: ISO_C_BINDING
        INTEGER(C_INT), INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(LIBXS_SMM_DISPATCH_AUX(m, n, k), f)
  END FUNCTION

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_DMM_DISPATCH
  !DIR$ ATTRIBUTES INLINE :: LIBXS_DMM_DISPATCH
  FUNCTION LIBXS_DMM_DISPATCH(m, n, k) RESULT(f)
    PROCEDURE(LIBXS_DMM_FUNCTION), POINTER :: f
    INTEGER, INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_DMM_DISPATCH_AUX
      TYPE(C_FUNPTR) PURE FUNCTION LIBXS_DMM_DISPATCH_AUX(m, n, k) BIND(C, NAME="libxs_dmm_dispatch")
        USE, INTRINSIC :: ISO_C_BINDING
        INTEGER(C_INT), INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(LIBXS_DMM_DISPATCH_AUX(m, n, k), f)
  END FUNCTION

  ! Dispatched matrix-matrix multiplication; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_SMM
  !DIR$ ATTRIBUTES INLINE :: LIBXS_SMM
  SUBROUTINE LIBXS_SMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    PROCEDURE(LIBXS_SMM_FUNCTION), POINTER :: f
    IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
      f => LIBXS_SMM_DISPATCH(m, n, k)
      IF (ASSOCIATED(f)) THEN
        CALL f(a, b, c)
      ELSE
        CALL LIBXS_SIMM(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL LIBXS_SBLASMM(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE

  ! Dispatched matrix-matrix multiplication; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXS_DMM
  !DIR$ ATTRIBUTES INLINE :: LIBXS_DMM
  SUBROUTINE LIBXS_DMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    PROCEDURE(LIBXS_DMM_FUNCTION), POINTER :: f
    IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
      f => LIBXS_DMM_DISPATCH(m, n, k)
      IF (ASSOCIATED(f)) THEN
        CALL f(a, b, c)
      ELSE
        CALL LIBXS_DIMM(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL LIBXS_DBLASMM(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE
END MODULE
