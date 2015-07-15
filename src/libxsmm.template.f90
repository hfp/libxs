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
  USE, INTRINSIC :: ISO_C_BINDING
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

  INTERFACE libxs_blasmm
    MODULE PROCEDURE libxs_sblasmm, libxs_dblasmm
  END INTERFACE

  INTERFACE libxs_imm
    MODULE PROCEDURE libxs_simm, libxs_dimm
  END INTERFACE

  INTERFACE libxs_mm
    MODULE PROCEDURE libxs_smm, libxs_dmm
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
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_sblasmm
  !DIR$ ATTRIBUTES INLINE :: libxs_sblasmm
  PURE SUBROUTINE libxs_sblasmm(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_BLASMM(LIBXS_SINGLE_PRECISION, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using BLAS; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dblasmm
  !DIR$ ATTRIBUTES INLINE :: libxs_dblasmm
  PURE SUBROUTINE libxs_dblasmm(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_BLASMM(LIBXS_DOUBLE_PRECISION, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_simm
  !DIR$ ATTRIBUTES INLINE :: libxs_simm
  PURE SUBROUTINE libxs_simm(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_IMM(LIBXS_SINGLE_PRECISION, LIBXS_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dimm
  !DIR$ ATTRIBUTES INLINE :: libxs_dimm
  PURE SUBROUTINE libxs_dimm(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXS_IMM(LIBXS_DOUBLE_PRECISION, LIBXS_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_smm_dispatch
  !DIR$ ATTRIBUTES INLINE :: libxs_smm_dispatch
  FUNCTION libxs_smm_dispatch(m, n, k) RESULT(f)
    PROCEDURE(LIBXS_SMM_FUNCTION), POINTER :: f
    INTEGER, INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_smm_dispatch_aux
      TYPE(C_FUNPTR) PURE FUNCTION libxs_smm_dispatch_aux(m, n, k) BIND(C, NAME="libxs_smm_dispatch")
        USE, INTRINSIC :: ISO_C_BINDING
        INTEGER(C_INT), INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(libxs_smm_dispatch_aux(m, n, k), f)
  END FUNCTION

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dmm_dispatch
  !DIR$ ATTRIBUTES INLINE :: libxs_dmm_dispatch
  FUNCTION libxs_dmm_dispatch(m, n, k) RESULT(f)
    PROCEDURE(LIBXS_DMM_FUNCTION), POINTER :: f
    INTEGER, INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_dmm_dispatch_aux
      TYPE(C_FUNPTR) PURE FUNCTION libxs_dmm_dispatch_aux(m, n, k) BIND(C, NAME="libxs_dmm_dispatch")
        USE, INTRINSIC :: ISO_C_BINDING
        INTEGER(C_INT), INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(libxs_dmm_dispatch_aux(m, n, k), f)
  END FUNCTION

  ! Dispatched matrix-matrix multiplication; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxs_smm
  !DIR$ ATTRIBUTES INLINE :: libxs_smm
  SUBROUTINE libxs_smm(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    PROCEDURE(LIBXS_SMM_FUNCTION), POINTER :: f
    IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
      f => libxs_smm_dispatch(m, n, k)
      IF (ASSOCIATED(f)) THEN
        CALL f(a, b, c)
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
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXS_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    PROCEDURE(LIBXS_DMM_FUNCTION), POINTER :: f
    IF (LIBXS_MAX_MNK.GE.(m * n * k)) THEN
      f => libxs_dmm_dispatch(m, n, k)
      IF (ASSOCIATED(f)) THEN
        CALL f(a, b, c)
      ELSE
        CALL libxs_dimm(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxs_dblasmm(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE
END MODULE
