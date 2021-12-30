!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                        !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.)
!=======================================================================!

! This (micro-)benchmark is a simplified variant of the C implementation;
! the main point of dispatch.f is to show compatibility with FORTRAN 77.
! NOTE: CPU_TIME is a Fortran 96 intrinsic, libxs_xmmdispatch must be
! called with all arguments when relying on FORTRAN 77.
!
! IMPORTANT: please use the type-safe F2003 interface (libxs.f or module)
!            unless FORTRAN 77 compatibility is really needed!
!
      PROGRAM dispatch
        !USE :: LIBXS
        !IMPLICIT NONE

        INTEGER, PARAMETER :: PRECISION = 0 ! LIBXS_DATATYPE_F64
        INTEGER, PARAMETER :: M = 23, N = 23, K = 23
        INTEGER, PARAMETER :: LDA = M, LDB = K, LDC = M
        DOUBLE PRECISION, PARAMETER :: Alpha = 1D0
        DOUBLE PRECISION, PARAMETER :: Beta = 1D0
        INTEGER, PARAMETER :: Flags = 0
        INTEGER, PARAMETER :: Prefetch = 0

        DOUBLE PRECISION :: start, dcall, ddisp
        INTEGER :: i, size = 10000000
        ! Can be called using:
        ! - libxs_xmmcall_abc(function, a, b, c)
        ! - libxs_xmmcall[_prf](function, a, b, c, pa, pb, pc)
        INTEGER(8) :: function

        WRITE(*, "(A,I0,A)") "Dispatching ", size," calls..."

        ! run non-inline function to measure call overhead of an "empty" function
        ! subsequent calls (see above) of libxs_init are not doing any work
        !
        CALL CPU_TIME(start)
        DO i = 1, size
          CALL libxs_init()
        END DO
        CALL CPU_TIME(dcall)
        dcall = dcall - start

        ! first invocation may initialize some internals (libxs_init),
        ! or actually generate code (code gen. time is out of scope)
        ! NOTE: libxs_xmmdispatch must be called with all arguments
        ! when relying on FORTRAN 77.
        !
        CALL libxs_xmmdispatch(function, PRECISION, M, N, K,          &
     &    LDA, LDB, LDC, Alpha, Beta, Flags, Prefetch)

        CALL CPU_TIME(start)
        DO i = 1, size
          ! NOTE: libxs_xmmdispatch must be called with all arguments
          ! when relying on FORTRAN 77.
          CALL libxs_xmmdispatch(function, PRECISION, M, N, K,        &
     &      LDA, LDB, LDC, Alpha, Beta, Flags, Prefetch)
        END DO
        CALL CPU_TIME(ddisp)
        ddisp = ddisp - start

        IF ((0.LT.dcall).AND.(0.LT.ddisp)) THEN
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "dispatch calls/s: ",     &
     &                      (1D-6 * REAL(size, 8) / ddisp), " MHz"
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "empty calls/s:    ",     &
     &                      (1D-6 * REAL(size, 8) / dcall), " MHz"
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "overhead:         ",     &
     &                      (ddisp / dcall), "x"
        END IF
        WRITE(*, "(A)") "Finished"
      END PROGRAM

