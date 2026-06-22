!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      PROGRAM syrk_sample
        USE :: LIBXS_JIT, ONLY: LIBXS_DATATYPE_F64,                     &
     &    LIBXS_TIMER_TICK_KIND,                                        &
     &    libxs_gemm_config_t,                                          &
     &    libxs_syrk_dispatch, libxs_syrk,                              &
     &    libxs_syr2k_dispatch, libxs_syr2k,                            &
     &    libxs_matdiff_t, libxs_matdiff, libxs_matdiff_clear,          &
     &    libxs_timer_tick, libxs_timer_duration,                       &
     &    libxs_init, libxs_finalize,                                   &
     &    C_LOC, C_PTR, C_NULL_PTR, C_ASSOCIATED,                       &
     &    C_F_POINTER, C_DOUBLE, C_INT
        IMPLICIT NONE

        INTERFACE
          SUBROUTINE DSYRK(uplo, trans, n, k,                           &
     &    alpha, a, lda, beta, c, ldc)                                  &
     &    BIND(C, NAME="dsyrk_")
            USE, INTRINSIC :: ISO_C_BINDING,                            &
     &        ONLY: C_DOUBLE, C_INT, C_CHAR
            CHARACTER(1, C_CHAR), INTENT(IN) :: uplo, trans
            INTEGER(C_INT), INTENT(IN) :: n, k, lda, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta, a(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
          END SUBROUTINE
          SUBROUTINE DSYR2K(uplo, trans, n, k,                          &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="dsyr2k_")
            USE, INTRINSIC :: ISO_C_BINDING,                            &
     &        ONLY: C_DOUBLE, C_INT, C_CHAR
            CHARACTER(1, C_CHAR), INTENT(IN) :: uplo, trans
            INTEGER(C_INT), INTENT(IN) :: n, k, lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta, a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
          END SUBROUTINE
        END INTERFACE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: n, k, argc, r, nrepeat, direct
        CHARACTER(32) :: argv
        REAL(T), ALLOCATABLE, TARGET :: a(:,:), b(:,:)
        REAL(T), ALLOCATABLE, TARGET :: c(:,:), cref(:,:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a, b, c, cref
        REAL(T) :: alpha, beta
        TYPE(libxs_gemm_config_t), POINTER :: config
        TYPE(C_PTR) :: ptr
        TYPE(libxs_matdiff_t) :: diff
        INTEGER(LIBXS_TIMER_TICK_KIND) :: t0, t1
        DOUBLE PRECISION :: duration, gflops

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") n
        ELSE
          n = 64
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") k
        ELSE
          k = n
        END IF
        IF (3 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(3, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 100
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") direct
        ELSE
          direct = 0
        END IF

        alpha = 1D0; beta = 1D0

        WRITE(*, "(A,I0,A,I0,A,I0,A,I0)")                               &
     &    "SYRK: N=", n, " K=", k, " nrepeat=", nrepeat,                &
     &    " direct=", direct

        CALL libxs_init()

        ALLOCATE(a(n, k), b(n, k), c(n, n), cref(n, n))

        CALL RANDOM_NUMBER(a)
        CALL RANDOM_NUMBER(b)
        c = 0D0; cref = 0D0

        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- libxs_syrk (lower) ---"

        IF (0 .EQ. direct) THEN
          ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
          IF (.NOT. C_ASSOCIATED(ptr)) THEN
            WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch returned NULL"
            ERROR STOP 1
          END IF
          CALL C_F_POINTER(ptr, config)
        END IF

        CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)
        IF (0 .EQ. direct) THEN
          CALL libxs_syrk(config, 'L', alpha, beta,                     &
     &      C_LOC(a), C_LOC(c))
        ELSE
          CALL libxs_syrk(LIBXS_DATATYPE_F64, n, k, n, n,               &
     &      'L', alpha, beta, C_LOC(a), C_LOC(c))
        END IF

        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(*, "(A,E12.5)")                                           &
     &    "  max error (lower): ", diff%linf_abs

        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- libxs_syr2k (upper) ---"

        c = 0D0; cref = 0D0

        IF (0 .EQ. direct) THEN
          ptr = libxs_syr2k_dispatch(LIBXS_DATATYPE_F64,                &
     &      n, k, n, n, n)
          IF (.NOT. C_ASSOCIATED(ptr)) THEN
            WRITE(*, "(A)")                                             &
     &        "FAILED: libxs_syr2k_dispatch returned NULL"
            ERROR STOP 1
          END IF
          CALL C_F_POINTER(ptr, config)
        END IF

        CALL DSYR2K('U', 'N', n, k, alpha, a, n, b, n, beta, cref, n)
        IF (0 .EQ. direct) THEN
          CALL libxs_syr2k(config, 'U', alpha, beta,                    &
     &      C_LOC(a), C_LOC(b), C_LOC(c))
        ELSE
          CALL libxs_syr2k(LIBXS_DATATYPE_F64, n, k, n, n, n,           &
     &      'U', alpha, beta, C_LOC(a), C_LOC(b), C_LOC(c))
        END IF

        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(*, "(A,E12.5)")                                           &
     &    "  max error (upper): ", diff%linf_abs

        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- SYRK performance ---"

        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
        CALL C_F_POINTER(ptr, config)

        gflops = DBLE(n) * DBLE(n) * DBLE(k) * 2D-9

        cref = 0D0
        CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)
        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
          CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)
        END DO
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)
        IF (0D0 < duration) THEN
          WRITE(*, "(A,F10.3,A,I0,A)")                                  &
     &      "  BLAS: ", duration, " s (", nrepeat, " calls)"
          WRITE(*, "(A,F10.1,A)")                                       &
     &      "        ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF

        c = 0D0
        CALL libxs_syrk(config, 'L', alpha, beta, C_LOC(a), C_LOC(c))
        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
          CALL libxs_syrk(config, 'L', alpha, beta,                     &
     &      C_LOC(a), C_LOC(c))
        END DO
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)
        IF (0D0 < duration) THEN
          WRITE(*, "(A,F10.3,A,I0,A)")                                  &
     &      "  LIBXS:", duration, " s (", nrepeat, " calls)"
          WRITE(*, "(A,F10.1,A)")                                       &
     &      "        ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF

        DEALLOCATE(a, b, c, cref)
        CALL libxs_finalize()
      END PROGRAM
