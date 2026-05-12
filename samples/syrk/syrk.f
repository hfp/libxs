!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      PROGRAM syrk_sample
        USE :: LIBXS, ONLY: LIBXS_DATATYPE_F64,                         &
     &    LIBXS_TIMER_TICK_KIND,                                        &
     &    libxs_gemm_config_t,                                          &
     &    libxs_syrk_dispatch, libxs_syrk,                              &
     &    libxs_syr2k_dispatch, libxs_syr2k,                            &
     &    libxs_matdiff_t, libxs_matdiff, libxs_matdiff_clear,          &
     &    libxs_matdiff_epsilon,                                        &
     &    libxs_timer_tick, libxs_timer_duration,                       &
     &    libxs_init, libxs_finalize,                                   &
     &    C_LOC, C_PTR, C_NULL_PTR, C_ASSOCIATED,                       &
     &    C_F_POINTER, C_DOUBLE, C_INT
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: n, k, argc, r, nrepeat, i, j
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
        INTEGER(C_INT) :: rc

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

        alpha = 1D0; beta = 1D0

        WRITE(*, "(A,I0,A,I0,A,I0)")                                    &
     &    "syrk(F): N=", n, " K=", k, " nrepeat=", nrepeat

        CALL libxs_init()

        ALLOCATE(a(n, k), b(n, k), c(n, n), cref(n, n))

        CALL RANDOM_NUMBER(a)
        CALL RANDOM_NUMBER(b)
        c = 0D0; cref = 0D0

        ! SYRK: C := alpha*A*A^T + beta*C (lower triangle)
        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- libxs_syrk (lower) ---"

        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
        IF (.NOT. C_ASSOCIATED(ptr)) THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch returned NULL"
          ERROR STOP 1
        END IF
        CALL C_F_POINTER(ptr, config)

        ! Reference: DSYRK via loops
        DO j = 1, n
          DO i = j, n
            cref(i, j) = alpha * DOT_PRODUCT(a(i,:), a(j,:))            &
     &                 + beta * cref(i, j)
          END DO
        END DO

        rc = libxs_syrk(config, 'L', alpha, beta, C_LOC(a), C_LOC(c))
        IF (0 /= rc) THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk returned error"
          ERROR STOP 1
        END IF

        ! Validate lower triangle
        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(*, "(A,E12.5)")                                           &
     &    "  max error (lower): ", diff%linf_abs

        ! SYR2K: C := alpha*(A*B^T + B*A^T) + beta*C (upper)
        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- libxs_syr2k (upper) ---"

        c = 0D0; cref = 0D0

        ptr = libxs_syr2k_dispatch(                                     &
     &    LIBXS_DATATYPE_F64, n, k, n, n, n)
        IF (.NOT. C_ASSOCIATED(ptr)) THEN
          WRITE(*, "(A)") "FAILED: libxs_syr2k_dispatch returned NULL"
          ERROR STOP 1
        END IF
        CALL C_F_POINTER(ptr, config)

        ! Reference: DSYR2K via loops
        DO j = 1, n
          DO i = 1, j
            cref(i, j) = alpha                                          &
     &        * (DOT_PRODUCT(a(i,:), b(j,:))                            &
     &         + DOT_PRODUCT(b(i,:), a(j,:)))                           &
     &        + beta * cref(i, j)
          END DO
        END DO

        rc = libxs_syr2k(config, 'U', alpha, beta,                      &
     &    C_LOC(a), C_LOC(b), C_LOC(c))
        IF (0 /= rc) THEN
          WRITE(*, "(A)") "FAILED: libxs_syr2k returned error"
          ERROR STOP 1
        END IF

        ! Validate upper triangle
        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(*, "(A,E12.5)")                                           &
     &    "  max error (upper): ", diff%linf_abs

        ! Performance: SYRK
        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- SYRK performance ---"

        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
        CALL C_F_POINTER(ptr, config)
        c = 0D0

        ! warmup
        rc = libxs_syrk(config, 'L', alpha, beta, C_LOC(a), C_LOC(c))

        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
          rc = libxs_syrk(config, 'L', alpha, beta,                     &
     &      C_LOC(a), C_LOC(c))
        END DO
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)

        ! SYRK flops: alpha*A*A^T touches n*(n+1)/2 * (2k-1) FMA
        gflops = DBLE(n) * DBLE(n) * DBLE(k) * 2D-9
        IF (0D0 < duration) THEN
          WRITE(*, "(A,F10.3,A,I0,A)")                                  &
     &      "  time: ", duration, " s (", nrepeat, " calls)"
          WRITE(*, "(A,F10.1,A)")                                       &
     &      "  perf: ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF

        DEALLOCATE(a, b, c, cref)
        CALL libxs_finalize()
      END PROGRAM
