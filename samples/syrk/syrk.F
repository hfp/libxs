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
     &    libxs_timer_tick, libxs_timer_duration,                       &
     &    libxs_init, libxs_finalize,                                   &
     &    C_LOC, C_PTR, C_NULL_PTR, C_ASSOCIATED,                       &
     &    C_FUNPTR, C_F_POINTER, C_DOUBLE, C_INT
        USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_FUNLOC
        IMPLICIT NONE
#if defined(__MKL)
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
        END INTERFACE
#endif
#if defined(__BLAS) || defined(__MKL)
        INTERFACE
          SUBROUTINE DGEMM(transa, transb, m, n, k,                     &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="dgemm_")
            USE, INTRINSIC :: ISO_C_BINDING,                            &
     &        ONLY: C_DOUBLE, C_INT, C_CHAR
            CHARACTER(1, C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(C_INT), INTENT(IN) :: m, n, k, lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta, a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
          END SUBROUTINE
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
#endif

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: n, k, argc, r, nrepeat
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

        alpha = 1D0; beta = 1D0

        WRITE(*, "(A,I0,A,I0,A,I0)")                                    &
     &    "SYRK: N=", n, " K=", k, " nrepeat=", nrepeat

        CALL libxs_init()

        ALLOCATE(a(n, k), b(n, k), c(n, n), cref(n, n))

        CALL RANDOM_NUMBER(a)
        CALL RANDOM_NUMBER(b)
        c = 0D0; cref = 0D0

        ! SYRK: C := alpha*A*A^T + beta*C (lower triangle)
        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- libxs_syrk (lower) ---"

#if defined(__MKL)
        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n,       &
     &    jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),        &
     &    jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),                &
     &    dgemm_blas=C_FUNLOC(DGEMM))
#elif defined(__BLAS)
        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n,       &
     &    dgemm_blas=C_FUNLOC(DGEMM))
#else
        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
#endif
        IF (.NOT. C_ASSOCIATED(ptr)) THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch returned NULL"
          ERROR STOP 1
        END IF
        CALL C_F_POINTER(ptr, config)

        CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)
        CALL libxs_syrk(config, 'L', alpha, beta, C_LOC(a), C_LOC(c))

        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(*, "(A,E12.5)")                                           &
     &    "  max error (lower): ", diff%linf_abs

        ! SYR2K: C := alpha*(A*B^T + B*A^T) + beta*C (upper)
        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- libxs_syr2k (upper) ---"

        c = 0D0; cref = 0D0

#if defined(__MKL)
        ptr = libxs_syr2k_dispatch(LIBXS_DATATYPE_F64,                  &
     &    n, k, n, n, n,                                                &
     &    jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),        &
     &    jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),                &
     &    dgemm_blas=C_FUNLOC(DGEMM))
#elif defined(__BLAS)
        ptr = libxs_syr2k_dispatch(LIBXS_DATATYPE_F64,                  &
     &    n, k, n, n, n, dgemm_blas=C_FUNLOC(DGEMM))
#else
        ptr = libxs_syr2k_dispatch(LIBXS_DATATYPE_F64,                  &
     &    n, k, n, n, n)
#endif
        IF (.NOT. C_ASSOCIATED(ptr)) THEN
          WRITE(*, "(A)") "FAILED: libxs_syr2k_dispatch returned NULL"
          ERROR STOP 1
        END IF
        CALL C_F_POINTER(ptr, config)

        CALL DSYR2K('U', 'N', n, k, alpha, a, n, b, n, beta, cref, n)
        CALL libxs_syr2k(config, 'U', alpha, beta,                      &
     &    C_LOC(a), C_LOC(b), C_LOC(c))

        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(*, "(A,E12.5)")                                           &
     &    "  max error (upper): ", diff%linf_abs

        ! Performance comparison
        WRITE(*, "(A)") ""
        WRITE(*, "(A)") "--- SYRK performance ---"

#if defined(__MKL)
        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n,       &
     &    jit_create_dgemm=C_FUNLOC(mkl_cblas_jit_create_dgemm),        &
     &    jit_get_dgemm=C_FUNLOC(mkl_jit_get_dgemm_ptr),                &
     &    dgemm_blas=C_FUNLOC(DGEMM))
#elif defined(__BLAS)
        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n,       &
     &    dgemm_blas=C_FUNLOC(DGEMM))
#else
        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
#endif
        CALL C_F_POINTER(ptr, config)

        gflops = DBLE(n) * DBLE(n) * DBLE(k) * 2D-9

        ! BLAS timing
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

        ! LIBXS timing
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
