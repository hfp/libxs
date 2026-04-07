!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      PROGRAM gemm_index
        USE :: LIBXS, ONLY: LIBXS_DATATYPE_F64,                         &
     &    LIBXS_GEMM_FLAG_NOLOCK,                                       &
     &    LIBXS_TIMER_TICK_KIND,                                        &
     &    libxs_gemm_config_t,                                          &
     &    libxs_gemm_index,                                             &
     &    libxs_gemm_index_task,                                        &
     &    libxs_timer_tick,                                             &
     &    libxs_timer_duration,                                         &
     &    libxs_init, libxs_finalize
        USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_LOC, C_INT,            &
     &    C_DOUBLE, C_SIZEOF, C_FUNPTR, C_PTR, C_NULL_PTR
!$      USE :: OMP_LIB, ONLY: omp_get_thread_num,                       &
!$   &    omp_get_num_threads
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: m, n, k, batchsize, nrepeat
        INTEGER :: lda, ldb, ldc
        INTEGER :: stride_a, stride_b, stride_c
        INTEGER :: argc, r, i
        CHARACTER(32) :: argv

        REAL(T), ALLOCATABLE, TARGET :: a(:), b(:), c(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a, b, c
        INTEGER(C_INT), ALLOCATABLE, TARGET :: ia(:), ib(:), ic(:)
        REAL(T), TARGET :: alpha, beta
        TYPE(libxs_gemm_config_t) :: config
        INTEGER(LIBXS_TIMER_TICK_KIND) :: t0, t1
        DOUBLE PRECISION :: duration, gflops
#if defined(__MKL)
        INTEGER, PARAMETER :: MKL_JIT_SUCCESS = 0
        TYPE(C_PTR) :: jitter = C_NULL_PTR
        TYPE(C_FUNPTR) :: jit_cfptr
        INTEGER :: jit_status
        INTERFACE
          FUNCTION mkl_jit_create_dgemm(jitter,                         &
     &    transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)          &
     &    RESULT(status) BIND(C)
            IMPORT :: C_PTR, C_INT, C_DOUBLE
            INTEGER(C_INT) :: status
            TYPE(C_PTR) :: jitter
            CHARACTER :: transa, transb
            INTEGER(C_INT) :: m, n, k, lda, ldb, ldc
            REAL(C_DOUBLE) :: alpha, beta
          END FUNCTION
          FUNCTION mkl_jit_get_dgemm_ptr(jitter)                        &
     &    RESULT(ptr) BIND(C)
            IMPORT :: C_FUNPTR, C_PTR
            TYPE(C_FUNPTR) :: ptr
            TYPE(C_PTR), INTENT(IN), VALUE :: jitter
          END FUNCTION
          FUNCTION mkl_jit_destroy(jitter)                              &
     &    RESULT(status) BIND(C)
            IMPORT :: C_PTR, C_INT
            INTEGER(C_INT) :: status
            TYPE(C_PTR), INTENT(IN), VALUE :: jitter
          END FUNCTION
        END INTERFACE
#endif

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") m
        ELSE
          m = 23
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") n
        ELSE
          n = m
        END IF
        IF (3 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(3, argv)
          READ(argv, "(I32)") k
        ELSE
          k = m
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") batchsize
        ELSE
          batchsize = 30000
        END IF
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 3
        END IF

        lda = m; ldb = k; ldc = m
        stride_a = lda * k
        stride_b = ldb * n
        stride_c = ldc * n
        alpha = 1D0; beta = 1D0

        WRITE(*, "(A,I0,A,I0,A,I0,A,I0,A,I0)")                          &
     &    "gemm_index(F): M=", m, " N=", n, " K=", k,                   &
     &    " batch=", batchsize, " nrepeat=", nrepeat

        CALL libxs_init()

        ALLOCATE(a(stride_a * batchsize))
        ALLOCATE(b(stride_b * batchsize))
        ALLOCATE(c(stride_c * batchsize))
        ALLOCATE(ia(batchsize), ib(batchsize), ic(batchsize))

        ! Initialize matrices with simple values
        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(i)                      &
        !$OMP   SHARED(batchsize, stride_a, stride_b, stride_c, a, b, c)
        DO i = 1, batchsize
          a((i-1)*stride_a+1 : i*stride_a) = 1D0 / batchsize
          b((i-1)*stride_b+1 : i*stride_b) = 1D0 / batchsize
          c((i-1)*stride_c+1 : i*stride_c) = 0D0
        END DO

        ! Populate one-based element-offset index arrays (Fortran)
        DO i = 1, batchsize
          ia(i) = (i - 1) * stride_a + 1
          ib(i) = (i - 1) * stride_b + 1
          ic(i) = (i - 1) * stride_c + 1
        END DO

        ! Zero-initialize config, enable NOLOCK (unique C locations)
        config = libxs_gemm_config_t()
        config%flags = LIBXS_GEMM_FLAG_NOLOCK

        ! Dispatch MKL JIT kernel if available
#if defined(__MKL)
        jit_status = mkl_jit_create_dgemm(jitter,                       &
     &    'N', 'N', m, n, k, alpha, lda, ldb, beta, ldc)
        IF (MKL_JIT_SUCCESS .EQ. jit_status) THEN
          config%dgemm_jit = mkl_jit_get_dgemm_ptr(jitter)
          config%jitter = jitter
          WRITE(*, "(A)") "  MKL JIT kernel dispatched"
        END IF
#endif

        ! Warmup
        CALL libxs_gemm_index(                                          &
     &    C_LOC(a), C_LOC(ia), C_LOC(b), C_LOC(ib),                     &
     &    C_LOC(c), C_LOC(ic),                                          &
     &    INT(C_SIZEOF(ia(1))), 1,                                      &
     &    batchsize, config)

        gflops = 2D0 * REAL(m, 8) * REAL(n, 8) * REAL(k, 8) * 1D-9

        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
#if defined(_OPENMP)
          !$OMP PARALLEL DEFAULT(NONE)                                  &
          !$OMP   SHARED(a, ia, b, ib, c, ic,                           &
          !$OMP          batchsize, config)
          CALL libxs_gemm_index_task(                                   &
     &      C_LOC(a), C_LOC(ia), C_LOC(b), C_LOC(ib),                   &
     &      C_LOC(c), C_LOC(ic),                                        &
     &      INT(C_SIZEOF(ia(1))), 1,                                    &
     &      batchsize, config,                                          &
     &      omp_get_thread_num(), omp_get_num_threads())
          !$OMP END PARALLEL
#else
          CALL libxs_gemm_index(                                        &
     &      C_LOC(a), C_LOC(ia), C_LOC(b), C_LOC(ib),                   &
     &      C_LOC(c), C_LOC(ic),                                        &
     &      INT(C_SIZEOF(ia(1))), 1,                                    &
     &      batchsize, config)
#endif
        END DO
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)

        IF (0D0 .LT. duration) THEN
          WRITE(*, "(A,F10.3,A,I0,A)")                                  &
     &      "Total time : ", duration, " s (", nrepeat, " repeats)"
          WRITE(*, "(A,F10.1,A)")                                       &
     &      "Performance: ",                                            &
     &      gflops * batchsize * nrepeat / duration,                    &
     &      " GFLOPS/s"
        END IF

#if defined(__MKL)
        jit_status = mkl_jit_destroy(jitter)
#endif
        DEALLOCATE(a, b, c)
        DEALLOCATE(ia, ib, ic)
        CALL libxs_finalize()
      END PROGRAM
