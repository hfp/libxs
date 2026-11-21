!=======================================================================!
! Copyright (c) 2009-2026 Hans Pabst                                    !
! Copyright (c) 2009-2026 Intel Corporation                             !
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
     &    libxs_syrk_dispatch, libxs_syrk, libxs_syrk_task,             &
     &    libxs_syrk_ntasks,                                            &
     &    libxs_syr2k_dispatch, libxs_syr2k,                            &
     &    libxs_matdiff_t, libxs_matdiff, libxs_matdiff_clear,          &
     &    libxs_timer_tick, libxs_timer_duration,                       &
     &    libxs_init, libxs_finalize,                                   &
     &    C_LOC, C_PTR, C_NULL_PTR, C_ASSOCIATED,                       &
     &    C_F_POINTER, C_INT
        USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: OUTPUT_UNIT
#if defined(__MPI)
        USE :: LIBXS_JIT, ONLY: libxs_syrk_dist_t, libxs_pdist_t
#endif
!$      USE :: OMP_LIB, ONLY: omp_get_thread_num,                       &
!$   &    omp_get_num_threads, omp_get_max_threads
        IMPLICIT NONE
#if defined(__MPI)
        INCLUDE 'mpif.h'
#endif

        ! implicit interface: the compiler passes the hidden lengths
        EXTERNAL :: DSYRK, DSYR2K

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: n, k, argc, r, nrepeat, direct
        INTEGER :: i, tid, nt, tasks, ntasks, nthreads, nsyrk
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
        INTEGER :: iout
#if defined(__MPI)
        INTEGER(C_INT), ALLOCATABLE, TARGET :: ranks(:)
        REAL(T), ALLOCATABLE, TARGET :: cd(:,:)
        TYPE(libxs_syrk_dist_t) :: dist
        TYPE(libxs_pdist_t) :: cdist
        TYPE(C_PTR) :: aptr
        REAL(T) :: derr, dmax
        INTEGER :: ierr, myrank, nranks, nprow, npcol, myprow, mypcol
        INTEGER :: block, kstart, kloc, lrows, lcols, ldc, j
#endif

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
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") tasks
        ELSE
          tasks = 4
        END IF

        iout = OUTPUT_UNIT
#if defined(__MPI)
        IF (6 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(6, argv)
          READ(argv, "(I32)") block
        ELSE
          block = 32
        END IF
        CALL MPI_INIT(ierr)
        CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        ! the output of rank 0 (every rank runs the local sections)
        IF (0 .NE. myrank) OPEN(NEWUNIT=iout, STATUS="SCRATCH")
#endif

        nthreads = 1
!$      nthreads = omp_get_max_threads()
        ntasks = nthreads * MAX(tasks, 1)

        alpha = 1D0; beta = 1D0

        WRITE(iout, "(A,I0,A,I0,A,I0,A,I0)")                            &
     &    "SYRK: N=", n, " K=", k, " nrepeat=", nrepeat,                &
     &    " direct=", direct

        CALL libxs_init()

        ALLOCATE(a(n, k), b(n, k), c(n, n), cref(n, n))

        CALL RANDOM_NUMBER(a)
        CALL RANDOM_NUMBER(b)
        c = 0D0; cref = 0D0

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- libxs_syrk (lower) ---"

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
        WRITE(iout, "(A,E12.5)")                                        &
     &    "  max error (lower): ", diff%linf_abs

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- libxs_syr2k (upper) ---"

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
        WRITE(iout, "(A,E12.5)")                                        &
     &    "  max error (upper): ", diff%linf_abs

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- libxs_syrk_task (OpenMP) ---"

        ptr = libxs_syrk_dispatch(LIBXS_DATATYPE_F64, n, k, n, n)
        IF (.NOT. C_ASSOCIATED(ptr)) THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch returned NULL"
          ERROR STOP 1
        END IF
        CALL C_F_POINTER(ptr, config)
        nsyrk = 1
!$      nsyrk = libxs_syrk_ntasks(config, INT(nthreads, C_INT))
        WRITE(iout, "(A,I0,A,I0,A,I0)")                                 &
     &    "  threads=", nthreads, " ntasks=", ntasks,                   &
     &    " libxs_syrk=", nsyrk

        cref = 0D0
        CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)

        c = 0D0
!$OMP PARALLEL DEFAULT(NONE) PRIVATE(tid, nt)                           &
!$OMP&  SHARED(config, alpha, beta, a, c)
        tid = 0
        nt = 1
!$      tid = omp_get_thread_num()
!$      nt = omp_get_num_threads()
        CALL libxs_syrk_task(config, 'L', alpha, beta,                  &
     &    C_LOC(a), C_LOC(c), tid, nt)
!$OMP END PARALLEL

        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(iout, "(A,E12.5)")                                        &
     &    "  max error (omp): ", diff%linf_abs

        c = 0D0
!$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(DYNAMIC) PRIVATE(i)            &
!$OMP&  SHARED(config, alpha, beta, a, c, ntasks)
        DO i = 0, ntasks - 1
          CALL libxs_syrk_task(config, 'L', alpha, beta,                &
     &      C_LOC(a), C_LOC(c), i, ntasks)
        END DO
!$OMP END PARALLEL DO

        CALL libxs_matdiff_clear(diff)
        CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,              &
     &    C_LOC(cref), C_LOC(c))
        WRITE(iout, "(A,E12.5)")                                        &
     &    "  max error (omp tasks): ", diff%linf_abs

#if defined(__MPI)
        WRITE(iout, "(A)") ""
        WRITE(iout, "(A,I0,A,I0,A,I0)")                                 &
     &    "--- libxs_syrk (distributed) --- ranks=", nranks,            &
     &    " block=", block, " K/rank=", k / nranks

        ! every rank: the same A and the full reference
        CALL MPI_BCAST(a, n * k, MPI_DOUBLE_PRECISION, 0,               &
     &    MPI_COMM_WORLD, ierr)
        cref = 0D0
        CALL DSYRK('L', 'N', n, k, alpha, a, n, 0D0, cref, n)

        ! K in stripes, C block-cyclic on a near-square grid
        nprow = MAX(INT(SQRT(DBLE(nranks))), 1)
        DO WHILE (0 .NE. MOD(nranks, nprow))
          nprow = nprow - 1
        END DO
        npcol = nranks / nprow
        myprow = MOD(myrank, nprow)
        mypcol = myrank / nprow
        ALLOCATE(ranks(nranks))
        DO j = 1, nranks
          ranks(j) = j - 1
        END DO
        lrows = numroc(n, block, myprow, nprow)
        lcols = numroc(n, block, mypcol, npcol)
        ldc = MAX(lrows, 1)
        ALLOCATE(cd(ldc, MAX(lcols, 1)))
        cd = 0D0
        kstart = myrank * k / nranks + 1
        kloc = (myrank + 1) * k / nranks - kstart + 1
        aptr = C_LOC(a)
        IF (0 .LT. kloc) aptr = C_LOC(a(1, kstart))
        cdist%m = n; cdist%n = n
        cdist%mb = block; cdist%nb = block
        cdist%nprow = nprow; cdist%npcol = npcol
        cdist%rsrc = 0; cdist%csrc = 0
        cdist%ranks = C_LOC(ranks)

        IF (0 .EQ. libxs_syrk_dispatch(dist, LIBXS_DATATYPE_F64,        &
     &    n, kloc, n, ldc, cdist, MPI_COMM_WORLD))                      &
     &  THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch (distributed)"
          ERROR STOP 1
        END IF
        CALL libxs_syrk(dist, 'L', alpha, 0D0, aptr, C_LOC(cd))

        derr = 0D0
        DO j = 1, lcols
          DO i = 1, lrows
            IF (l2g(i, block, myprow, nprow) .GE.                       &
     &          l2g(j, block, mypcol, npcol))                           &
     &      THEN
              derr = MAX(derr, ABS(cd(i, j)                             &
     &          - cref(l2g(i, block, myprow, nprow),                    &
     &                 l2g(j, block, mypcol, npcol))))
            END IF
          END DO
        END DO
        CALL MPI_REDUCE(derr, dmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX,   &
     &    0, MPI_COMM_WORLD, ierr)
        WRITE(iout, "(A,E12.5)")                                        &
     &    "  max error (dist): ", dmax

        gflops = DBLE(n) * DBLE(n) * DBLE(k) * 2D-9
        CALL MPI_BARRIER(MPI_COMM_WORLD, ierr)
        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
          CALL libxs_syrk(dist, 'L', alpha, 0D0, aptr, C_LOC(cd))
        END DO
        CALL MPI_BARRIER(MPI_COMM_WORLD, ierr)
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)
        IF (0D0 < duration) THEN
          WRITE(iout, "(A,F10.3,A,I0,A)")                               &
     &      "  DIST: ", duration, " s (", nrepeat, " calls)"
          WRITE(iout, "(A,F10.1,A)")                                    &
     &      "        ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF
        DEALLOCATE(ranks, cd)
#endif

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- SYRK performance ---"

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
          WRITE(iout, "(A,F10.3,A,I0,A)")                               &
     &      "  DSYRK:", duration, " s (", nrepeat, " calls)"
          WRITE(iout, "(A,F10.1,A)")                                    &
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
          WRITE(iout, "(A,F10.3,A,I0,A)")                               &
     &      "  LIBXS:", duration, " s (", nrepeat, " calls)"
          WRITE(iout, "(A,F10.1,A)")                                    &
     &      "        ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF

        c = 0D0
        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
!$OMP PARALLEL DEFAULT(NONE) PRIVATE(tid, nt)                           &
!$OMP&  SHARED(config, alpha, beta, a, c)
          tid = 0
          nt = 1
!$        tid = omp_get_thread_num()
!$        nt = omp_get_num_threads()
          CALL libxs_syrk_task(config, 'L', alpha, beta,                &
     &      C_LOC(a), C_LOC(c), tid, nt)
!$OMP END PARALLEL
        END DO
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)
        IF (0D0 < duration) THEN
          WRITE(iout, "(A,F10.3,A,I0,A)")                               &
     &      "  OMP:  ", duration, " s (", nrepeat, " calls)"
          WRITE(iout, "(A,F10.1,A)")                                    &
     &      "        ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF

        c = 0D0
        t0 = libxs_timer_tick()
        DO r = 1, nrepeat
!$OMP PARALLEL DO DEFAULT(NONE) SCHEDULE(DYNAMIC) PRIVATE(i)            &
!$OMP&  SHARED(config, alpha, beta, a, c, ntasks)
          DO i = 0, ntasks - 1
            CALL libxs_syrk_task(config, 'L', alpha, beta,              &
     &        C_LOC(a), C_LOC(c), i, ntasks)
          END DO
!$OMP END PARALLEL DO
        END DO
        t1 = libxs_timer_tick()
        duration = libxs_timer_duration(t0, t1)
        IF (0D0 < duration) THEN
          WRITE(iout, "(A,F10.3,A,I0,A)")                               &
     &      "  TASK: ", duration, " s (", nrepeat, " calls)"
          WRITE(iout, "(A,F10.1,A)")                                    &
     &      "        ", gflops * DBLE(nrepeat) / duration,              &
     &      " GFLOPS/s"
        END IF

        DEALLOCATE(a, b, c, cref)
        CALL libxs_finalize()
#if defined(__MPI)
        CALL MPI_FINALIZE(ierr)

      CONTAINS
        ! rows (or columns) of a block-cyclic distribution with block
        ! size nb that grid row (or column) iproc holds (as NUMROC)
        PURE INTEGER FUNCTION numroc(nglobal, nb, iproc, nprocs)
          INTEGER, INTENT(IN) :: nglobal, nb, iproc, nprocs
          INTEGER :: nblocks, extra
          nblocks = nglobal / nb
          extra = MOD(nblocks, nprocs)
          numroc = (nblocks / nprocs) * nb
          IF (iproc .LT. extra) THEN
            numroc = numroc + nb
          ELSE IF (iproc .EQ. extra) THEN
            numroc = numroc + MOD(nglobal, nb)
          END IF
        END FUNCTION

        ! global index of the local index il (as INDXL2G, one-based)
        PURE INTEGER FUNCTION l2g(il, nb, iproc, nprocs)
          INTEGER, INTENT(IN) :: il, nb, iproc, nprocs
          l2g = ((il - 1) / nb * nprocs + iproc) * nb                   &
     &      + MOD(il - 1, nb) + 1
        END FUNCTION
#endif
      END PROGRAM
