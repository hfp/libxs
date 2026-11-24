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
     &    libxs_syrk_t, libxs_syrk_dispatch, libxs_syrk_release,        &
     &    libxs_syrk, libxs_syrk_task, libxs_syrk_ntasks,               &
     &    libxs_matdiff_t, libxs_matdiff, libxs_matdiff_clear,          &
     &    libxs_timer_tick, libxs_timer_duration,                       &
     &    libxs_init, libxs_finalize,                                   &
     &    C_LOC, C_PTR, C_NULL_PTR, C_INT
        USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: OUTPUT_UNIT
#if defined(__MPI)
        USE :: LIBXS_JIT, ONLY: libxs_pdist_t
#endif
!$      USE :: OMP_LIB, ONLY: omp_get_max_threads
        IMPLICIT NONE
#if defined(__MPI)
        INCLUDE 'mpif.h'
#endif

        ! implicit interface: the compiler passes the hidden lengths
        EXTERNAL :: DSYRK, DSYR2K

        INTEGER, PARAMETER :: T = KIND(0D0)
        ! seconds of a timed line unless nrepeat is given
        DOUBLE PRECISION, PARAMETER :: tbudget = 1D0
        INTEGER :: n, k, argc, nrepeat, direct, myrank
        INTEGER :: i, tasks, ntasks, nthreads, nsyrk
        CHARACTER(32) :: argv
        REAL(T), ALLOCATABLE, TARGET :: a(:,:), b(:,:), at(:,:)
        REAL(T), ALLOCATABLE, TARGET :: c(:,:), cref(:,:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a, b, at, c, cref
        REAL(T) :: alpha, beta
        TYPE(libxs_syrk_t) :: plan, plan2k, plant
        TYPE(libxs_matdiff_t) :: diff
        INTEGER(LIBXS_TIMER_TICK_KIND) :: t0, t1
        DOUBLE PRECISION :: gflops
        INTEGER :: iout
#if defined(__MPI)
        INTEGER(C_INT), ALLOCATABLE, TARGET :: ranks(:)
        REAL(T), ALLOCATABLE, TARGET :: cd(:,:), ct(:,:)
        TYPE(libxs_syrk_t) :: pland
        TYPE(libxs_pdist_t) :: dist
        TYPE(C_PTR) :: aptr
        REAL(T) :: derr(2), dmax(2)
        INTEGER :: ierr, level, nranks
        INTEGER :: nprow, npcol, myprow, mypcol
        INTEGER :: block, kstart, kloc, lrows, lcols, ldc, j, ig, jg
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
          nrepeat = 0
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
        myrank = 0
#if defined(__MPI)
        IF (6 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(6, argv)
          READ(argv, "(I32)") block
        ELSE
          block = 32
        END IF
        ! any task may communicate from MPI_THREAD_SERIALIZED on
        CALL MPI_INIT_THREAD(MPI_THREAD_MULTIPLE, level, ierr)
        CALL MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
        CALL MPI_COMM_SIZE(MPI_COMM_WORLD, nranks, ierr)
        ! the output of rank 0 (every rank runs the local sections)
        IF (0 .NE. myrank) OPEN(NEWUNIT=iout, STATUS="SCRATCH")
#endif

        nthreads = 1
!$      nthreads = omp_get_max_threads()
        ntasks = nthreads * MAX(tasks, 1)

        alpha = 1D0; beta = 1D0
        ! one triangle: an addition and a multiplication per element
        gflops = DBLE(n) * DBLE(n + 1) * DBLE(k) * 1D-9

        IF (0 .LT. nrepeat) THEN
          WRITE(argv, "(I0)") nrepeat
        ELSE
          argv = "auto"
        END IF
        WRITE(iout, "(A,I0,A,I0,A,A,A,I0)")                             &
     &    "SYRK: N=", n, " K=", k, " nrepeat=", TRIM(argv),             &
     &    " direct=", direct

        CALL libxs_init()

        ALLOCATE(a(n, k), b(n, k), at(k, n), c(n, n), cref(n, n))

        CALL RANDOM_NUMBER(a)
        CALL RANDOM_NUMBER(b)
        at = TRANSPOSE(a)

        IF (0 .EQ. libxs_syrk_dispatch(plan, LIBXS_DATATYPE_F64,        &
     &    'L', 'N', n, k, n, 0, n)                                      &
     &    .OR. 0 .EQ. libxs_syrk_dispatch(plan2k, LIBXS_DATATYPE_F64,   &
     &    'U', 'N', n, k, n, n, n)                                      &
     &    .OR. 0 .EQ. libxs_syrk_dispatch(plant, LIBXS_DATATYPE_F64,    &
     &    'U', 'T', n, k, k, 0, n))                                     &
     &  THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch"
          ERROR STOP 1
        END IF

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- SYRK (lower) ---"

        c = 0D0; cref = 0D0
        CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)
        IF (0 .EQ. direct) THEN
          CALL libxs_syrk(plan, alpha, C_LOC(a), beta, C_LOC(c))
        ELSE
          CALL libxs_syrk(LIBXS_DATATYPE_F64, 'L', 'N', n, k,           &
     &      alpha, C_LOC(a), n, beta, C_LOC(c), n)
        END IF
        CALL report("lower")

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- SYR2K (upper) ---"

        c = 0D0; cref = 0D0
        CALL DSYR2K('U', 'N', n, k, alpha, a, n, b, n, beta, cref, n)
        IF (0 .EQ. direct) THEN
          CALL libxs_syrk(plan2k, alpha, C_LOC(a), C_LOC(b),            &
     &      beta, C_LOC(c))
        ELSE
          CALL libxs_syrk(LIBXS_DATATYPE_F64, 'U', 'N', n, k,           &
     &      alpha, C_LOC(a), n, C_LOC(b), n, beta, C_LOC(c), n)
        END IF
        CALL report("upper")

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- SYRK (transposed, upper) ---"

        c = 0D0; cref = 0D0
        CALL DSYRK('U', 'T', n, k, alpha, at, k, beta, cref, n)
        IF (0 .EQ. direct) THEN
          CALL libxs_syrk(plant, alpha, C_LOC(at), beta, C_LOC(c))
        ELSE
          CALL libxs_syrk(LIBXS_DATATYPE_F64, 'U', 'T', n, k,           &
     &      alpha, C_LOC(at), k, beta, C_LOC(c), n)
        END IF
        CALL report("transposed")

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- libxs_syrk_task (OpenMP) ---"

        nsyrk = 1
!$      nsyrk = libxs_syrk_ntasks(plan, INT(nthreads, C_INT))
        WRITE(iout, "(A,I0,A,I0,A,I0)")                                 &
     &    "  threads=", nthreads, " ntasks=", ntasks,                   &
     &    " libxs_syrk=", nsyrk

        cref = 0D0
        CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)

        c = 0D0
        CALL run_tasks(nthreads)
        CALL report("omp")

        c = 0D0
        CALL run_tasks(ntasks)
        CALL report("omp tasks")

#if defined(__MPI)
        WRITE(iout, "(A)") ""
        WRITE(iout, "(A,I0,A,I0,A,I0,A,I0)")                            &
     &    "--- SYRK (distributed) --- ranks=", nranks,                  &
     &    " threads=", nthreads,                                        &
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
        ALLOCATE(cd(ldc, MAX(lcols, 1)), ct(ldc, MAX(lcols, 1)))
        cd = 0D0; ct = 0D0
        kstart = myrank * k / nranks + 1
        kloc = (myrank + 1) * k / nranks - kstart + 1
        aptr = C_NULL_PTR
        IF (0 .LT. kloc) aptr = C_LOC(a(1, kstart))
        dist%fcomm = MPI_COMM_WORLD
        dist%mb = block; dist%nb = block
        dist%nprow = nprow; dist%npcol = npcol
        dist%rsrc = 0; dist%csrc = 0
        dist%ranks = C_LOC(ranks)

        IF (0 .EQ. libxs_syrk_dispatch(pland, LIBXS_DATATYPE_F64,       &
     &    'L', 'N', n, kloc, n, 0, ldc, dist))                          &
     &  THEN
          WRITE(*, "(A)") "FAILED: libxs_syrk_dispatch (distributed)"
          ERROR STOP 1
        END IF
        CALL libxs_syrk(pland, alpha, aptr, 0D0, C_LOC(cd))

        ! the same bits with more tasks than threads
!$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(NONE) PRIVATE(i)            &
!$OMP&  SHARED(pland, alpha, aptr, ct, ntasks)
        DO i = 0, ntasks - 1
          CALL libxs_syrk_task(pland, alpha, aptr, 0D0, C_LOC(ct),      &
     &      i, ntasks)
        END DO
!$OMP END PARALLEL DO

        derr = 0D0
        DO j = 1, lcols
          jg = l2g(j, block, mypcol, npcol)
          DO i = 1, lrows
            ig = l2g(i, block, myprow, nprow)
            derr(2) = MAX(derr(2), ABS(cd(i, j) - ct(i, j)))
            IF (ig .GE. jg) THEN
              derr(1) = MAX(derr(1), ABS(cd(i, j) - cref(ig, jg)))
            ELSE
              derr(1) = MAX(derr(1), ABS(cd(i, j)))
            END IF
          END DO
        END DO
        CALL MPI_REDUCE(derr, dmax, 2, MPI_DOUBLE_PRECISION, MPI_MAX,   &
     &    0, MPI_COMM_WORLD, ierr)
        IF (0 .EQ. myrank) THEN
          WRITE(iout, "(A,E12.5)") "  max error (dist): ", dmax(1)
          WRITE(iout, "(A,E12.5)") "  max error (dist tasks): ", dmax(2)
        END IF
#endif

        WRITE(iout, "(A)") ""
        WRITE(iout, "(A)") "--- SYRK performance ---"

        cref = 0D0; c = 0D0
        CALL bench("DSYRK:")
        CALL bench("LIBXS:")
        CALL bench("OMP:  ")
        CALL bench("TASK: ")
#if defined(__MPI)
        CALL bench("DIST: ")
        CALL libxs_syrk_release(pland)
        DEALLOCATE(ranks, cd, ct)
#endif

        DEALLOCATE(a, b, at, c, cref)
        CALL libxs_finalize()
#if defined(__MPI)
        CALL MPI_FINALIZE(ierr)
#endif

      CONTAINS
        ! the local SYRK as count tasks, which are handed out on demand
        SUBROUTINE run_tasks(count)
          INTEGER, INTENT(IN) :: count
          INTEGER :: task
!$OMP PARALLEL DO SCHEDULE(DYNAMIC) DEFAULT(NONE) PRIVATE(task)         &
!$OMP&  SHARED(plan, alpha, beta, a, c, count)
          DO task = 0, count - 1
            CALL libxs_syrk_task(plan, alpha, C_LOC(a), beta,           &
     &        C_LOC(c), task, count)
          END DO
!$OMP END PARALLEL DO
        END SUBROUTINE

        ! largest difference to the reference
        SUBROUTINE report(title)
          CHARACTER(*), INTENT(IN) :: title
          CALL libxs_matdiff_clear(diff)
          CALL libxs_matdiff(diff, LIBXS_DATATYPE_F64, n, n,            &
     &      C_LOC(cref), C_LOC(c))
          WRITE(iout, "(A,E12.5)")                                      &
     &      "  max error (" // title // "): ", diff%linf_abs
        END SUBROUTINE

        ! one call of the line title: the local lines on rank 0 only
        ! (concurrent ranks would contend), the distributed on all ranks
        SUBROUTINE run(title)
          CHARACTER(*), INTENT(IN) :: title
          IF (0 .EQ. myrank .OR. "DIST: " .EQ. title) THEN
            SELECT CASE (title)
            CASE ("DSYRK:")
              CALL DSYRK('L', 'N', n, k, alpha, a, n, beta, cref, n)
            CASE ("LIBXS:")
              CALL libxs_syrk(plan, alpha, C_LOC(a), beta, C_LOC(c))
            CASE ("OMP:  ")
              CALL run_tasks(nthreads)
            CASE ("TASK: ")
              CALL run_tasks(ntasks)
#if defined(__MPI)
            CASE ("DIST: ")
              CALL libxs_syrk(pland, alpha, aptr, 0D0, C_LOC(cd))
#endif
            END SELECT
          END IF
        END SUBROUTINE

        ! a warm-up call, then nrepeat calls or the calls of about
        ! tbudget seconds (the same count on every rank)
        SUBROUTINE bench(title)
          CHARACTER(*), INTENT(IN) :: title
          DOUBLE PRECISION :: duration
          INTEGER :: nrep, r
#if defined(__MPI)
          CALL MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
          t0 = libxs_timer_tick()
          CALL run(title)
          t1 = libxs_timer_tick()
          duration = libxs_timer_duration(t0, t1)
#if defined(__MPI)
          CALL MPI_ALLREDUCE(MPI_IN_PLACE, duration, 1,                 &
     &      MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
#endif
          IF (0 .LT. nrepeat) THEN
            nrep = nrepeat
          ELSE IF (tbudget .LT. duration * 1D6) THEN
            nrep = MAX(NINT(tbudget / duration), 1)
          ELSE
            nrep = 1000000
          END IF
          t0 = libxs_timer_tick()
          DO r = 1, nrep
            CALL run(title)
          END DO
#if defined(__MPI)
          CALL MPI_BARRIER(MPI_COMM_WORLD, ierr)
#endif
          t1 = libxs_timer_tick()
          CALL timing(title, nrep)
        END SUBROUTINE

        ! time and rate of nrep calls between t0 and t1
        SUBROUTINE timing(title, nrep)
          CHARACTER(*), INTENT(IN) :: title
          INTEGER, INTENT(IN) :: nrep
          DOUBLE PRECISION :: duration
          duration = libxs_timer_duration(t0, t1)
          IF (0D0 < duration) THEN
            WRITE(iout, "(A,F10.3,A,I0,A)")                             &
     &        "  " // title, duration, " s (", nrep, " calls)"
            WRITE(iout, "(A,F10.1,A)")                                  &
     &        "        ", gflops * DBLE(nrep) / duration,               &
     &        " GFLOPS/s"
          END IF
        END SUBROUTINE

#if defined(__MPI)
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
