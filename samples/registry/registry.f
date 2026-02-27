!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.)
!=======================================================================!

! Microbenchmark for the registry (key-value store).
! Measures registration (write) and cold/cached lookup.
! This is a simplified Fortran version of registry.c.
!
      PROGRAM registry
        USE :: LIBXS, ONLY: C_PTR, C_INT, C_DOUBLE,                     &
     &    C_SIZE_T, C_LOC, C_SIZEOF, C_ASSOCIATED,                      &
     &    LIBXS_REGINFO,                                                &
     &    libxs_init, libxs_finalize, libxs_registry_create,            &
     &    libxs_registry_destroy, libxs_registry_set,                   &
     &    libxs_registry_get, libxs_registry_has,                       &
     &    libxs_registry_info
        IMPLICIT NONE

        INTEGER, PARAMETER :: ntotal = 10000                            &
     &  , nrepeat = 10

        !> Key type (binary-safe); fields match registry.c.
        TYPE, BIND(C) :: bench_key_t
          INTEGER(C_INT) :: id, tag, pad
        END TYPE

        !> Value type carrying two doubles.
        TYPE, BIND(C) :: bench_value_t
          REAL(C_DOUBLE) :: data(2)
        END TYPE

        TYPE(C_PTR)    :: reg, cptr
        TYPE(bench_key_t),   TARGET :: keys(ntotal)
        TYPE(bench_value_t), TARGET :: vals(ntotal)
        TYPE(LIBXS_REGINFO) :: info
        DOUBLE PRECISION :: start, twrite, tcold, tcached
        INTEGER :: i, j, n, result

        result = 0
        ! initialise keys and values
        DO i = 1, ntotal
          keys(i)%id  = i - 1
          keys(i)%tag = IEOR(i - 1, INT(Z'ABCD'))
          keys(i)%pad = 0
          vals(i)%data(1) = DBLE(i - 1)
          vals(i)%data(2) = DBLE((i - 1) * 2)
        END DO

        WRITE(*, "(A,I0,A,I0,A)")                                       &
     &    "Registry benchmark: ", ntotal, " keys, ",                    &
     &    nrepeat, " repeats"

        CALL libxs_init()

      ! (1) Registration: insert all keys
        IF (libxs_registry_create(reg) .NE. 0) THEN
          WRITE(*, "(A)") "FAILED to create registry"
          STOP 1
        END IF

        CALL CPU_TIME(start)
        DO i = 1, ntotal
          cptr = libxs_registry_set(reg,                                &
     &      C_LOC(keys(i)),                                             &
     &      INT(C_SIZEOF(keys(i)), C_SIZE_T),                           &
     &      C_LOC(vals(i)),                                             &
     &      INT(C_SIZEOF(vals(i)), C_SIZE_T))
          IF (.NOT.C_ASSOCIATED(cptr)) THEN
            result = 1
            EXIT
          END IF
        END DO
        CALL CPU_TIME(twrite)
        twrite = twrite - start

        IF (result .NE. 0) THEN
          WRITE(*, "(A)") "FAILED during registration"
          CALL libxs_registry_destroy(reg)
          STOP 1
        END IF

        IF (libxs_registry_info(reg, info) .EQ. 0) THEN
          WRITE(*, "(1A,A,I0,A,I0,A,I0)")                               &
     &      CHAR(9), "size=", info%size,                                &
     &      " capacity=", info%capacity,                                &
     &      " nbytes=", info%nbytes
        END IF

      ! (2) Cold lookup: sequential (defeats TLS cache by spread)
        CALL CPU_TIME(start)
        DO n = 1, nrepeat
          DO i = 1, ntotal
            j = MOD(i * 7, ntotal) + 1
            cptr = libxs_registry_get(reg,                              &
     &        C_LOC(keys(j)),                                           &
     &        INT(C_SIZEOF(keys(j)), C_SIZE_T))
            IF (.NOT.C_ASSOCIATED(cptr)) THEN
              result = 1
              EXIT
            END IF
          END DO
          IF (result .NE. 0) EXIT
        END DO
        CALL CPU_TIME(tcold)
        tcold = tcold - start

      ! (3) Cached lookup: cycle through a small set
        CALL CPU_TIME(start)
        DO n = 1, nrepeat
          DO i = 1, ntotal
            j = MOD(i - 1, 16) + 1
            cptr = libxs_registry_get(reg,                              &
     &        C_LOC(keys(j)),                                           &
     &        INT(C_SIZEOF(keys(j)), C_SIZE_T))
            IF (.NOT.C_ASSOCIATED(cptr)) THEN
              result = 1
              EXIT
            END IF
          END DO
          IF (result .NE. 0) EXIT
        END DO
        CALL CPU_TIME(tcached)
        tcached = tcached - start

      ! (4) Verify some entries via libxs_registry_has
        DO i = 1, ntotal
          IF (libxs_registry_has(reg,                                   &
     &      C_LOC(keys(i)),                                             &
     &      INT(C_SIZEOF(keys(i)), C_SIZE_T)) .EQ. 0)                   &
     &    THEN
            result = 1
            EXIT
          END IF
        END DO

        CALL libxs_registry_destroy(reg)

        IF (result .NE. 0) THEN
          WRITE(*, "(A)") "FAILED"
          STOP 1
        END IF

        IF (0.LT.twrite) THEN
          WRITE(*, "(1A,A,F10.1,A)")                                    &
     &      CHAR(9), "registration:   ",                                &
     &      1D9 * twrite / ntotal, " ns/op"
        END IF
        IF (0.LT.tcold) THEN
          WRITE(*, "(1A,A,F10.1,A)")                                    &
     &      CHAR(9), "cold lookup:    ",                                &
     &      1D9 * tcold / DBLE(ntotal * nrepeat), " ns/op"
        END IF
        IF (0.LT.tcached) THEN
          WRITE(*, "(1A,A,F10.1,A)")                                    &
     &      CHAR(9), "cached lookup:  ",                                &
     &      1D9 * tcached / DBLE(ntotal * nrepeat), " ns/op"
        END IF

        CALL libxs_finalize()
        WRITE(*, "(A)") "Finished"
      END PROGRAM
