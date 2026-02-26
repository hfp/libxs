!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXS library.                               !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxs/                    !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!

      MODULE LIBXS
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &    C_DOUBLE, C_FLOAT, C_LONG_LONG, C_INT,                        &
     &    C_CHAR, C_INT8_T, C_SIZE_T, C_PTR, C_NULL_PTR,                &
     &    C_F_POINTER, C_ASSOCIATED, C_LOC, C_SIZEOF
        IMPLICIT NONE
        PRIVATE

        !> Public API: constants.
        PUBLIC :: LIBXS_TIMER_TICK_KIND
        PUBLIC :: LIBXS_DATATYPE_F64, LIBXS_DATATYPE_F32
        PUBLIC :: LIBXS_DATATYPE_I64, LIBXS_DATATYPE_U64
        PUBLIC :: LIBXS_DATATYPE_I32, LIBXS_DATATYPE_U32
        PUBLIC :: LIBXS_DATATYPE_I16, LIBXS_DATATYPE_U16
        PUBLIC :: LIBXS_DATATYPE_I8,  LIBXS_DATATYPE_U8
        PUBLIC :: LIBXS_DATATYPE_UNKNOWN
        PUBLIC :: LIBXS_TARGET_ARCH_UNKNOWN
        PUBLIC :: LIBXS_TARGET_ARCH_GENERIC
        PUBLIC :: LIBXS_X86_GENERIC, LIBXS_X86_SSE3
        PUBLIC :: LIBXS_X86_SSE42, LIBXS_X86_AVX
        PUBLIC :: LIBXS_X86_AVX2, LIBXS_X86_AVX512
        PUBLIC :: LIBXS_X86_ALLFEAT
        PUBLIC :: LIBXS_AARCH64, LIBXS_AARCH64_SVE128
        PUBLIC :: LIBXS_AARCH64_SVE256, LIBXS_AARCH64_SVE512
        PUBLIC :: LIBXS_AARCH64_ALLFEAT

        !> Public API: types and procedures.
        PUBLIC :: LIBXS_MATDIFF_INFO
        PUBLIC :: LIBXS_REGINFO
        PUBLIC :: libxs_init, libxs_finalize
        PUBLIC :: libxs_timer_tick, libxs_timer_duration
        PUBLIC :: libxs_malloc, libxs_free
        PUBLIC :: libxs_malloc_pool, libxs_free_pool
        PUBLIC :: libxs_hash, libxs_hash_string
        PUBLIC :: libxs_diff
        PUBLIC :: libxs_matdiff, libxs_matdiff_reduce
        PUBLIC :: libxs_matdiff_clear, libxs_matdiff_epsilon
        PUBLIC :: libxs_registry_create
        PUBLIC :: libxs_registry_destroy
        PUBLIC :: libxs_registry_set
        PUBLIC :: libxs_registry_get
        PUBLIC :: libxs_registry_has
        PUBLIC :: libxs_registry_remove
        PUBLIC :: libxs_registry_info
        PUBLIC :: libxs_typesize

        !> Re-exported from ISO_C_BINDING for convenience.
        PUBLIC :: C_DOUBLE, C_FLOAT, C_INT, C_LONG_LONG
        PUBLIC :: C_CHAR, C_INT8_T, C_SIZE_T
        PUBLIC :: C_PTR, C_NULL_PTR
        PUBLIC :: C_F_POINTER, C_ASSOCIATED, C_LOC, C_SIZEOF

        !> Integer kind used by timer interface.
        INTEGER(C_INT), PARAMETER :: LIBXS_TIMER_TICK_KIND = C_LONG_LONG

        !> Enumerates element/data types.
        !> The raw value encodes type-size in bits [7:4].
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXS_DATATYPE_F64     = IOR( 0, ISHFT( 8, 4)),               &
     &    LIBXS_DATATYPE_F32     = IOR( 1, ISHFT( 4, 4)),               &
     &    LIBXS_DATATYPE_I64     = IOR( 2, ISHFT( 8, 4)),               &
     &    LIBXS_DATATYPE_U64     = IOR( 3, ISHFT( 8, 4)),               &
     &    LIBXS_DATATYPE_I32     = IOR( 4, ISHFT( 4, 4)),               &
     &    LIBXS_DATATYPE_U32     = IOR( 5, ISHFT( 4, 4)),               &
     &    LIBXS_DATATYPE_I16     = IOR( 6, ISHFT( 2, 4)),               &
     &    LIBXS_DATATYPE_U16     = IOR( 7, ISHFT( 2, 4)),               &
     &    LIBXS_DATATYPE_I8      = IOR( 8, ISHFT( 1, 4)),               &
     &    LIBXS_DATATYPE_U8      = IOR( 9, ISHFT( 1, 4)),               &
     &    LIBXS_DATATYPE_UNKNOWN = 10

        !> Enumerates the available target architectures and ISA
        !> extensions as returned by libxs_cpuid.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXS_TARGET_ARCH_UNKNOWN   = 0,                              &
     &    LIBXS_TARGET_ARCH_GENERIC   = 1,                              &
     &    LIBXS_X86_GENERIC           = 1002,                           &
     &    LIBXS_X86_SSE3              = 1003,                           &
     &    LIBXS_X86_SSE42             = 1004,                           &
     &    LIBXS_X86_AVX               = 1005,                           &
     &    LIBXS_X86_AVX2              = 1006,                           &
     &    LIBXS_X86_AVX512            = 1100,                           &
     &    LIBXS_X86_ALLFEAT           = 1999,                           &
     &    LIBXS_AARCH64               = 2001,                           &
     &    LIBXS_AARCH64_SVE128        = 2201,                           &
     &    LIBXS_AARCH64_SVE256        = 2301,                           &
     &    LIBXS_AARCH64_SVE512        = 2401,                           &
     &    LIBXS_AARCH64_ALLFEAT       = 2999

        !> Structure of differences with matrix norms according
        !> to http://www.netlib.org/lapack/lug/node75.html).
        TYPE, BIND(C) :: LIBXS_MATDIFF_INFO
          REAL(C_DOUBLE) :: norm1_abs, norm1_rel !! One-norm
          REAL(C_DOUBLE) :: normi_abs, normi_rel !! Infinity-norm
          REAL(C_DOUBLE) :: normf_rel            !! Froebenius-norm
          !> Maximum difference, L2-norm (absolute and relative),
          !> and R-squared.
          REAL(C_DOUBLE) :: linf_abs, linf_rel, l2_abs, l2_rel, rsq
          !> Statistics: sum/l1, min, max, avg, variance.
          REAL(C_DOUBLE) :: l1_ref, min_ref, max_ref, avg_ref, var_ref
          REAL(C_DOUBLE) :: l1_tst, min_tst, max_tst, avg_tst, var_tst
          !> Values(v_ref, v_tst) at location of largest linf_abs.
          REAL(C_DOUBLE) :: v_ref, v_tst
          !> Location (m, n) and reduction index (i, r).
          INTEGER(C_INT) :: m, n, i, r
        END TYPE

        !> Registry status information.
        TYPE, BIND(C) :: LIBXS_REGINFO
          INTEGER(C_SIZE_T) :: capacity, size, nbytes
        END TYPE

        INTERFACE
          !> Initialize the library.
          SUBROUTINE libxs_init() BIND(C)
          END SUBROUTINE

          !> De-initialize the library and free internal memory.
          SUBROUTINE libxs_finalize() BIND(C)
          END SUBROUTINE

          !> Returns the current tick of a monotonic timer source.
          !> Uses a platform-specific resolution; convert to
          !> seconds with libxs_timer_duration.
          INTEGER(LIBXS_TIMER_TICK_KIND)                                &
     &    FUNCTION libxs_timer_tick() BIND(C)
            IMPORT :: LIBXS_TIMER_TICK_KIND
          END FUNCTION

          !> Returns the duration (in seconds) between two ticks
          !> received by libxs_timer_tick.
          FUNCTION libxs_timer_duration(tick0, tick1) BIND(C)
            IMPORT :: LIBXS_TIMER_TICK_KIND, C_DOUBLE
            INTEGER(LIBXS_TIMER_TICK_KIND), INTENT(IN), VALUE :: tick0
            INTEGER(LIBXS_TIMER_TICK_KIND), INTENT(IN), VALUE :: tick1
            REAL(C_DOUBLE) :: libxs_timer_duration
          END FUNCTION

          !> Internal binding (use libxs_malloc instead).
          FUNCTION libxs_malloc_c(nbytes, alignment)                    &
     &    BIND(C, NAME="libxs_malloc")
            IMPORT :: C_SIZE_T, C_PTR
            INTEGER(C_SIZE_T), INTENT(IN), VALUE :: nbytes
            INTEGER(C_SIZE_T), INTENT(IN), VALUE :: alignment
            TYPE(C_PTR) :: libxs_malloc_c
          END FUNCTION

          !> Free memory allocated by libxs_malloc.
          SUBROUTINE libxs_free(ptr) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: ptr
          END SUBROUTINE

          !> Allocate the pool for libxs_malloc.
          SUBROUTINE libxs_malloc_pool() BIND(C)
          END SUBROUTINE

          !> Free unused memory (pool).
          SUBROUTINE libxs_free_pool() BIND(C)
          END SUBROUTINE

          !> Calculate a 64-bit hash for a character string.
          FUNCTION libxs_hash_string(string) BIND(C)
            IMPORT :: C_CHAR, C_LONG_LONG
            CHARACTER(C_CHAR), INTENT(IN) :: string(*)
            INTEGER(C_LONG_LONG) :: libxs_hash_string
          END FUNCTION

          !> Calculate a hash value for the given data.
          PURE FUNCTION libxs_hash_c(data, size, seed)                  &
     &    BIND(C, NAME="libxs_hash")
            IMPORT :: C_INT, C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: data
            INTEGER(C_INT), INTENT(IN), VALUE :: size
            INTEGER(C_INT), INTENT(IN), VALUE :: seed
            INTEGER(C_INT) :: libxs_hash_c
          END FUNCTION

          !> Compare two memory regions (binds to libxs_memcmp).
          PURE FUNCTION libxs_memcmp(a, b, nbytes)                      &
     &    BIND(C) RESULT(diff)
            IMPORT :: C_PTR, C_SIZE_T, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE      :: a, b
            INTEGER(C_SIZE_T), INTENT(IN), VALUE :: nbytes
            INTEGER(C_INT) :: diff
          END FUNCTION

          !> Reduces matdiff info (max function). Initialize
          !> output with libxs_matdiff_clear first.
          PURE SUBROUTINE libxs_matdiff_reduce(output, input)           &
     &    BIND(C)
            IMPORT :: LIBXS_MATDIFF_INFO
            TYPE(LIBXS_MATDIFF_INFO), INTENT(INOUT) :: output
            TYPE(LIBXS_MATDIFF_INFO), INTENT(IN)    :: input
          END SUBROUTINE

          !> Clears the given info-structure, e.g., for the initial
          !> reduction-value (libxs_matdiff_reduce).
          PURE SUBROUTINE libxs_matdiff_clear(info) BIND(C)
            IMPORT :: LIBXS_MATDIFF_INFO
            TYPE(LIBXS_MATDIFF_INFO), INTENT(OUT) :: info
          END SUBROUTINE

          !> Combine absolute and relative norms into a single
          !> value which can be used to check against a margin.
          FUNCTION libxs_matdiff_epsilon(input) BIND(C)
            IMPORT :: LIBXS_MATDIFF_INFO, C_DOUBLE
            TYPE(LIBXS_MATDIFF_INFO), INTENT(IN) :: input
            REAL(C_DOUBLE) :: libxs_matdiff_epsilon
          END FUNCTION
          !> Create a registry (key-value store).
          !> Returns 0 on success.
          FUNCTION libxs_registry_create(registry)                      &
     &    BIND(C)
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(OUT) :: registry
            INTEGER(C_INT) :: libxs_registry_create
          END FUNCTION
          !> Destroy registry and release all entries.
          SUBROUTINE libxs_registry_destroy(registry)                   &
     &    BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
          END SUBROUTINE
          !> Register a key-value pair. Returns a pointer
          !> to the stored value, or C_NULL_PTR on failure.
          FUNCTION libxs_registry_set(registry,                         &
     &    key, key_size, value_init, value_size)                        &
     &    BIND(C)
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: value_init
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      value_size
            TYPE(C_PTR) :: libxs_registry_set
          END FUNCTION
          !> Query a value by key. Returns C_NULL_PTR
          !> if the key is not found.
          FUNCTION libxs_registry_get(registry,                         &
     &    key, key_size) BIND(C)
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR) :: libxs_registry_get
          END FUNCTION
          !> Check if a key exists (non-zero if found).
          FUNCTION libxs_registry_has(registry,                         &
     &    key, key_size) BIND(C)
            IMPORT :: C_PTR, C_SIZE_T, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            INTEGER(C_INT) :: libxs_registry_has
          END FUNCTION
          !> Remove a key-value pair from the registry.
          SUBROUTINE libxs_registry_remove(registry,                    &
     &    key, key_size) BIND(C)
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
          END SUBROUTINE
          !> Get registry information.
          !> Returns 0 on success.
          FUNCTION libxs_registry_info(registry,                        &
     &    info) BIND(C)
            IMPORT :: C_PTR, C_INT, LIBXS_REGINFO
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(LIBXS_REGINFO), INTENT(OUT) :: info
            INTEGER(C_INT) :: libxs_registry_info
          END FUNCTION
        END INTERFACE

        !> Allocate memory (alignment=0: automatic).
        INTERFACE libxs_malloc
          MODULE PROCEDURE libxs_malloc_bytes
        END INTERFACE

        !> Calculate a hash value for a given array and seed.
        INTERFACE libxs_hash
          MODULE PROCEDURE libxs_hash_char, libxs_hash_i8
          MODULE PROCEDURE libxs_hash_i32, libxs_hash_i64
        END INTERFACE

        !> Check if two arrays differ (.TRUE. if different).
        INTERFACE libxs_diff
          MODULE PROCEDURE libxs_diff_char, libxs_diff_i8
          MODULE PROCEDURE libxs_diff_i32, libxs_diff_i64
        END INTERFACE

      CONTAINS
        !> Extract type-size (in Bytes) from datatype enum.
        ELEMENTAL FUNCTION libxs_typesize(datatype)
          INTEGER(C_INT), INTENT(IN) :: datatype
          INTEGER(C_INT) :: libxs_typesize
          libxs_typesize = ISHFT(datatype, -4)
        END FUNCTION

        !> Allocate from a pool reaching steady-state.
        !> alignment=0 (default): automatic, based on size.
        FUNCTION libxs_malloc_bytes(nbytes, alignment)
          INTEGER(C_SIZE_T), INTENT(IN) :: nbytes
          INTEGER(C_SIZE_T), INTENT(IN), OPTIONAL :: alignment
          TYPE(C_PTR) :: libxs_malloc_bytes
          IF (PRESENT(alignment)) THEN
            libxs_malloc_bytes =                                        &
     &        libxs_malloc_c(nbytes, alignment)
          ELSE
            libxs_malloc_bytes =                                        &
     &        libxs_malloc_c(nbytes, 0_C_SIZE_T)
          END IF
        END FUNCTION

        !> Matdiff wrapper with OPTIONAL arguments.
        SUBROUTINE libxs_matdiff(info, datatype, m, n,                  &
     &  ref, tst, ldref, ldtst)
          INTEGER(C_INT), INTENT(IN) :: datatype
          INTEGER(C_INT), INTENT(IN) :: m
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: n, ldref, ldtst
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: ref, tst
          TYPE(LIBXS_MATDIFF_INFO), INTENT(OUT) :: info
          INTEGER(C_INT) :: nn, rc
          TYPE(C_PTR) :: rr, tt
          INTERFACE
            FUNCTION internal_matdiff(info,                             &
     &      datatype, m, n, ref, tst, ldref, ldtst)                     &
     &      RESULT(res) BIND(C, NAME="libxs_matdiff")
              IMPORT :: LIBXS_MATDIFF_INFO, C_PTR, C_INT
              INTEGER(C_INT), INTENT(IN), VALUE         :: datatype
              INTEGER(C_INT), INTENT(IN), VALUE         :: m, n
              TYPE(C_PTR), INTENT(IN), VALUE            :: ref, tst
              INTEGER(C_INT), INTENT(IN)                :: ldref, ldtst
              TYPE(LIBXS_MATDIFF_INFO), INTENT(OUT)     :: info
              INTEGER(C_INT) :: res
            END FUNCTION
          END INTERFACE
          IF (PRESENT(n)) THEN; nn = n; ELSE; nn = m; END IF
          IF (PRESENT(ref)) THEN; rr = ref
          ELSE; rr = C_NULL_PTR; END IF
          IF (PRESENT(tst)) THEN; tt = tst
          ELSE; tt = C_NULL_PTR; END IF
          rc = internal_matdiff(info, datatype, m, nn,                  &
     &      rr, tt, ldref, ldtst)
        END SUBROUTINE

        !> Calculates a hash value for the given array and seed.
        FUNCTION libxs_hash_char(key, seed)
          CHARACTER(C_CHAR), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_char
          libxs_hash_char = libxs_hash_c(C_LOC(key), SIZE(key), seed)
        END FUNCTION

        FUNCTION libxs_hash_i8(key, seed)
          INTEGER(C_INT8_T), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_i8
          libxs_hash_i8 = libxs_hash_c(C_LOC(key), SIZE(key), seed)
        END FUNCTION

        FUNCTION libxs_hash_i32(key, seed)
          INTEGER(C_INT), INTENT(IN), CONTIGUOUS,                       &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_i32
          libxs_hash_i32 = libxs_hash_c(                                &
     &      C_LOC(key), SIZE(key) * 4, seed)
        END FUNCTION

        FUNCTION libxs_hash_i64(key, seed)
          INTEGER(C_LONG_LONG), INTENT(IN), CONTIGUOUS,                 &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_i64
          libxs_hash_i64 = libxs_hash_c(                                &
     &      C_LOC(key), SIZE(key) * 8, seed)
        END FUNCTION

        !> Calculates if there is a difference between two arrays.
        FUNCTION libxs_diff_char(a, b)
          CHARACTER(C_CHAR), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: a(:), b(:)
          LOGICAL :: libxs_diff_char
          IF (SIZE(a, KIND=C_LONG_LONG)                                 &
     &      .EQ. SIZE(b, KIND=C_LONG_LONG)) THEN
            libxs_diff_char = (0 .NE. libxs_memcmp(                     &
     &        C_LOC(a), C_LOC(b),                                       &
     &        INT(SIZE(a), KIND=C_SIZE_T)))
          ELSE
            libxs_diff_char = .TRUE.
          END IF
        END FUNCTION

        FUNCTION libxs_diff_i8(a, b)
          INTEGER(C_INT8_T), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: a(:), b(:)
          LOGICAL :: libxs_diff_i8
          IF (SIZE(a, KIND=C_LONG_LONG)                                 &
     &      .EQ. SIZE(b, KIND=C_LONG_LONG)) THEN
            libxs_diff_i8 = (0 .NE. libxs_memcmp(                       &
     &        C_LOC(a), C_LOC(b),                                       &
     &        INT(SIZE(a), KIND=C_SIZE_T)))
          ELSE
            libxs_diff_i8 = .TRUE.
          END IF
        END FUNCTION

        FUNCTION libxs_diff_i32(a, b)
          INTEGER(C_INT), INTENT(IN), CONTIGUOUS,                       &
     &      TARGET :: a(:), b(:)
          LOGICAL :: libxs_diff_i32
          IF (SIZE(a, KIND=C_LONG_LONG)                                 &
     &      .EQ. SIZE(b, KIND=C_LONG_LONG)) THEN
            libxs_diff_i32 = (0 .NE. libxs_memcmp(                      &
     &        C_LOC(a), C_LOC(b),                                       &
     &        INT(SIZE(a), KIND=C_SIZE_T) * 4))
          ELSE
            libxs_diff_i32 = .TRUE.
          END IF
        END FUNCTION

        FUNCTION libxs_diff_i64(a, b)
          INTEGER(C_LONG_LONG), INTENT(IN), CONTIGUOUS,                 &
     &      TARGET :: a(:), b(:)
          LOGICAL :: libxs_diff_i64
          IF (SIZE(a, KIND=C_LONG_LONG)                                 &
     &      .EQ. SIZE(b, KIND=C_LONG_LONG)) THEN
            libxs_diff_i64 = (0 .NE. libxs_memcmp(                      &
     &        C_LOC(a), C_LOC(b),                                       &
     &        INT(SIZE(a), KIND=C_SIZE_T) * 8))
          ELSE
            libxs_diff_i64 = .TRUE.
          END IF
        END FUNCTION
      END MODULE
