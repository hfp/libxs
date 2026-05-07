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
     &    C_FUNPTR, C_NULL_FUNPTR,                                      &
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
        PUBLIC :: LIBXS_DATATYPE_C64, LIBXS_DATATYPE_C32
        PUBLIC :: LIBXS_TARGET_ARCH_UNKNOWN
        PUBLIC :: LIBXS_TARGET_ARCH_GENERIC
        PUBLIC :: LIBXS_X86_GENERIC, LIBXS_X86_SSE3
        PUBLIC :: LIBXS_X86_SSE42, LIBXS_X86_AVX
        PUBLIC :: LIBXS_X86_AVX2, LIBXS_X86_AVX10_256
        PUBLIC :: LIBXS_X86_AVX512, LIBXS_X86_AVX512_AMX
        PUBLIC :: LIBXS_X86_AVX512_INT8
        PUBLIC :: LIBXS_X86_AVX10_512, LIBXS_X86_ALLFEAT
        PUBLIC :: LIBXS_AARCH64, LIBXS_AARCH64_SVE128
        PUBLIC :: LIBXS_AARCH64_SVE256, LIBXS_AARCH64_SVE512
        PUBLIC :: LIBXS_AARCH64_ALLFEAT

        !> Public API: types and procedures.
        PUBLIC :: libxs_matdiff_t
        PUBLIC :: libxs_fprint_t, libxs_timer_info_t
        PUBLIC :: libxs_registry_info_t
        PUBLIC :: libxs_init, libxs_finalize
        PUBLIC :: libxs_timer_tick, libxs_timer_duration
        PUBLIC :: libxs_timer_info
        PUBLIC :: libxs_get_verbosity, libxs_set_verbosity
        PUBLIC :: libxs_malloc, libxs_free
        PUBLIC :: libxs_malloc_pool, libxs_malloc_xpool
        PUBLIC :: libxs_malloc_arg, libxs_free_pool
        PUBLIC :: libxs_hash, libxs_hash_string
        PUBLIC :: libxs_hash_iso3309, libxs_adler32
        PUBLIC :: libxs_diff
        PUBLIC :: libxs_matdiff, libxs_matdiff_reduce
        PUBLIC :: libxs_matdiff_clear, libxs_matdiff_epsilon
        PUBLIC :: libxs_matdiff_combine
        PUBLIC :: libxs_fprint, libxs_fprint_diff
        PUBLIC :: libxs_registry_create
        PUBLIC :: libxs_registry_destroy
        PUBLIC :: libxs_registry_lock
        PUBLIC :: libxs_registry_set
        PUBLIC :: libxs_registry_get
        PUBLIC :: libxs_registry_get_copy
        PUBLIC :: libxs_registry_has
        PUBLIC :: libxs_registry_remove
        PUBLIC :: libxs_registry_extract
        PUBLIC :: libxs_registry_info
        PUBLIC :: libxs_memcmp
        PUBLIC :: libxs_matcopy, libxs_matcopy_task
        PUBLIC :: libxs_otrans, libxs_otrans_task
        PUBLIC :: libxs_itrans, libxs_itrans_task
        PUBLIC :: libxs_typesize
        PUBLIC :: libxs_cpuid_t
        PUBLIC :: libxs_cpuid, libxs_cpuid_name
        PUBLIC :: libxs_cpuid_id, libxs_cpuid_vlen
        PUBLIC :: libxs_cpuid_amx_enable

        !> Public API: GEMM dispatch.
        PUBLIC :: libxs_gemm_shape_t, libxs_gemm_config_t
        PUBLIC :: LIBXS_GEMM_FLAGS_DEFAULT, LIBXS_GEMM_FLAG_NOLOCK
        PUBLIC :: libxs_gemm_ready, libxs_gemm_call
        PUBLIC :: libxs_gemm_dispatch, libxs_gemm_release
        PUBLIC :: libxs_gemm_batch, libxs_gemm_batch_task
        PUBLIC :: libxs_gemm_index, libxs_gemm_index_task
        PUBLIC :: libxs_syr2k_dispatch, libxs_syrk_dispatch
        PUBLIC :: libxs_syr2k, libxs_syrk

        !> Re-exported from ISO_C_BINDING for convenience.
        PUBLIC :: C_DOUBLE, C_FLOAT, C_INT, C_LONG_LONG
        PUBLIC :: C_CHAR, C_INT8_T, C_SIZE_T
        PUBLIC :: C_PTR, C_NULL_PTR
        PUBLIC :: C_FUNPTR, C_NULL_FUNPTR
        PUBLIC :: C_F_POINTER, C_ASSOCIATED, C_LOC, C_SIZEOF

        !> Integer kind used by timer interface.
        INTEGER(C_INT), PARAMETER :: LIBXS_TIMER_TICK_KIND = C_LONG_LONG

        !> Enumerates element/data types.
        !> The raw value encodes type-size in bits [7:4].
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXS_DATATYPE_F64     = IOR(0, ISHFT(8, 4)),                 &
     &    LIBXS_DATATYPE_F32     = IOR(1, ISHFT(4, 4)),                 &
     &    LIBXS_DATATYPE_C64     = IOR(2, ISHFT(16, 4)),                &
     &    LIBXS_DATATYPE_C32     = IOR(3, ISHFT(8, 4)),                 &
     &    LIBXS_DATATYPE_I64     = IOR(4, ISHFT(8, 4)),                 &
     &    LIBXS_DATATYPE_U64     = IOR(5, ISHFT(8, 4)),                 &
     &    LIBXS_DATATYPE_I32     = IOR(6, ISHFT(4, 4)),                 &
     &    LIBXS_DATATYPE_U32     = IOR(7, ISHFT(4, 4)),                 &
     &    LIBXS_DATATYPE_I16     = IOR(8, ISHFT(2, 4)),                 &
     &    LIBXS_DATATYPE_U16     = IOR(9, ISHFT(2, 4)),                 &
     &    LIBXS_DATATYPE_I8      = IOR(10, ISHFT(1, 4)),                &
     &    LIBXS_DATATYPE_U8      = IOR(11, ISHFT(1, 4)),                &
     &    LIBXS_DATATYPE_UNKNOWN = 12

        !> Enumerates GEMM batch synchronization flags (bitfield).
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXS_GEMM_FLAGS_DEFAULT = 0,                                 &
     &    LIBXS_GEMM_FLAG_NOLOCK   = 1

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
     &    LIBXS_X86_AVX10_256         = 1030,                           &
     &    LIBXS_X86_AVX512            = 1100,                           &
     &    LIBXS_X86_AVX512_AMX        = 1105,                           &
     &    LIBXS_X86_AVX512_INT8       = 1110,                           &
     &    LIBXS_X86_AVX10_512         = 1200,                           &
     &    LIBXS_X86_ALLFEAT           = 1999,                           &
     &    LIBXS_AARCH64               = 2001,                           &
     &    LIBXS_AARCH64_SVE128        = 2201,                           &
     &    LIBXS_AARCH64_SVE256        = 2301,                           &
     &    LIBXS_AARCH64_SVE512        = 2401,                           &
     &    LIBXS_AARCH64_ALLFEAT       = 2999

        !> Structure of differences with matrix norms according
        !> to http://www.netlib.org/lapack/lug/node75.html).
        TYPE, BIND(C) :: libxs_matdiff_t
          REAL(C_DOUBLE) :: norm1_abs, norm1_rel !! One-norm
          REAL(C_DOUBLE) :: normi_abs, normi_rel !! Infinity-norm
          REAL(C_DOUBLE) :: normf_rel            !! Froebenius-norm
          !> Maximum difference, L2-norm (absolute and relative),
          !> and R-squared.
          REAL(C_DOUBLE) :: linf_abs, linf_rel, l2_abs, l2_rel, rsq
          !> Statistics: sum/l1, min, max, avg, variance.
          REAL(C_DOUBLE) :: l1_ref, min_ref, max_ref, avg_ref, var_ref
          REAL(C_DOUBLE) :: l1_tst, min_tst, max_tst, avg_tst, var_tst
          !> Diagonal statistics: min and max of diagonal elements.
          REAL(C_DOUBLE) :: diag_min_ref, diag_max_ref
          REAL(C_DOUBLE) :: diag_min_tst, diag_max_tst
          !> Values(v_ref, v_tst) at location of largest linf_abs.
          REAL(C_DOUBLE) :: v_ref, v_tst
          !> Cumulative weight for online mean.
          REAL(C_DOUBLE) :: w
          !> Location (m, n) and reduction index (i, r).
          INTEGER(C_INT) :: m, n, i, r
        END TYPE

        !> Registry status information.
        TYPE, BIND(C) :: libxs_registry_info_t
          INTEGER(C_SIZE_T) :: capacity, size, nbytes
        END TYPE

        !> CPU identification result.
        TYPE, BIND(C) :: libxs_cpuid_t
          CHARACTER(C_CHAR) :: model(256)
          INTEGER(C_INT) :: constant_tsc
        END TYPE

        !> Maximum derivative order for fingerprints.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXS_FPRINT_MAXORDER = 8
        PUBLIC :: LIBXS_FPRINT_MAXORDER

        !> Foeppl polynomial fingerprint.
        TYPE, BIND(C) :: libxs_fprint_t
          REAL(C_DOUBLE) :: l2(LIBXS_FPRINT_MAXORDER + 1)
          REAL(C_DOUBLE) :: l1(LIBXS_FPRINT_MAXORDER + 1)
          REAL(C_DOUBLE) :: linf(LIBXS_FPRINT_MAXORDER + 1)
          INTEGER(C_INT) :: order, n
        END TYPE

        !> Timer properties.
        TYPE, BIND(C) :: libxs_timer_info_t
          INTEGER(C_INT) :: tsc
        END TYPE

        !> GEMM shape: problem geometry, transpose flags,
        !> and scalar coefficients. Alpha/beta stored as
        !> double (float promoted without loss). Also serves
        !> as registry key when caching configurations.
        TYPE, BIND(C) :: libxs_gemm_shape_t
          INTEGER(C_INT) :: datatype   = 0
          INTEGER(C_INT) :: transa     = 0
          INTEGER(C_INT) :: transb     = 0
          INTEGER(C_INT) :: m = 0, n = 0, k = 0
          INTEGER(C_INT) :: lda = 0, ldb = 0, ldc = 0
          REAL(C_DOUBLE)  :: alpha = 0, beta = 0
        END TYPE

        !> GEMM kernel configuration.
        !> All function-pointer fields are C_FUNPTR (default
        !> C_NULL_FUNPTR). Populate from Fortran with C_FUNLOC
        !> or pass to C code that calls libxs_gemm_dispatch.
        !> The shape member is populated by libxs_gemm_dispatch.
        TYPE, BIND(C) :: libxs_gemm_config_t
          TYPE(C_FUNPTR) :: dgemm_blas = C_NULL_FUNPTR
          TYPE(C_FUNPTR) :: sgemm_blas = C_NULL_FUNPTR
          TYPE(C_FUNPTR) :: dgemm_jit  = C_NULL_FUNPTR
          TYPE(C_FUNPTR) :: sgemm_jit  = C_NULL_FUNPTR
          TYPE(C_FUNPTR) :: xgemm      = C_NULL_FUNPTR
          TYPE(C_PTR)    :: jitter     = C_NULL_PTR
          INTEGER(C_INT) :: flags      = LIBXS_GEMM_FLAGS_DEFAULT
          TYPE(libxs_gemm_shape_t) :: shape
        END TYPE

        INTERFACE
          !> Returns the detected ISA level for the current
          !> platform. Pass C_NULL_PTR instead of info to skip
          !> filling the CPU properties structure.
          PURE FUNCTION libxs_cpuid(info) BIND(C)
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: info
            INTEGER(C_INT) :: libxs_cpuid
          END FUNCTION

          !> Returns a human-readable name for the given ISA
          !> level id. The returned pointer is to static storage.
          PURE FUNCTION libxs_cpuid_name(id) BIND(C)
            IMPORT :: C_PTR, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: id
            TYPE(C_PTR) :: libxs_cpuid_name
          END FUNCTION

          !> Translates a CPU architecture name (e.g., "avx2")
          !> to the corresponding LIBXS id constant.
          PURE FUNCTION libxs_cpuid_id(name) BIND(C)
            IMPORT :: C_CHAR, C_INT
            CHARACTER(C_CHAR), INTENT(IN) :: name(*)
            INTEGER(C_INT) :: libxs_cpuid_id
          END FUNCTION

          !> Returns the SIMD vector length (VLEN) in bytes
          !> for the given ISA level; zero if scalar.
          PURE FUNCTION libxs_cpuid_vlen(id) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: id
            INTEGER(C_INT) :: libxs_cpuid_vlen
          END FUNCTION

          !> Request AMX tile state from the OS.
          !> Returns 0 on success, -1 on failure.
          FUNCTION libxs_cpuid_amx_enable() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxs_cpuid_amx_enable
          END FUNCTION

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
          PURE FUNCTION libxs_timer_duration(tick0, tick1) BIND(C)
            IMPORT :: LIBXS_TIMER_TICK_KIND, C_DOUBLE
            INTEGER(LIBXS_TIMER_TICK_KIND), INTENT(IN), VALUE :: tick0
            INTEGER(LIBXS_TIMER_TICK_KIND), INTENT(IN), VALUE :: tick1
            REAL(C_DOUBLE) :: libxs_timer_duration
          END FUNCTION

          !> Internal binding (use libxs_malloc instead).
          FUNCTION libxs_malloc_c(pool, nbytes, alignment)              &
     &    BIND(C, NAME="libxs_malloc")
            IMPORT :: C_SIZE_T, C_INT, C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: pool
            INTEGER(C_SIZE_T), INTENT(IN), VALUE :: nbytes
            INTEGER(C_INT), INTENT(IN), VALUE :: alignment
            TYPE(C_PTR) :: libxs_malloc_c
          END FUNCTION

          !> Free memory allocated by libxs_malloc.
          SUBROUTINE libxs_free(ptr) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: ptr
          END SUBROUTINE

          !> Create a memory pool (returns opaque handle).
          FUNCTION libxs_malloc_pool(malloc_fn, free_fn)                &
     &    BIND(C)
            IMPORT :: C_PTR, C_FUNPTR
            TYPE(C_FUNPTR), INTENT(IN), VALUE :: malloc_fn
            TYPE(C_FUNPTR), INTENT(IN), VALUE :: free_fn
            TYPE(C_PTR) :: libxs_malloc_pool
          END FUNCTION

          !> Destroy pool and free all associated memory.
          SUBROUTINE libxs_free_pool(pool) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: pool
          END SUBROUTINE

          !> Create a pool with extended allocators (per-thread extra).
          FUNCTION libxs_malloc_xpool(malloc_fn,                        &
     &    free_fn, max_nthreads) BIND(C)
            IMPORT :: C_PTR, C_FUNPTR, C_INT
            TYPE(C_FUNPTR), INTENT(IN), VALUE :: malloc_fn
            TYPE(C_FUNPTR), INTENT(IN), VALUE :: free_fn
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &        max_nthreads
            TYPE(C_PTR) :: libxs_malloc_xpool
          END FUNCTION

          !> Set the per-thread extra argument for an extended pool.
          SUBROUTINE libxs_malloc_arg(pool, extra) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: pool
            TYPE(C_PTR), INTENT(IN), VALUE :: extra
          END SUBROUTINE

          !> Calculate a 64-bit hash for a character string.
          PURE FUNCTION libxs_hash_string(string) BIND(C)
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

          !> CRC-32 (ISO 3309 polynomial) for the given data.
          PURE FUNCTION libxs_hash_iso3309(data, size, seed)            &
     &    BIND(C)
            IMPORT :: C_INT, C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: data
            INTEGER(C_INT), INTENT(IN), VALUE :: size
            INTEGER(C_INT), INTENT(IN), VALUE :: seed
            INTEGER(C_INT) :: libxs_hash_iso3309
          END FUNCTION

          !> Adler-32 checksum for the given data.
          PURE FUNCTION libxs_adler32(data, size, seed)                 &
     &    BIND(C)
            IMPORT :: C_INT, C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: data
            INTEGER(C_INT), INTENT(IN), VALUE :: size
            INTEGER(C_INT), INTENT(IN), VALUE :: seed
            INTEGER(C_INT) :: libxs_adler32
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
            IMPORT :: libxs_matdiff_t
            TYPE(libxs_matdiff_t), INTENT(INOUT) :: output
            TYPE(libxs_matdiff_t), INTENT(IN)    :: input
          END SUBROUTINE

          !> Clears the given info-structure, e.g., for the initial
          !> reduction-value (libxs_matdiff_reduce).
          PURE SUBROUTINE libxs_matdiff_clear(info) BIND(C)
            IMPORT :: libxs_matdiff_t
            TYPE(libxs_matdiff_t), INTENT(OUT) :: info
          END SUBROUTINE

          !> Combine absolute and relative norms into a single
          !> value which can be used to check against a margin.
          PURE FUNCTION libxs_matdiff_epsilon(input) BIND(C)
            IMPORT :: libxs_matdiff_t, C_DOUBLE
            TYPE(libxs_matdiff_t), INTENT(IN) :: input
            REAL(C_DOUBLE) :: libxs_matdiff_epsilon
          END FUNCTION
          !> Create a registry (key-value store).
          !> Returns C_NULL_PTR in case of an error.
          FUNCTION libxs_registry_create()                              &
     &    BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR) :: libxs_registry_create
          END FUNCTION
          !> Destroy registry and release all entries.
          SUBROUTINE libxs_registry_destroy(registry)                   &
     &    BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
          END SUBROUTINE
          !> Return internal lock of registry.
          FUNCTION libxs_registry_lock(registry)                        &
     &    BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR) :: libxs_registry_lock
          END FUNCTION
          !> Internal C bindings for registry (lock argument).
          FUNCTION internal_registry_set_c(registry,                    &
     &    key, key_size, value_init, value_size,                        &
     &    lock) BIND(C, NAME="libxs_registry_set")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: value_init
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      value_size
            TYPE(C_PTR), INTENT(IN), VALUE :: lock
            TYPE(C_PTR) :: internal_registry_set_c
          END FUNCTION
          FUNCTION internal_registry_get_c(registry,                    &
     &    key, key_size, lock)                                          &
     &    BIND(C, NAME="libxs_registry_get")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: lock
            TYPE(C_PTR) :: internal_registry_get_c
          END FUNCTION
          FUNCTION internal_registry_get_copy_c(registry,               &
     &    key, key_size, value_out, value_size,                         &
     &    lock) BIND(C, NAME="libxs_registry_get_copy")
            IMPORT :: C_PTR, C_SIZE_T, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: value_out
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      value_size
            TYPE(C_PTR), INTENT(IN), VALUE :: lock
            INTEGER(C_INT) :: internal_registry_get_copy_c
          END FUNCTION
          FUNCTION internal_registry_has_c(registry,                    &
     &    key, key_size, lock)                                          &
     &    BIND(C, NAME="libxs_registry_has")
            IMPORT :: C_PTR, C_SIZE_T, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: lock
            INTEGER(C_INT) :: internal_registry_has_c
          END FUNCTION
          SUBROUTINE internal_registry_remove_c(registry,               &
     &    key, key_size, lock)                                          &
     &    BIND(C, NAME="libxs_registry_remove")
            IMPORT :: C_PTR, C_SIZE_T
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: lock
          END SUBROUTINE
          FUNCTION internal_registry_extract_c(registry,                &
     &    key, key_size, value_out, value_size,                         &
     &    lock) BIND(C, NAME="libxs_registry_extract")
            IMPORT :: C_PTR, C_SIZE_T, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      key_size
            TYPE(C_PTR), INTENT(IN), VALUE :: value_out
            INTEGER(C_SIZE_T), INTENT(IN), VALUE ::                     &
     &      value_size
            TYPE(C_PTR), INTENT(IN), VALUE :: lock
            INTEGER(C_INT) :: internal_registry_extract_c
          END FUNCTION
          !> Get registry information.
          !> Returns 0 on success.
          FUNCTION libxs_registry_info(registry,                        &
     &    info) BIND(C)
            IMPORT :: C_PTR, C_INT, libxs_registry_info_t
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            TYPE(libxs_registry_info_t), INTENT(OUT) :: info
            INTEGER(C_INT) :: libxs_registry_info
          END FUNCTION

          !> Copy a matrix (internal, use libxs_matcopy wrapper).
          PURE SUBROUTINE internal_matcopy(outm, inm,                   &
     &    typesize, m, n, ldi, ldo)                                     &
     &    BIND(C, NAME="libxs_matcopy")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
            INTEGER(C_INT), INTENT(IN), VALUE :: typesize
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &      m, n, ldi, ldo
          END SUBROUTINE

          !> Copy a matrix: task variant (internal).
          PURE SUBROUTINE internal_matcopy_task(outm, inm,              &
     &    typesize, m, n, ldi, ldo, tid, ntasks)                        &
     &    BIND(C, NAME="libxs_matcopy_task")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
            INTEGER(C_INT), INTENT(IN), VALUE :: typesize
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &      m, n, ldi, ldo, tid, ntasks
          END SUBROUTINE

          !> Out-of-place transpose (internal).
          PURE SUBROUTINE internal_otrans(outm, inm,                    &
     &    typesize, m, n, ldi, ldo)                                     &
     &    BIND(C, NAME="libxs_otrans")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
            INTEGER(C_INT), INTENT(IN), VALUE :: typesize
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &      m, n, ldi, ldo
          END SUBROUTINE

          !> Out-of-place transpose: task variant (internal).
          PURE SUBROUTINE internal_otrans_task(outm, inm,               &
     &    typesize, m, n, ldi, ldo, tid, ntasks)                        &
     &    BIND(C, NAME="libxs_otrans_task")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
            INTEGER(C_INT), INTENT(IN), VALUE :: typesize
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &      m, n, ldi, ldo, tid, ntasks
          END SUBROUTINE

          !> Pointer-array GEMM batch.
          !> Shape, alpha/beta, datatype from config%shape.
          SUBROUTINE libxs_gemm_batch(                                  &
     &    a_array, b_array, c_array,                                    &
     &    batchsize, config) BIND(C)
            IMPORT :: C_PTR, C_INT,                                     &
     &        libxs_gemm_config_t
            INTEGER(C_INT), INTENT(IN), VALUE :: batchsize
            TYPE(C_PTR), INTENT(IN) :: a_array(*), b_array(*)
            TYPE(C_PTR) :: c_array(*)
            TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          END SUBROUTINE

          !> Pointer-array GEMM batch: per-thread variant.
          SUBROUTINE libxs_gemm_batch_task(                             &
     &    a_array, b_array, c_array,                                    &
     &    batchsize, config, tid, ntasks) BIND(C)
            IMPORT :: C_PTR, C_INT,                                     &
     &        libxs_gemm_config_t
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &        batchsize, tid, ntasks
            TYPE(C_PTR), INTENT(IN) :: a_array(*), b_array(*)
            TYPE(C_PTR) :: c_array(*)
            TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          END SUBROUTINE

          !> Index-array GEMM batch: element-offsets into
          !> contiguous A, B, C buffers.
          !> index_stride is the Byte-stride used to walk
          !> stride_a, stride_b, stride_c (e.g. 4 for
          !> packed INTEGER(C_INT) arrays).
          !> index_stride=0: constant-stride mode.
          !> index_base selects the indexing convention:
          !> 0 for zero-based (C), 1 for one-based (Fortran).
          !> Shape, alpha/beta, datatype from config%shape.
          SUBROUTINE libxs_gemm_index(                                  &
     &    a, stride_a, b, stride_b, c, stride_c,                        &
     &    index_stride, index_base,                                     &
     &    batchsize, config) BIND(C)
            IMPORT :: C_PTR, C_INT,                                     &
     &        libxs_gemm_config_t
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &        index_stride, index_base, batchsize
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b
            TYPE(C_PTR), VALUE :: c
            TYPE(C_PTR), INTENT(IN), VALUE ::                           &
     &        stride_a, stride_b, stride_c
            TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          END SUBROUTINE

          !> Index-array GEMM batch: per-thread variant.
          SUBROUTINE libxs_gemm_index_task(                             &
     &    a, stride_a, b, stride_b, c, stride_c,                        &
     &    index_stride, index_base,                                     &
     &    batchsize, config, tid, ntasks) BIND(C)
            IMPORT :: C_PTR, C_INT,                                     &
     &        libxs_gemm_config_t
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &        index_stride, index_base,                                 &
     &        batchsize, tid, ntasks
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b
            TYPE(C_PTR), VALUE :: c
            TYPE(C_PTR), INTENT(IN), VALUE ::                           &
     &        stride_a, stride_b, stride_c
            TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          END SUBROUTINE

          !> Internal C binding for libxs_syr2k_dispatch.
          FUNCTION internal_syr2k_dispatch(config,                      &
     &    datatype, n, k, lda, ldb, ldc,                                &
     &    scratch_size, registry)                                       &
     &    BIND(C, NAME="libxs_syr2k_dispatch")
            IMPORT :: libxs_gemm_config_t,                              &
     &        C_PTR, C_INT, C_SIZE_T
            TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &        datatype, n, k, lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: scratch_size
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            INTEGER(C_INT) :: internal_syr2k_dispatch
          END FUNCTION

          !> Internal C binding for libxs_syrk_dispatch.
          FUNCTION internal_syrk_dispatch(config,                       &
     &    datatype, n, k, lda, ldc,                                     &
     &    scratch_size, registry)                                       &
     &    BIND(C, NAME="libxs_syrk_dispatch")
            IMPORT :: libxs_gemm_config_t,                              &
     &        C_PTR, C_INT, C_SIZE_T
            TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &        datatype, n, k, lda, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: scratch_size
            TYPE(C_PTR), INTENT(IN), VALUE :: registry
            INTEGER(C_INT) :: internal_syrk_dispatch
          END FUNCTION

          !> Internal C binding for libxs_syr2k.
          FUNCTION internal_syr2k(config, uplo,                         &
     &    alpha, beta, a, b, c, scratch)                                &
     &    BIND(C, NAME="libxs_syr2k")
            IMPORT :: libxs_gemm_config_t,                              &
     &        C_PTR, C_INT, C_DOUBLE, C_CHAR
            TYPE(libxs_gemm_config_t), INTENT(IN) :: config
            CHARACTER(C_CHAR), INTENT(IN), VALUE :: uplo
            REAL(C_DOUBLE), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b
            TYPE(C_PTR), VALUE :: c
            TYPE(C_PTR), INTENT(IN), VALUE :: scratch
            INTEGER(C_INT) :: internal_syr2k
          END FUNCTION

          !> Internal C binding for libxs_syrk.
          FUNCTION internal_syrk(config, uplo,                          &
     &    alpha, beta, a, c, scratch)                                   &
     &    BIND(C, NAME="libxs_syrk")
            IMPORT :: libxs_gemm_config_t,                              &
     &        C_PTR, C_INT, C_DOUBLE, C_CHAR
            TYPE(libxs_gemm_config_t), INTENT(IN) :: config
            CHARACTER(C_CHAR), INTENT(IN), VALUE :: uplo
            REAL(C_DOUBLE), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a
            TYPE(C_PTR), VALUE :: c
            TYPE(C_PTR), INTENT(IN), VALUE :: scratch
            INTEGER(C_INT) :: internal_syrk
          END FUNCTION

          !> Combine two single-matrix diffs into a meta-diff.
          FUNCTION libxs_matdiff_combine(output, input)                 &
     &    BIND(C)
            IMPORT :: libxs_matdiff_t, C_INT
            TYPE(libxs_matdiff_t), INTENT(INOUT) :: output
            TYPE(libxs_matdiff_t), INTENT(IN)    :: input
            INTEGER(C_INT) :: libxs_matdiff_combine
          END FUNCTION

          !> Query the verbosity level.
          FUNCTION libxs_get_verbosity() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxs_get_verbosity
          END FUNCTION

          !> Set the verbosity level.
          SUBROUTINE libxs_set_verbosity(level) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: level
          END SUBROUTINE

          !> Query timer properties.
          FUNCTION libxs_timer_info(info) BIND(C)
            IMPORT :: libxs_timer_info_t, C_INT
            TYPE(libxs_timer_info_t), INTENT(OUT) :: info
            INTEGER(C_INT) :: libxs_timer_info
          END FUNCTION

          !> Fingerprint (internal, use libxs_fprint wrapper).
          FUNCTION internal_fprint(info,                                &
     &    datatype, data, ndims, shape, stride,                         &
     &    order, axis) BIND(C, NAME="libxs_fprint")
            IMPORT :: libxs_fprint_t, C_PTR, C_INT, C_SIZE_T
            TYPE(libxs_fprint_t), INTENT(OUT)   :: info
            INTEGER(C_INT), INTENT(IN), VALUE   :: datatype
            TYPE(C_PTR), INTENT(IN), VALUE      :: data
            INTEGER(C_INT), INTENT(IN), VALUE   :: ndims
            INTEGER(C_SIZE_T), INTENT(IN)       :: shape(*)
            INTEGER(C_SIZE_T), INTENT(IN)       :: stride(*)
            INTEGER(C_INT), INTENT(IN), VALUE   :: order
            INTEGER(C_INT), INTENT(IN), VALUE   :: axis
            INTEGER(C_INT) :: internal_fprint
          END FUNCTION

          !> Fingerprint distance (internal).
          FUNCTION internal_fprint_diff(a, b, weights)                  &
     &    BIND(C, NAME="libxs_fprint_diff")
            IMPORT :: libxs_fprint_t, C_PTR, C_DOUBLE
            TYPE(libxs_fprint_t), INTENT(IN)  :: a, b
            TYPE(C_PTR), INTENT(IN), VALUE    :: weights
            REAL(C_DOUBLE) :: internal_fprint_diff
          END FUNCTION

          !> In-place transpose (internal).
          SUBROUTINE internal_itrans(inout, typesize,                   &
     &    m, n, ldi, ldo, scratch)                                      &
     &    BIND(C, NAME="libxs_itrans")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: inout, scratch
            INTEGER(C_INT), INTENT(IN), VALUE :: typesize
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &      m, n, ldi, ldo
          END SUBROUTINE

          !> In-place transpose: task variant (internal).
          SUBROUTINE internal_itrans_task(inout, typesize,              &
     &    m, n, ldi, ldo, scratch, tid, ntasks)                         &
     &    BIND(C, NAME="libxs_itrans_task")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: inout, scratch
            INTEGER(C_INT), INTENT(IN), VALUE :: typesize
            INTEGER(C_INT), INTENT(IN), VALUE ::                        &
     &      m, n, ldi, ldo, tid, ntasks
          END SUBROUTINE

        END INTERFACE

        !> Allocate memory (flags=0: automatic).
        INTERFACE libxs_malloc
          MODULE PROCEDURE libxs_malloc_bytes
        END INTERFACE

        !> Calculate a hash value for a given array and seed.
        INTERFACE libxs_hash
          MODULE PROCEDURE libxs_hash_xchar, libxs_hash_xi8
          MODULE PROCEDURE libxs_hash_xi32, libxs_hash_xi64
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

        !> Copy a matrix (or zero if source is NULL).
        !> ldo defaults to m (tightly packed output).
        PURE SUBROUTINE libxs_matcopy(outm, inm, typesize,              &
     &  m, n, ldi, ldo)
          TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
          INTEGER(C_INT), INTENT(IN), VALUE :: typesize
          INTEGER(C_INT), INTENT(IN), VALUE :: m, n, ldi
          INTEGER(C_INT), INTENT(IN), VALUE, OPTIONAL :: ldo
          IF (PRESENT(ldo)) THEN
            CALL internal_matcopy(outm, inm, typesize,                  &
     &        m, n, ldi, ldo)
          ELSE
            CALL internal_matcopy(outm, inm, typesize,                  &
     &        m, n, ldi, m)
          END IF
        END SUBROUTINE

        !> Copy a matrix: task variant for external threading.
        !> ldo defaults to m (tightly packed output).
        PURE SUBROUTINE libxs_matcopy_task(outm, inm,                   &
     &  typesize, m, n, ldi, ldo, tid, ntasks)
          TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
          INTEGER(C_INT), INTENT(IN), VALUE :: typesize
          INTEGER(C_INT), INTENT(IN), VALUE ::                          &
     &      m, n, ldi, tid, ntasks
          INTEGER(C_INT), INTENT(IN), VALUE, OPTIONAL :: ldo
          IF (PRESENT(ldo)) THEN
            CALL internal_matcopy_task(outm, inm, typesize,             &
     &        m, n, ldi, ldo, tid, ntasks)
          ELSE
            CALL internal_matcopy_task(outm, inm, typesize,             &
     &        m, n, ldi, m, tid, ntasks)
          END IF
        END SUBROUTINE

        !> Out-of-place matrix transpose.
        !> ldo defaults to n (tightly packed transposed output).
        PURE SUBROUTINE libxs_otrans(outm, inm, typesize,               &
     &  m, n, ldi, ldo)
          TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
          INTEGER(C_INT), INTENT(IN), VALUE :: typesize
          INTEGER(C_INT), INTENT(IN), VALUE :: m, n, ldi
          INTEGER(C_INT), INTENT(IN), VALUE, OPTIONAL :: ldo
          IF (PRESENT(ldo)) THEN
            CALL internal_otrans(outm, inm, typesize,                   &
     &        m, n, ldi, ldo)
          ELSE
            CALL internal_otrans(outm, inm, typesize,                   &
     &        m, n, ldi, n)
          END IF
        END SUBROUTINE

        !> Out-of-place transpose: task variant.
        !> ldo defaults to n (tightly packed transposed output).
        PURE SUBROUTINE libxs_otrans_task(outm, inm,                    &
     &  typesize, m, n, ldi, ldo, tid, ntasks)
          TYPE(C_PTR), INTENT(IN), VALUE :: outm, inm
          INTEGER(C_INT), INTENT(IN), VALUE :: typesize
          INTEGER(C_INT), INTENT(IN), VALUE ::                          &
     &      m, n, ldi, tid, ntasks
          INTEGER(C_INT), INTENT(IN), VALUE, OPTIONAL :: ldo
          IF (PRESENT(ldo)) THEN
            CALL internal_otrans_task(outm, inm, typesize,              &
     &        m, n, ldi, ldo, tid, ntasks)
          ELSE
            CALL internal_otrans_task(outm, inm, typesize,              &
     &        m, n, ldi, n, tid, ntasks)
          END IF
        END SUBROUTINE

        !> Allocate from a pool reaching steady-state.
        !> alignment=0 (LIBXS_MALLOC_AUTO, default): automatic.
        !> alignment=1 (LIBXS_MALLOC_NATIVE): preserve allocator's pointer.
        FUNCTION libxs_malloc_bytes(pool, nbytes, alignment)
          TYPE(C_PTR), INTENT(IN) :: pool
          INTEGER(C_SIZE_T), INTENT(IN) :: nbytes
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: alignment
          TYPE(C_PTR) :: libxs_malloc_bytes
          IF (PRESENT(alignment)) THEN
            libxs_malloc_bytes =                                        &
     &        libxs_malloc_c(pool, nbytes, alignment)
          ELSE
            libxs_malloc_bytes =                                        &
     &        libxs_malloc_c(pool, nbytes, 0)
          END IF
        END FUNCTION

        !> Matdiff wrapper with OPTIONAL arguments.
        SUBROUTINE libxs_matdiff(info, datatype, m, n,                  &
     &  ref, tst, ldref, ldtst)
          INTEGER(C_INT), INTENT(IN) :: datatype
          INTEGER(C_INT), INTENT(IN) :: m
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: n, ldref, ldtst
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: ref, tst
          TYPE(libxs_matdiff_t), INTENT(OUT) :: info
          INTEGER(C_INT) :: nn, rc
          INTEGER(C_INT), TARGET :: lr, lt
          TYPE(C_PTR) :: rr, tt, plr, plt
          INTERFACE
            FUNCTION internal_matdiff(info,                             &
     &      datatype, m, n, ref, tst, ldref, ldtst)                     &
     &      RESULT(res) BIND(C, NAME="libxs_matdiff")
              IMPORT :: libxs_matdiff_t, C_PTR, C_INT
              INTEGER(C_INT), INTENT(IN), VALUE :: datatype
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n
              TYPE(C_PTR), INTENT(IN), VALUE :: ref, tst
              TYPE(C_PTR), INTENT(IN), VALUE :: ldref, ldtst
              TYPE(libxs_matdiff_t), INTENT(OUT) :: info
              INTEGER(C_INT) :: res
            END FUNCTION
          END INTERFACE
          IF (PRESENT(n)) THEN; nn = n; ELSE; nn = m; END IF
          IF (PRESENT(ref)) THEN; rr = ref
          ELSE; rr = C_NULL_PTR; END IF
          IF (PRESENT(tst)) THEN; tt = tst
          ELSE; tt = C_NULL_PTR; END IF
          IF (PRESENT(ldref)) THEN
            lr = ldref; plr = C_LOC(lr)
          ELSE; plr = C_NULL_PTR; END IF
          IF (PRESENT(ldtst)) THEN
            lt = ldtst; plt = C_LOC(lt)
          ELSE; plt = C_NULL_PTR; END IF
          rc = internal_matdiff(info, datatype, m, nn,                  &
     &      rr, tt, plr, plt)
        END SUBROUTINE

        !> Foeppl polynomial fingerprint.
        !> order defaults to LIBXS_FPRINT_MAXORDER.
        !> axis defaults to -1 (hierarchical).
        FUNCTION libxs_fprint(info, datatype, data,                     &
     &  ndims, shape, stride, order, axis)
          TYPE(libxs_fprint_t), INTENT(OUT) :: info
          INTEGER(C_INT), INTENT(IN) :: datatype
          TYPE(C_PTR), INTENT(IN) :: data
          INTEGER(C_INT), INTENT(IN) :: ndims
          INTEGER(C_SIZE_T), INTENT(IN) :: shape(ndims)
          INTEGER(C_SIZE_T), INTENT(IN) :: stride(ndims)
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: order, axis
          INTEGER(C_INT) :: libxs_fprint
          INTEGER(C_INT) :: oo, aa
          IF (PRESENT(order)) THEN
            oo = order
          ELSE; oo = LIBXS_FPRINT_MAXORDER; END IF
          IF (PRESENT(axis)) THEN
            aa = axis
          ELSE; aa = -1; END IF
          libxs_fprint = internal_fprint(info, datatype, data,          &
     &      ndims, shape, stride, oo, aa)
        END FUNCTION

        !> Weighted Sobolev distance between two fingerprints.
        !> If weights is absent, default weights are used.
        FUNCTION libxs_fprint_diff(a, b, weights)
          TYPE(libxs_fprint_t), INTENT(IN) :: a, b
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET ::               &
     &      weights(*)
          REAL(C_DOUBLE) :: libxs_fprint_diff
          IF (PRESENT(weights)) THEN
            libxs_fprint_diff = internal_fprint_diff(                   &
     &        a, b, C_LOC(weights))
          ELSE
            libxs_fprint_diff = internal_fprint_diff(                   &
     &        a, b, C_NULL_PTR)
          END IF
        END FUNCTION

        !> In-place matrix transpose.
        !> ldo defaults to n. scratch defaults to NULL
        !> (auto-allocate internally).
        SUBROUTINE libxs_itrans(inout, typesize,                        &
     &  m, n, ldi, ldo, scratch)
          TYPE(C_PTR), INTENT(IN) :: inout
          INTEGER(C_INT), INTENT(IN) :: typesize
          INTEGER(C_INT), INTENT(IN) :: m, n, ldi
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: ldo
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: scratch
          INTEGER(C_INT) :: lo
          TYPE(C_PTR) :: ss
          IF (PRESENT(ldo)) THEN; lo = ldo
          ELSE; lo = n; END IF
          IF (PRESENT(scratch)) THEN; ss = scratch
          ELSE; ss = C_NULL_PTR; END IF
          CALL internal_itrans(inout, typesize,                         &
     &      m, n, ldi, lo, ss)
        END SUBROUTINE

        !> In-place transpose: task variant.
        !> ldo defaults to n. scratch defaults to NULL.
        SUBROUTINE libxs_itrans_task(inout, typesize,                   &
     &  m, n, ldi, ldo, scratch, tid, ntasks)
          TYPE(C_PTR), INTENT(IN) :: inout
          INTEGER(C_INT), INTENT(IN) :: typesize
          INTEGER(C_INT), INTENT(IN) ::                                 &
     &      m, n, ldi, tid, ntasks
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: ldo
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: scratch
          INTEGER(C_INT) :: lo
          TYPE(C_PTR) :: ss
          IF (PRESENT(ldo)) THEN; lo = ldo
          ELSE; lo = n; END IF
          IF (PRESENT(scratch)) THEN; ss = scratch
          ELSE; ss = C_NULL_PTR; END IF
          CALL internal_itrans_task(inout, typesize,                    &
     &      m, n, ldi, lo, ss, tid, ntasks)
        END SUBROUTINE

        !> Calculates a hash value for the given array and seed.
        PURE FUNCTION libxs_hash_xchar(key, seed)
          CHARACTER(C_CHAR), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_xchar
          libxs_hash_xchar = libxs_hash_c(C_LOC(key), SIZE(key), seed)
        END FUNCTION

        PURE FUNCTION libxs_hash_char(key)
          CHARACTER(C_CHAR), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: key(:)
          INTEGER(C_INT) :: libxs_hash_char
          libxs_hash_char = libxs_hash_c(C_LOC(key), SIZE(key), 0)
        END FUNCTION

        PURE FUNCTION libxs_hash_xi8(key, seed)
          INTEGER(C_INT8_T), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_xi8
          libxs_hash_xi8 = libxs_hash_c(C_LOC(key), SIZE(key), seed)
        END FUNCTION

        PURE FUNCTION libxs_hash_i8(key)
          INTEGER(C_INT8_T), INTENT(IN), CONTIGUOUS,                    &
     &      TARGET :: key(:)
          INTEGER(C_INT) :: libxs_hash_i8
          libxs_hash_i8 = libxs_hash_c(C_LOC(key), SIZE(key), 0)
        END FUNCTION

        PURE FUNCTION libxs_hash_xi32(key, seed)
          INTEGER(C_INT), INTENT(IN), CONTIGUOUS,                       &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_xi32
          libxs_hash_xi32 = libxs_hash_c(                               &
     &      C_LOC(key), SIZE(key) * 4, seed)
        END FUNCTION

        PURE FUNCTION libxs_hash_i32(key)
          INTEGER(C_INT), INTENT(IN), CONTIGUOUS,                       &
     &      TARGET :: key(:)
          INTEGER(C_INT) :: libxs_hash_i32
          libxs_hash_i32 = libxs_hash_c(                                &
     &      C_LOC(key), SIZE(key) * 4, 0)
        END FUNCTION

        PURE FUNCTION libxs_hash_xi64(key, seed)
          INTEGER(C_LONG_LONG), INTENT(IN), CONTIGUOUS,                 &
     &      TARGET :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxs_hash_xi64
          libxs_hash_xi64 = libxs_hash_c(                               &
     &      C_LOC(key), SIZE(key) * 8, seed)
        END FUNCTION

        PURE FUNCTION libxs_hash_i64(key)
          INTEGER(C_LONG_LONG), INTENT(IN), CONTIGUOUS,                 &
     &      TARGET :: key(:)
          INTEGER(C_INT) :: libxs_hash_i64
          libxs_hash_i64 = libxs_hash_c(                                &
     &      C_LOC(key), SIZE(key) * 8, 0)
        END FUNCTION

        !> Calculates if there is a difference between two arrays.
        PURE FUNCTION libxs_diff_char(a, b)
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

        PURE FUNCTION libxs_diff_i8(a, b)
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

        PURE FUNCTION libxs_diff_i32(a, b)
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

        PURE FUNCTION libxs_diff_i64(a, b)
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

        !> Register a key-value pair. Returns a pointer
        !> to the stored value, or C_NULL_PTR on failure.
        FUNCTION libxs_registry_set(registry,                           &
     &  key, key_size, value_init, value_size)
          TYPE(C_PTR), INTENT(IN) :: registry
          TYPE(C_PTR), INTENT(IN) :: key
          INTEGER(C_SIZE_T), INTENT(IN) :: key_size
          TYPE(C_PTR), INTENT(IN) :: value_init
          INTEGER(C_SIZE_T), INTENT(IN) :: value_size
          TYPE(C_PTR) :: libxs_registry_set
          libxs_registry_set = internal_registry_set_c(                 &
     &      registry, key, key_size,                                    &
     &      value_init, value_size, libxs_registry_lock(registry))
        END FUNCTION

        !> Query a value by key. Returns C_NULL_PTR
        !> if the key is not found.
        FUNCTION libxs_registry_get(registry, key, key_size)
          TYPE(C_PTR), INTENT(IN) :: registry
          TYPE(C_PTR), INTENT(IN) :: key
          INTEGER(C_SIZE_T), INTENT(IN) :: key_size
          TYPE(C_PTR) :: libxs_registry_get
          libxs_registry_get = internal_registry_get_c(                 &
     &      registry, key, key_size, libxs_registry_lock(registry))
        END FUNCTION

        !> Thread-safe query: copies up to value_size bytes
        !> into value_out under the lock. Returns non-zero
        !> if the key was found.
        FUNCTION libxs_registry_get_copy(registry,                      &
     &  key, key_size, value_out, value_size)
          TYPE(C_PTR), INTENT(IN) :: registry
          TYPE(C_PTR), INTENT(IN) :: key
          INTEGER(C_SIZE_T), INTENT(IN) :: key_size
          TYPE(C_PTR), INTENT(IN) :: value_out
          INTEGER(C_SIZE_T), INTENT(IN) :: value_size
          INTEGER(C_INT) :: libxs_registry_get_copy
          libxs_registry_get_copy = internal_registry_get_copy_c(       &
     &      registry, key, key_size,                                    &
     &      value_out, value_size, libxs_registry_lock(registry))
        END FUNCTION

        !> Check if a key exists (non-zero if found).
        FUNCTION libxs_registry_has(registry, key, key_size)
          TYPE(C_PTR), INTENT(IN) :: registry
          TYPE(C_PTR), INTENT(IN) :: key
          INTEGER(C_SIZE_T), INTENT(IN) :: key_size
          INTEGER(C_INT) :: libxs_registry_has
          libxs_registry_has = internal_registry_has_c(                 &
     &      registry, key, key_size, libxs_registry_lock(registry))
        END FUNCTION

        !> Remove a key-value pair from the registry.
        SUBROUTINE libxs_registry_remove(registry, key, key_size)
          TYPE(C_PTR), INTENT(IN) :: registry
          TYPE(C_PTR), INTENT(IN) :: key
          INTEGER(C_SIZE_T), INTENT(IN) :: key_size
          CALL internal_registry_remove_c(                              &
     &      registry, key, key_size, libxs_registry_lock(registry))
        END SUBROUTINE

        !> Atomically retrieve and remove a key-value pair.
        !> Copies up to value_size bytes into value_out,
        !> then removes the entry. Returns non-zero if found.
        FUNCTION libxs_registry_extract(registry,                       &
     &  key, key_size, value_out, value_size)
          TYPE(C_PTR), INTENT(IN) :: registry
          TYPE(C_PTR), INTENT(IN) :: key
          INTEGER(C_SIZE_T), INTENT(IN) :: key_size
          TYPE(C_PTR), INTENT(IN) :: value_out
          INTEGER(C_SIZE_T), INTENT(IN) :: value_size
          INTEGER(C_INT) :: libxs_registry_extract
          libxs_registry_extract = internal_registry_extract_c(         &
     &      registry, key, key_size,                                    &
     &      value_out, value_size, libxs_registry_lock(registry))
        END FUNCTION

        !> Check if a dispatched GEMM config holds a usable
        !> kernel. Returns nonzero if libxs_gemm_call would
        !> succeed, zero otherwise.
        PURE FUNCTION libxs_gemm_ready(config)
          TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          INTEGER(C_INT) :: libxs_gemm_ready
          INTERFACE
            PURE FUNCTION internal_gemm_ready_f(config)                 &
     &      RESULT(res) BIND(C, NAME="libxs_gemm_ready_f")
              IMPORT :: libxs_gemm_config_t, C_INT
              TYPE(libxs_gemm_config_t), INTENT(IN) :: config
              INTEGER(C_INT) :: res
            END FUNCTION
          END INTERFACE
          libxs_gemm_ready = internal_gemm_ready_f(config)
        END FUNCTION

        !> Call the GEMM kernel dispatched into config.
        !> Returns 0 on success, nonzero on failure.
        FUNCTION libxs_gemm_call(config, a, b, c)
          TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b
          TYPE(C_PTR), VALUE :: c
          INTEGER(C_INT) :: libxs_gemm_call
          INTERFACE
            FUNCTION internal_gemm_call_f(config, a, b, c)              &
     &      RESULT(res) BIND(C, NAME="libxs_gemm_call_f")
              IMPORT :: libxs_gemm_config_t, C_PTR, C_INT
              TYPE(libxs_gemm_config_t), INTENT(IN) :: config
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b
              TYPE(C_PTR), VALUE :: c
              INTEGER(C_INT) :: res
            END FUNCTION
          END INTERFACE
          libxs_gemm_call = internal_gemm_call_f(config, a, b, c)
        END FUNCTION

        !> Populate config with caller-provided kernel
        !> pointers and GEMM shape. Supports MKL JIT
        !> (dgemm_jit/sgemm_jit + jitter), LIBXSMM (xgemm),
        !> and BLAS (dgemm_blas/sgemm_blas).
        !> Returns nonzero if the config holds a usable
        !> kernel for libxs_gemm_call (JIT or XGEMM).
        !> BLAS-only configs return zero but are valid
        !> for the batch functions.
        !> If registry is present, dispatched configs are
        !> cached by shape: a hit copies the cached config,
        !> a miss dispatches and stores the result.
        FUNCTION libxs_gemm_dispatch(config,                            &
     &    datatype, transa, transb,                                     &
     &    m, n, k, lda, ldb, ldc, alpha, beta,                          &
     &    dgemm_jit, sgemm_jit, jitter,                                 &
     &    xgemm, dgemm_blas, sgemm_blas,                                &
     &    registry)
          TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: datatype
          CHARACTER(C_CHAR), INTENT(IN), OPTIONAL ::                    &
     &      transa, transb
          INTEGER(C_INT), INTENT(IN), OPTIONAL ::                       &
     &      m, n, k, lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(C_FUNPTR), INTENT(IN), OPTIONAL :: dgemm_jit
          TYPE(C_FUNPTR), INTENT(IN), OPTIONAL :: sgemm_jit
          TYPE(C_PTR),    INTENT(IN), OPTIONAL :: jitter
          TYPE(C_FUNPTR), INTENT(IN), OPTIONAL :: xgemm
          TYPE(C_FUNPTR), INTENT(IN), OPTIONAL :: dgemm_blas
          TYPE(C_FUNPTR), INTENT(IN), OPTIONAL :: sgemm_blas
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          INTEGER(C_INT) :: libxs_gemm_dispatch
          TYPE(libxs_gemm_shape_t), TARGET :: key
          TYPE(libxs_gemm_config_t), TARGET :: tmp
          TYPE(libxs_gemm_config_t), POINTER :: cached
          TYPE(C_PTR) :: ptr
          LOGICAL :: from_cache
          from_cache = .FALSE.
          !> Set shape from optional parameters.
          IF (PRESENT(datatype))                                        &
     &      config%shape%datatype = datatype
          IF (PRESENT(transa))                                          &
     &      config%shape%transa = ICHAR(transa)
          IF (PRESENT(transb))                                          &
     &      config%shape%transb = ICHAR(transb)
          IF (PRESENT(m)) config%shape%m = m
          IF (PRESENT(n)) config%shape%n = n
          IF (PRESENT(k)) config%shape%k = k
          IF (PRESENT(lda)) config%shape%lda = lda
          IF (PRESENT(ldb)) config%shape%ldb = ldb
          IF (PRESENT(ldc)) config%shape%ldc = ldc
          IF (PRESENT(alpha)) config%shape%alpha = alpha
          IF (PRESENT(beta)) config%shape%beta = beta
          !> Registry lookup (cache hit copies full config).
          IF (PRESENT(registry)) THEN
            key = config%shape
            ptr = libxs_registry_get(registry,                          &
     &        C_LOC(key),                                               &
     &        INT(C_SIZEOF(key), C_SIZE_T))
            IF (C_ASSOCIATED(ptr)) THEN
              CALL C_F_POINTER(ptr, cached)
              config = cached
              from_cache = .TRUE.
            END IF
          END IF
          !> Set kernel pointers (skipped on cache hit).
          IF (.NOT. from_cache) THEN
            IF (PRESENT(dgemm_jit))                                     &
     &        config%dgemm_jit = dgemm_jit
            IF (PRESENT(sgemm_jit))                                     &
     &        config%sgemm_jit = sgemm_jit
            IF (PRESENT(jitter))                                        &
     &        config%jitter = jitter
            IF (PRESENT(xgemm)) config%xgemm = xgemm
            IF (PRESENT(dgemm_blas))                                    &
     &        config%dgemm_blas = dgemm_blas
            IF (PRESENT(sgemm_blas))                                    &
     &        config%sgemm_blas = sgemm_blas
            !> Cache miss: store in registry.
            IF (PRESENT(registry)) THEN
              tmp = config
              ptr = libxs_registry_set(registry,                        &
     &          C_LOC(key),                                             &
     &          INT(C_SIZEOF(key), C_SIZE_T),                           &
     &          C_LOC(tmp),                                             &
     &          INT(C_SIZEOF(tmp), C_SIZE_T))
            END IF
          END IF
          libxs_gemm_dispatch = libxs_gemm_ready(config)
        END FUNCTION

        !> Release a GEMM config: clear all kernel pointers.
        !> If the config holds a JIT jitter handle, the
        !> caller must destroy it first (e.g., call
        !> mkl_jit_destroy before calling this routine).
        SUBROUTINE libxs_gemm_release(config)
          TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
          config%dgemm_jit  = C_NULL_FUNPTR
          config%sgemm_jit  = C_NULL_FUNPTR
          config%jitter     = C_NULL_PTR
          config%xgemm      = C_NULL_FUNPTR
          config%dgemm_blas = C_NULL_FUNPTR
          config%sgemm_blas = C_NULL_FUNPTR
        END SUBROUTINE

        !> Dispatch a GEMM config for SYR2K.
        !> scratch_size (optional): receives required scratch
        !> buffer size in bytes.
        !> registry (optional): cache dispatched configs.
        FUNCTION libxs_syr2k_dispatch(config,                           &
     &  datatype, n, k, lda, ldb, ldc, scratch_size, registry)
          TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k, lda, ldb, ldc
          INTEGER(C_SIZE_T), INTENT(OUT), OPTIONAL, TARGET ::           &
     &      scratch_size
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          INTEGER(C_INT) :: libxs_syr2k_dispatch
          TYPE(C_PTR) :: pss, preg
          IF (PRESENT(scratch_size)) THEN
            pss = C_LOC(scratch_size)
          ELSE; pss = C_NULL_PTR; END IF
          IF (PRESENT(registry)) THEN
            preg = registry
          ELSE; preg = C_NULL_PTR; END IF
          libxs_syr2k_dispatch = internal_syr2k_dispatch(               &
     &      config, datatype, n, k, lda, ldb, ldc, pss, preg)
        END FUNCTION

        !> Dispatch a GEMM config for SYRK.
        !> scratch_size (optional): receives required scratch
        !> buffer size in bytes.
        !> registry (optional): cache dispatched configs.
        FUNCTION libxs_syrk_dispatch(config,                            &
     &  datatype, n, k, lda, ldc, scratch_size, registry)
          TYPE(libxs_gemm_config_t), INTENT(INOUT) :: config
          INTEGER(C_INT), INTENT(IN) :: datatype, n, k, lda, ldc
          INTEGER(C_SIZE_T), INTENT(OUT), OPTIONAL, TARGET ::           &
     &      scratch_size
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: registry
          INTEGER(C_INT) :: libxs_syrk_dispatch
          TYPE(C_PTR) :: pss, preg
          IF (PRESENT(scratch_size)) THEN
            pss = C_LOC(scratch_size)
          ELSE; pss = C_NULL_PTR; END IF
          IF (PRESENT(registry)) THEN
            preg = registry
          ELSE; preg = C_NULL_PTR; END IF
          libxs_syrk_dispatch = internal_syrk_dispatch(                 &
     &      config, datatype, n, k, lda, ldc, pss, preg)
        END FUNCTION

        !> Symmetric rank-2k update.
        !> C := alpha*(A*B^T + B*A^T) + beta*C.
        !> scratch (optional): n*n workspace; if absent uses
        !> two-GEMM path (no scratch needed).
        FUNCTION libxs_syr2k(config, uplo, alpha, beta,                 &
     &  a, b, c, scratch)
          TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          CHARACTER(C_CHAR), INTENT(IN) :: uplo
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN) :: a, b
          TYPE(C_PTR) :: c
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: scratch
          INTEGER(C_INT) :: libxs_syr2k
          TYPE(C_PTR) :: ss
          IF (PRESENT(scratch)) THEN
            ss = scratch
          ELSE; ss = C_NULL_PTR; END IF
          libxs_syr2k = internal_syr2k(config, uplo,                    &
     &      alpha, beta, a, b, c, ss)
        END FUNCTION

        !> Symmetric rank-k update.
        !> C := alpha*A*A^T + beta*C.
        !> scratch (optional): n*n workspace; if absent uses
        !> single-BLAS path (no scratch needed).
        FUNCTION libxs_syrk(config, uplo, alpha, beta,                  &
     &  a, c, scratch)
          TYPE(libxs_gemm_config_t), INTENT(IN) :: config
          CHARACTER(C_CHAR), INTENT(IN) :: uplo
          REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          TYPE(C_PTR), INTENT(IN) :: a
          TYPE(C_PTR) :: c
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: scratch
          INTEGER(C_INT) :: libxs_syrk
          TYPE(C_PTR) :: ss
          IF (PRESENT(scratch)) THEN
            ss = scratch
          ELSE; ss = C_NULL_PTR; END IF
          libxs_syrk = internal_syrk(config, uplo,                      &
     &      alpha, beta, a, c, ss)
        END FUNCTION
      END MODULE
