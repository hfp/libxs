/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                          *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef LIBXS_MAIN_H
#define LIBXS_MAIN_H

#include <libxs.h>
/**
 * TF includes src/libxs_main.h and uses LIBXS's sync primitives
 * without including libxs_sync. However, libxs_sync.h shall be
 * an explicit include separate from including libxs.h.
 */
#include <libxs_sync.h>

/** Allow external definition to enable testing corner cases (exhausted registry space). */
#if !defined(LIBXS_CAPACITY_REGISTRY) /* must be POT */
# define LIBXS_CAPACITY_REGISTRY 131072
#endif
#if !defined(LIBXS_CAPACITY_CACHE) /* must be POT */
# define LIBXS_CAPACITY_CACHE 16
#endif

#if !defined(LIBXS_PAGE_MINSIZE)
# define LIBXS_PAGE_MINSIZE 4096 /* 4 KB */
#endif

#if !defined(LIBXS_BATCH_CHECK) && !defined(NDEBUG)
# define LIBXS_BATCH_CHECK
#endif

#if !defined(LIBXS_NTHREADS_MAX)
# if (0 != LIBXS_SYNC)
#   define LIBXS_NTHREADS_MAX 1024
# else
#   define LIBXS_NTHREADS_MAX 1
# endif
#endif
/* relies on LIBXS_NTHREADS_MAX */
#if !defined(LIBXS_NTHREADS_USE) && 0
# define LIBXS_NTHREADS_USE
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_MAX_NPOOLS)
# define LIBXS_MALLOC_SCRATCH_MAX_NPOOLS LIBXS_NTHREADS_MAX
#endif
#if !defined(LIBXS_MALLOC_SCRATCH_SCALE)
# define LIBXS_MALLOC_SCRATCH_SCALE 1.0
#endif
#if !defined(LIBXS_MALLOC_LIMIT)
# define LIBXS_MALLOC_LIMIT (2U << 20) /* 2 MB */
#endif
/* map memory also for non-executable buffers */
#if !defined(LIBXS_MALLOC_MMAP) && 0
# define LIBXS_MALLOC_MMAP
#endif
/* map memory for hooked allocation */
#if !defined(LIBXS_MALLOC_MMAP_HOOK) && 1
# define LIBXS_MALLOC_MMAP_HOOK
#endif
/* map memory for scratch buffers */
#if !defined(LIBXS_MALLOC_MMAP_SCRATCH) && 0
# define LIBXS_MALLOC_MMAP_SCRATCH
#endif
/* align if interceptor is disabled (moderated malloc) */
#if defined(LIBXS_MALLOC_MOD) && 0
# define LIBXS_MALLOC_MOD
#endif
#if !defined(LIBXS_MALLOC_HOOK_INTRINSIC) && 1
# if defined(LIBXS_PLATFORM_X86) && defined(LIBXS_INTRINSICS_INCLUDE) && \
    !defined(LIBXS_INTRINSICS_DEBUG) && !defined(LIBXS_MALLOC_MMAP)
#   define LIBXS_MALLOC_HOOK_INTRINSIC
# endif
#endif
#if !defined(LIBXS_MALLOC_HOOK_REALLOC) && 1
# if !defined(LIBXS_MALLOC_HOOK_INTRINSIC)
#   define LIBXS_MALLOC_HOOK_REALLOC
# endif
#endif
#if !defined(LIBXS_MALLOC_HOOK_CALLOC) && 1
# define LIBXS_MALLOC_HOOK_CALLOC
#endif
#if !defined(LIBXS_MALLOC_INTERNAL_CALLER_ID)
# define LIBXS_MALLOC_INTERNAL_CALLER_ID ((uintptr_t)LIBXS_UNLIMITED)
#endif
#if !defined(LIBXS_MALLOC_INTERNAL_CALLER)
# define LIBXS_MALLOC_INTERNAL_CALLER ((const void*)(LIBXS_MALLOC_INTERNAL_CALLER_ID))
#endif

#if !defined(LIBXS_INTERCEPT_DYNAMIC) && defined(LIBXS_BUILD) && \
    (defined(__GNUC__) || defined(_CRAYC)) && !defined(_WIN32) && !defined(__CYGWIN__) && \
   !(defined(__APPLE__) && defined(__MACH__) && LIBXS_VERSION2(6, 1) >= \
      LIBXS_VERSION2(__clang_major__, __clang_minor__))
# define LIBXS_INTERCEPT_DYNAMIC
#endif

#if !defined(LIBXS_MALLOC_HOOK_STATIC) && \
    (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD))) /* GLIBC */ && \
   (!defined(_WIN32)) /* TODO */
# define LIBXS_MALLOC_HOOK_STATIC
#endif
#if !defined(LIBXS_MALLOC_HOOK_DYNAMIC) && defined(LIBXS_INTERCEPT_DYNAMIC) && \
     defined(LIBXS_MALLOC_HOOK_STATIC) && !defined(_CRAYC) && !defined(__TRACE)
# define LIBXS_MALLOC_HOOK_DYNAMIC
#endif
#if (defined(LIBXS_MALLOC_HOOK_STATIC) || defined(LIBXS_MALLOC_HOOK_DYNAMIC))
# define LIBXS_MALLOC_HOOK
#endif
#if !defined(LIBXS_DNN_CONVOLUTION_SETUP_USE_NTS) && defined(LIBXS_MALLOC_HOOK) && \
    (defined(LIBXS_MALLOC_MOD) || (defined(LIBXS_MALLOC) && (0 != LIBXS_MALLOC)))
# define LIBXS_DNN_CONVOLUTION_SETUP_USE_NTS
#endif

#if defined(LIBXS_INTERCEPT_DYNAMIC)
# if defined(LIBXS_OFFLOAD_TARGET)
#   pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
# endif
# include <dlfcn.h>
# if defined(LIBXS_OFFLOAD_TARGET)
#   pragma offload_attribute(pop)
# endif
# if !defined(RTLD_NEXT)
#   define LIBXS_RTLD_NEXT ((void*)-1l)
# else
#   define LIBXS_RTLD_NEXT RTLD_NEXT
# endif
#endif

#if !defined(LIBXS_VERBOSITY_HIGH)
# define LIBXS_VERBOSITY_HIGH 3 /* secondary warning or info-verbosity */
#endif
#if !defined(LIBXS_VERBOSITY_WARN)
# define LIBXS_VERBOSITY_WARN ((LIBXS_VERBOSITY_HIGH) - LIBXS_MIN(1, LIBXS_VERBOSITY_HIGH))
#endif

#if !defined(LIBXS_LOCK)
# define LIBXS_LOCK LIBXS_LOCK_DEFAULT
#endif

#if !defined(LIBXS_EXT_MIN_NTASKS)
# define LIBXS_MIN_NTASKS(NT) 1
#endif
#if !defined(LIBXS_OVERHEAD)
# define LIBXS_OVERHEAD(NT) 0
#endif
#if !defined(LIBXS_NOOP_ARGS)
# define LIBXS_NOOP_ARGS(...)
#endif
#if !defined(LIBXS_NOOP)
# define LIBXS_NOOP
#endif

/** Check if M, N, K, or LDx fits into the descriptor. */
#if (0 != LIBXS_ILP64)
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) (0xFFFFFFFF >= (M) && 0xFFFFFFFF >= (N) && 0xFFFFFFFF >= (K))
#else /* always fits */
# define LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K) 1
#endif

#if defined(LIBXS_ASSERT) /* assert available */
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K) LIBXS_ASSERT(LIBXS_GEMM_NO_BYPASS_DIMS(M, N, K))
#else
# define LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K)
#endif

#if defined(LIBXS_UNPACKED)
# define LIBXS_DESCRIPTOR_CLEAR_AUX(DST, SIZE, FLAGS) LIBXS_MEMSET127(DST, 0, SIZE)
#else
# define LIBXS_DESCRIPTOR_CLEAR_AUX(DST, SIZE, FLAGS) \
    /*if (LIBXS_GEMM_FLAG_DESC_ISBIG <= (FLAGS)) LIBXS_MEMSET127(DST, 0, SIZE)*/
#endif
#define LIBXS_DESCRIPTOR_CLEAR(BLOB) \
  LIBXS_ASSERT((LIBXS_DESCRIPTOR_MAXSIZE) == sizeof(*(BLOB))); \
  LIBXS_DESCRIPTOR_CLEAR_AUX(BLOB, LIBXS_DESCRIPTOR_MAXSIZE, 0)

/** Low-level/internal GEMM descriptor initialization. */
#define LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(M, N, K); LIBXS_GEMM_DESCRIPTOR_DIM_CHECK(LDA, LDB, LDC); \
  LIBXS_DESCRIPTOR_CLEAR_AUX(&(DESCRIPTOR), sizeof(DESCRIPTOR), FLAGS); \
  (DESCRIPTOR).datatype = (unsigned char)(DATA_TYPE); (DESCRIPTOR).prefetch = (unsigned char)(PREFETCH); \
  (DESCRIPTOR).flags = (unsigned int)((FLAGS) | (LIBXS_NEQ(0, BETA) ? 0 : LIBXS_GEMM_FLAG_BETA_0)); \
  (DESCRIPTOR).m   = (unsigned int)(M);   (DESCRIPTOR).n   = (unsigned int)(N);   (DESCRIPTOR).k   = (unsigned int)(K); \
  (DESCRIPTOR).lda = (unsigned int)(LDA); (DESCRIPTOR).ldb = (unsigned int)(LDB); (DESCRIPTOR).ldc = (unsigned int)(LDC)

/** Similar to LIBXS_GEMM_DESCRIPTOR, but separately taking the input-/output-precision. */
#define LIBXS_GEMM_DESCRIPTOR2(DESCRIPTOR, IPREC, OPREC, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, LIBXS_GETENUM(IPREC, OPREC), FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/** Declare and construct a GEMM descriptor. */
#define LIBXS_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, DATA_TYPE, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  libxs_gemm_descriptor DESCRIPTOR; LIBXS_GEMM_DESCRIPTOR(DESCRIPTOR, DATA_TYPE, \
    FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

/** Similar to LIBXS_GEMM_DESCRIPTOR_TYPE, but separately taking the input-/output-precision. */
#define LIBXS_GEMM_DESCRIPTOR2_TYPE(DESCRIPTOR, IPREC, OPREC, FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH) \
  LIBXS_GEMM_DESCRIPTOR_TYPE(DESCRIPTOR, LIBXS_GETENUM(IPREC, OPREC), FLAGS, M, N, K, LDA, LDB, LDC, ALPHA, BETA, PREFETCH)

#define LIBXS_REGDESC_DEFAULT
#define LIBXS_REGDESC(START, MODIFIER) \
  START libxs_gemm_descriptor MODIFIER gemm; \
  START libxs_meltw_descriptor MODIFIER meltw; \
  START libxs_meqn_descriptor MODIFIER meqn

/**
* Packed structure, which stores the argument description of GEMM routines.
* The size of the structure is padded to LIBXS_DESCRIPTOR_MAXSIZE.
*/
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_gemm_descriptor {
  /** Extents of the matrix. */
  unsigned int m, n, k;
  /** Leading dimensions. */
  unsigned int lda, ldb, ldc;
  /** Set of flags. */
  unsigned int flags;
  /** Prefetch strategy. */
  unsigned char prefetch;
  /** Denotes the data-type. */
  unsigned char datatype;
  /**
   * Do not reorder elements between above and below blocks!
   */
  /** Denotes of optional eltwise data-type */
  unsigned char meltw_datatype_aux;
  /** multipurpose 64-bit field, currently used for: a) stride_a in brgemm */
  long long c1;
  /** multipurpose 64-bit field, currently used for: a) stride_b in brgemm */
  long long c2;
  /** multipurpose 8-bit field, currently used for: a) unroll hint in brgemm */
  unsigned char c3;
  /** LDx, LDy, LDz,  additional meltw LDs */
  unsigned int meltw_ldx, meltw_ldy, meltw_ldz;
  /** optional param field */
  unsigned short meltw_param;
  /** Set of flags */
  unsigned short meltw_flags;
  /** operation specifier */
  unsigned char meltw_operation;
  /* Ap, Bp, Cp */
  unsigned char eltw_ap_op;
  unsigned char eltw_bp_op;
  unsigned char eltw_cp_op;
  unsigned short eltw_ap_flags;
  unsigned short eltw_bp_flags;
  unsigned short eltw_cp_flags;
  unsigned short eltw_ap_param;
  unsigned short eltw_bp_param;
  unsigned short eltw_cp_param;
  unsigned int ldap;
  unsigned int ldbp;
  unsigned int ldcp;
  /* internal flags2 */
  unsigned char internal_flags_2;
};

/** Packed structure storing the mateltw argument description. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_meltw_descriptor {
  /** LDx, M, and N. */
  unsigned int m, n, ldi, ldo, ldi2, ldi3;
  /** Size of data element. */
  unsigned char datatype;
  unsigned char datatype1;
  unsigned char datatype2;
  /** Set of flags */
  unsigned short flags;
  /** optional param field */
  unsigned short param;
  /** operation specifier */
  unsigned char operation;
};

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_pspgemm_csr_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
  unsigned int packed_width;
} libxs_pspgemm_csr_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_pspgemm_csc_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* column_ptr;
  const unsigned int* row_idx;
  const void* values;
  unsigned int packed_width;
} libxs_pspgemm_csc_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_pgemm_ac_rm_descriptor {
  const libxs_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxs_pgemm_ac_rm_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_pgemm_bc_rm_descriptor {
  const libxs_gemm_descriptor* gemm;
  unsigned int packed_width;
} libxs_pgemm_bc_rm_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE LIBXS_MAY_ALIAS libxs_csr_reg_descriptor {
  const libxs_gemm_descriptor* gemm;
  const unsigned int* row_ptr;
  const unsigned int* column_idx;
  const void* values;
} libxs_csr_reg_descriptor;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_xcopykernel {
  libxs_meltwfunction_unary function;
  const void *ptr_const, *ptr;
} libxs_xcopykernel;

LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_code_pointer {
  void (*ptr_fn)(LIBXS_VARIADIC);
  const void* ptr_const;
  void* ptr;
  uintptr_t uval;
  intptr_t ival;
  libxs_xmmfunction xgemm; /* GEMM: smm, dmm, wimm, or void-function */
  libxs_xmeltwfunction xmateltw;
  libxs_matrix_eqn_function xmateqn;
} libxs_code_pointer;

struct LIBXS_RETARGETABLE libxs_fsspmdm {
  int M, N, K, ldb, ldc, N_chunksize;
  libxs_gemmfunction kernel;
  libxs_datatype datatype;
  void* a_dense;
};

/** Packed structure storing the mateltw argument description. */
LIBXS_EXTERN_C LIBXS_PACKED(struct LIBXS_RETARGETABLE) libxs_meqn_descriptor {
  /** LDx, M, and N. */
  unsigned int m, n, ldo;
  /** Size of data element. */
  unsigned char datatype;
  /** Set of flags */
  unsigned int eqn_idx;
};

typedef enum libxs_build_kind {
  LIBXS_BUILD_KIND_GEMM       = LIBXS_KERNEL_KIND_MATMUL,
  LIBXS_BUILD_KIND_MELTW      = LIBXS_KERNEL_KIND_MELTW,
  LIBXS_BUILD_KIND_MEQN       = LIBXS_KERNEL_KIND_MEQN,
  LIBXS_BUILD_KIND_USER       = LIBXS_KERNEL_KIND_USER,
  LIBXS_BUILD_KIND_PGEMMRMAC  = LIBXS_KERNEL_UNREGISTERED,
  LIBXS_BUILD_KIND_PGEMMRMBC,
  LIBXS_BUILD_KIND_PSPGEMM_CSR,
  LIBXS_BUILD_KIND_PSPGEMM_CSC,
  LIBXS_BUILD_KIND_SREG
} libxs_build_kind;

/** Integral type (libxs_kernel_kind, libxs_build_kind). */
#if defined(LIBXS_UNPACKED)
# define LIBXS_DESCRIPTOR_BIG(KIND) ((libxs_descriptor_kind)((KIND) | 0x8000000000000000))
# define LIBXS_DESCRIPTOR_ISBIG(KIND) ((int)(((libxs_descriptor_kind)(KIND)) >> 63))
# define LIBXS_DESCRIPTOR_KIND(KIND) ((int)(((libxs_descriptor_kind)(KIND)) & 0x7FFFFFFFFFFFFFFF))
typedef uint64_t libxs_descriptor_kind;
#else
# define LIBXS_DESCRIPTOR_BIG(KIND) ((libxs_descriptor_kind)((KIND) | 0x80))
# define LIBXS_DESCRIPTOR_ISBIG(KIND) ((int)((KIND) >> 7))
# define LIBXS_DESCRIPTOR_KIND(KIND) ((int)((KIND) & 0x7F))
typedef unsigned char libxs_descriptor_kind;
#endif

/** All descriptor types, which are valid for code-registration. */
LIBXS_EXTERN_C typedef union LIBXS_RETARGETABLE libxs_descriptor {
  unsigned char data[LIBXS_DESCRIPTOR_MAXSIZE];
  libxs_descriptor_kind kind; /* kind: must be the first member after "data" entry (above) */
  LIBXS_REGDESC(LIBXS_PACKED(struct) { libxs_descriptor_kind /*repeated kind*/ pad; , desc; });
  LIBXS_PACKED(struct) { libxs_descriptor_kind /*repeated kind*/ pad; unsigned char size; unsigned char desc[1]; } user;
} libxs_descriptor;

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_build_request {
  union {
    const void *ptr_const, *ptr; /* raw content */
    LIBXS_REGDESC(LIBXS_REGDESC_DEFAULT, const*);
    const libxs_pspgemm_csr_descriptor* pspgemm_csr;
    const libxs_pspgemm_csc_descriptor* pspgemm_csc;
    const libxs_pgemm_ac_rm_descriptor* pgemmacrm;
    const libxs_pgemm_bc_rm_descriptor* pgemmbcrm;
    const libxs_csr_reg_descriptor* sreg;
  } descriptor;
  libxs_build_kind kind;
  /* used by user-kind */
  size_t user_size;
} libxs_build_request;

typedef enum libxs_malloc_flags {
  LIBXS_MALLOC_FLAG_DEFAULT = 0,
  LIBXS_MALLOC_FLAG_SCRATCH = 1,
  LIBXS_MALLOC_FLAG_PRIVATE = 2,
  LIBXS_MALLOC_FLAG_REALLOC = 4,
  LIBXS_MALLOC_FLAG_PHUGE   = 8,
  LIBXS_MALLOC_FLAG_PLOCK   = 16,
  LIBXS_MALLOC_FLAG_MMAP    = 32,
  LIBXS_MALLOC_FLAG_R       = 64,
  LIBXS_MALLOC_FLAG_W       = 128,
  LIBXS_MALLOC_FLAG_X       = 256,
  LIBXS_MALLOC_FLAG_RW  = LIBXS_MALLOC_FLAG_R | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_WX  = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_W,
  LIBXS_MALLOC_FLAG_RWX = LIBXS_MALLOC_FLAG_X | LIBXS_MALLOC_FLAG_RW,
  LIBXS_MALLOC_FLAG_VALID       = LIBXS_MALLOC_FLAG_SCRATCH |
      LIBXS_MALLOC_FLAG_PRIVATE | LIBXS_MALLOC_FLAG_REALLOC |
      LIBXS_MALLOC_FLAG_PHUGE   | LIBXS_MALLOC_FLAG_PLOCK |
      LIBXS_MALLOC_FLAG_MMAP    | LIBXS_MALLOC_FLAG_RWX
} libxs_malloc_flags;

LIBXS_EXTERN_C typedef LIBXS_RETARGETABLE void* (*libxs_realloc_fun)(void* /*ptr*/, size_t /*size*/);

#if defined(LIBXS_MALLOC_HOOK_DYNAMIC)
LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_malloc_fntype {
  union { const void* dlsym; void* (*ptr)(size_t, size_t);  } alignmem;
  union { const void* dlsym; void* (*ptr)(size_t, size_t);  } memalign;
  union { const void* dlsym; libxs_malloc_fun ptr;        } malloc;
# if defined(LIBXS_MALLOC_HOOK_CALLOC)
  union { const void* dlsym; void* (*ptr)(size_t, size_t);  } calloc;
# endif
# if defined(LIBXS_MALLOC_HOOK_REALLOC)
  union { const void* dlsym; libxs_realloc_fun ptr;      } realloc;
# endif
  union { const void* dlsym; libxs_free_fun ptr;          } free;
} libxs_malloc_fntype;
LIBXS_APIVAR_PRIVATE(libxs_malloc_fntype libxs_malloc_fn);
#endif

#if (defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD)))
/* prototypes for GLIBC internal implementation */
LIBXS_EXTERN_C LIBXS_RETARGETABLE void* __libc_memalign(size_t alignment, size_t size);
LIBXS_EXTERN_C LIBXS_RETARGETABLE void* __libc_malloc(size_t size);
#if defined(LIBXS_MALLOC_HOOK_CALLOC)
LIBXS_EXTERN_C LIBXS_RETARGETABLE void* __libc_calloc(size_t num, size_t size);
#endif
#if defined(LIBXS_MALLOC_HOOK_REALLOC)
LIBXS_EXTERN_C LIBXS_RETARGETABLE void* __libc_realloc(void* ptr, size_t size);
#endif
LIBXS_EXTERN_C LIBXS_RETARGETABLE void  __libc_free(void* ptr);
#endif /*(defined(LIBXS_BUILD) && (1 < (LIBXS_BUILD)))*/
LIBXS_API_INTERN void* libxs_memalign_internal(size_t alignment, size_t size);

/* See https://sourceware.org/binutils/docs-2.34/ld/Options.html#index-_002d_002dwrap_003dsymbol */
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_memalign(size_t alignment, size_t size);
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_malloc(size_t size);
#if defined(LIBXS_MALLOC_HOOK_CALLOC)
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_calloc(size_t num, size_t size);
#endif
#if defined(LIBXS_MALLOC_HOOK_REALLOC)
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void* __real_realloc(void* ptr, size_t size);
#endif
LIBXS_API_INTERN LIBXS_ATTRIBUTE_WEAK void __real_free(void* ptr);

/** Retrieve internal information about a buffer (default memory domain). */
LIBXS_API int libxs_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra);

/** Initializes malloc hooks and other internals. */
LIBXS_API_INTERN void libxs_malloc_init(void);
LIBXS_API_INTERN void libxs_malloc_finalize(void);

/** Calculates an alignment depending on supposedly allocated size; alignment can be zero ("auto"). */
LIBXS_API_INTERN size_t libxs_alignment(size_t size, size_t alignment);

/** Same as libxs_set_default_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xset_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_default_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xget_default_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/** Same as libxs_set_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xset_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void* context, libxs_malloc_function malloc_fn, libxs_free_function free_fn);
/** Same as libxs_get_scratch_allocator, but takes a lock (can be NULL). */
LIBXS_API_INTERN int libxs_xget_scratch_allocator(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock,
  const void** context, libxs_malloc_function* malloc_fn, libxs_free_function* free_fn);

/**
 * Attribute memory allocation and protect with only the necessary flags (revoke other flags).
 * This procedure is not suitable for executable buffers, profiler support, etc.
 */
LIBXS_API_INTERN int libxs_malloc_xattrib(void* buffer, int flags, size_t size);

/**
 * Attribute memory allocation and protect with only the necessary flags.
 * This procedure is expected to run only one time per buffer, and may
 * relocate the given memory.
 */
LIBXS_API_INTERN int libxs_malloc_attrib(void** memory, int flags,
  /** If name is given, profiler support, and code dump (verbose mode) are supported. */
  const char* name,
  /** If data_size if given, amount of memory-attribution is lowered by data_size. */
  const size_t* data_size);

/** Like libxs_release_scratch, but takes a lock (can be NULL). */
LIBXS_API_INTERN void libxs_xrelease_scratch(LIBXS_LOCK_TYPE(LIBXS_LOCK)* lock);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXS_API int libxs_xmalloc(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
/** Release memory, which was allocated using libxs_[*]malloc. */
LIBXS_API void libxs_xfree(const void* memory, int check);

/**
 * Format for instance an amount of Bytes like libxs_format_value(result, sizeof(result), nbytes, "KMGT", "B", 10).
 * The value returned is in requested/determined unit so that the user can decide about printing the buffer.
 */
LIBXS_API_INTERN size_t libxs_format_value(char buffer[32], int buffer_size, size_t nbytes, const char scale[], const char* unit, int base);

/** Returns the type-name of data-type (can be also libxs_datatype). */
LIBXS_API_INTERN const char* libxs_typename(libxs_datatype datatype);

/** Dump data and (optionally) checks attempt to dump different data into an existing file (unique). */
LIBXS_API_INTERN int libxs_dump(const char* title, const char* name, const void* data, size_t size, int unique);

/** Services a build request, and (optionally) registers the code (use regindex=LIBXS_CAPACITY_REGISTRY for unmanaged code). */
LIBXS_API_INTERN int libxs_build(const libxs_build_request* request, unsigned int regindex, libxs_code_pointer* code);

/** Returns the type-size of data-type (can be also libxs_datatype). */
LIBXS_API unsigned char libxs_typesize(libxs_datatype datatype);

LIBXS_EXTERN_C typedef struct LIBXS_RETARGETABLE libxs_kernel_xinfo {
  /** Non-zero if kernel is registered. */
  unsigned int registered;
  /** Number of FLoating Point OPerationS (FLOPS). */
  unsigned int nflops;
} libxs_kernel_xinfo;

/** Receive information about JIT-generated code. */
LIBXS_API_INTERN const libxs_kernel_xinfo* libxs_get_kernel_xinfo(libxs_code_pointer code, const libxs_descriptor** desc, size_t* code_size);

/** Calculates duration in seconds from given RTC ticks. */
LIBXS_API_INTERN double libxs_timer_duration_rtc(libxs_timer_tickint tick0, libxs_timer_tickint tick1);
/** Returns the current tick of platform-specific real-time clock. */
LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_rtc(void);
/** Returns the current tick of a (monotonic) platform-specific counter. */
LIBXS_API_INTERN libxs_timer_tickint libxs_timer_tick_tsc(void);

LIBXS_API_INTERN void libxs_memory_init(int target_arch);
LIBXS_API_INTERN void libxs_memory_finalize(void);

/** Global lock; create an own lock for an independent domain. */
LIBXS_APIVAR_PUBLIC(LIBXS_LOCK_TYPE(LIBXS_LOCK) libxs_lock_global);
/** Determines whether a threaded implementation is synchronized or not. */
LIBXS_APIVAR_PUBLIC(int libxs_nosync);

/** Function used to allocate default memory. */
LIBXS_APIVAR_PRIVATE(libxs_malloc_function libxs_default_malloc_fn);
/** Function used to allocate scratch memory. */
LIBXS_APIVAR_PRIVATE(libxs_malloc_function libxs_scratch_malloc_fn);
/** Function used to release default memory. */
LIBXS_APIVAR_PRIVATE(libxs_free_function libxs_default_free_fn);
/** Function used to release scratch memory. */
LIBXS_APIVAR_PRIVATE(libxs_free_function libxs_scratch_free_fn);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXS_APIVAR_PRIVATE(const void* libxs_default_allocator_context);
/** If non-NULL, this context is used by the context-form of memory allocation. */
LIBXS_APIVAR_PRIVATE(const void* libxs_scratch_allocator_context);
/** Number of scratch memory pools used; clamped against internal maximum. */
LIBXS_APIVAR_PRIVATE(unsigned int libxs_scratch_pools);
/** Growth factor used to scale the scratch memory in case of reallocation. */
LIBXS_APIVAR_PRIVATE(double libxs_scratch_scale);
/** Number of seconds per RDTSC-cycle (zero or negative if RDTSC invalid). */
LIBXS_APIVAR_PRIVATE(double libxs_timer_scale);
/** Counts the number of attempts to create an SPMDM-handle. */
LIBXS_APIVAR_PRIVATE(unsigned int libxs_statistic_num_spmdm);
/** Counts the maximum number of thread that have been active. */
LIBXS_APIVAR_PRIVATE(unsigned int libxs_thread_count);

#endif /*LIBXS_MAIN_H*/
