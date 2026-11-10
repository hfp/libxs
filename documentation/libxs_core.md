# Core: Initialization and Common Types

Header: `libxs.h`

Library initialization, verbosity, the element-type enumeration shared by the
other modules, and the common macros of `libxs_macros.h` (included by every
LIBXS header). Environment variables (e.g., `LIBXS_VERBOSE`) are listed in the
Usage section of the [overview](index.md#usage).

## Initialization

```C
void libxs_init(void);
void libxs_finalize(void);
```

`libxs_init` is optional: functions that need the library initialize it on first
use, and library builds initialize at load time where the compiler supports
constructors. Call it to take the setup cost at a chosen point; repeated calls
have no effect. `libxs_finalize` releases internal memory such as the default
memory pool; it is optional as well and runs at unload where supported.

## Verbosity

```C
int libxs_get_verbosity(void);
void libxs_set_verbosity(int level);
```

| Level | Output |
|:------|:-------|
| 0 | None |
| 1 | Errors |
| 2 | Warnings |
| 3 | Information and reports (e.g., memory pool statistics) |
| negative | Everything |

The initial level comes from `LIBXS_VERBOSE` (an empty value means 1), and is
the maximum in debug builds when the variable is unset.

One warning is printed by `libxs_init` even when `LIBXS_VERBOSE` is unset: a
sequential OpenBLAS together with an OpenMP runtime (unless `OMP_NUM_THREADS=1`),
since such an OpenBLAS may corrupt results when called from several threads
(unless it was built with `USE_LOCKING=1`). Link the OpenMP or pthread build of
OpenBLAS, or set `LIBXS_VERBOSE=0` to mute the warning.

## Element Types

```C
typedef enum libxs_data_t {
  LIBXS_DATATYPE_F64 = 0 | (8 << 4),
  LIBXS_DATATYPE_F32 = 1 | (4 << 4),
  LIBXS_DATATYPE_C64 = 2 | (16 << 4),
  LIBXS_DATATYPE_C32 = 3 | (8 << 4),
  LIBXS_DATATYPE_I64 = 4 | (8 << 4),
  LIBXS_DATATYPE_U64 = 5 | (8 << 4),
  LIBXS_DATATYPE_I32 = 6 | (4 << 4),
  LIBXS_DATATYPE_U32 = 7 | (4 << 4),
  LIBXS_DATATYPE_I16 = 8 | (2 << 4),
  LIBXS_DATATYPE_U16 = 9 | (2 << 4),
  LIBXS_DATATYPE_I8  = 10 | (1 << 4),
  LIBXS_DATATYPE_U8  = 11 | (1 << 4),
  LIBXS_DATATYPE_UNKNOWN = 12
} libxs_data_t;
```

Real, complex (C64 is a pair of `double`, C32 a pair of `float`), signed and
unsigned integer types. Each enumerator carries the size of its type.

| Macro | Description |
|:------|:------------|
| `LIBXS_TYPESIZE(ENUM)` | Size of the type in bytes |
| `LIBXS_TYPEORDER(ENUM)` | Enumerator without the size |
| `LIBXS_DATATYPE(TYPE)` | Enumerator of a C type: `double`, `float`, `int`, `short`, or `char` |

```C
const char* libxs_typename(libxs_data_t datatype);
```

Name of the type: `"f64"`, `"f32"`, `"c64"`, `"c32"`, `"i64"`, `"u64"`,
`"i32"`, `"u32"`, `"i16"`, `"u16"`, `"i8"`, `"u8"`, or `"void"` if unknown.

```C
int libxs_dvalue(libxs_data_t datatype, const void* value, double* dvalue);
```

Convert the real or integer value at `value` to `double`. Returns
`EXIT_SUCCESS`, or `EXIT_FAILURE` for a NULL `value`, a complex, or an unknown
type.

## Version

| Macro | Description |
|:------|:------------|
| `LIBXS_VERSION_MAJOR`, `_MINOR`, `_UPDATE`, `_PATCH` | Version components |
| `LIBXS_VERSION_NUMBER` | `MAJOR * 10000 + MINOR * 100 + UPDATE` |
| `LIBXS_VERSION_CHECK(COMP, MAJOR, MINOR, UPDATE)` | Compare the version, e.g., `LIBXS_VERSION_CHECK(<, 2, 0, 0)` |
| `LIBXS_VERSION_GE(MAJOR, MINOR, UPDATE)` | Version is at least the given one |
| `LIBXS_VERSION2(MAJOR, MINOR)`, `LIBXS_VERSION3(MAJOR, MINOR, UPDATE)` | Encode a version number (also for compiler versions) |
| `LIBXS_BUILD_DATE` | Build date as `YYYYMMDD` |

The values come from `libxs_version.h`, which the build generates (`make` or
CMake) alongside the headers; it also defines `LIBXS_VERSION` (string) and
`LIBXS_BRANCH`. `libxs_macros.h` includes it wherever it is found, which
needs `__has_include` outside of the LIBXS build itself. Without it, the
components default to 0, except in a header-only build, which compiles all of
LIBXS and hence reports 9999.99.99 so that no version gate excludes code.

```C
const char* libxs_version(int* major, int* minor, int* update, int* patch);
```

Version of the library as built, which can differ from the macros an
application was compiled with (e.g., a newer shared library). Returns the
version string, and stores the components into the arguments that are not
NULL. From Fortran (`USE LIBXS`), the arguments are OPTIONAL and the result
is a `TYPE(C_PTR)` to the string.

## Common Macros

| Macro | Description |
|:------|:------------|
| `LIBXS_MIN(A, B)`, `LIBXS_MAX(A, B)` | Minimum, maximum |
| `LIBXS_ABS(A)`, `LIBXS_SIGN(A)` | Absolute value, sign (-1, 0, 1) |
| `LIBXS_DELTA(A, B)` | Absolute difference |
| `LIBXS_CLMP(VALUE, LO, HI)` | Clamp to `[LO, HI]` |
| `LIBXS_UPDIV(N, MULT)` | `N / MULT` rounded up |
| `LIBXS_UP(N, MULT)` | `N` rounded up to a multiple of `MULT` |
| `LIBXS_UP2(N, NPOT)`, `LIBXS_LO2(N, NPOT)` | Round up/down to a multiple of a power of two |
| `LIBXS_UP2POT(N)`, `LIBXS_LO2POT(N)` | Round up/down to a power of two |
| `LIBXS_ISPOT(A)` | Non-zero if `A` is a power of two |
| `LIBXS_MOD2(A, NPOT)` | `A` modulo a power of two |
| `LIBXS_DELIMS` | Default list separators `",;:"`; the first character is the default output separator |
| `LIBXS_CACHELINE`, `LIBXS_ALIGNMENT` | Cache-line size and default alignment (64 bytes) |

Arguments may be evaluated more than once.

Platform symbols: `LIBXS_PLATFORM_X86`, `LIBXS_PLATFORM_AARCH64`, or
`LIBXS_PLATFORM_RV64` is defined for the target, and `LIBXS_BITS` is the width
of a pointer.

## API Declarations for Dependent Libraries

A library depending on LIBXS can reuse its declaration scheme, which covers a
shared or static build of the library, its users, and header-only use:

```C
#if defined(MYLIB_SOURCE_H)
# define MYLIB_BUILD_KIND LIBXS_APIKIND_INLINE
#elif defined(MYLIB_BUILD)
# define MYLIB_BUILD_KIND LIBXS_APIKIND_EXPORT
#else
# define MYLIB_BUILD_KIND LIBXS_APIKIND_IMPORT
#endif
#define MYLIB_API        LIBXS_API_DECL(MYLIB_BUILD_KIND)
#define MYLIB_API_INTERN LIBXS_API_DECL_INTERN(MYLIB_BUILD_KIND)
```

`LIBXS_API_DECL(KIND)` decorates a public function, `LIBXS_API_DECL_INTERN(KIND)`
a function that is not exported. Variables use `LIBXS_APIVAR_DECL_PUBLIC(DECL, KIND)`
in a header with `LIBXS_APIVAR_DECL_PUBLIC_DEF(DECL, KIND)` in the source file,
or the `_PRIVATE` and `_PRIVATE_DEF` counterparts, and
`LIBXS_APIVAR_DECL_DEFINE(DECL, KIND)` declares and defines a private variable
at once.

## Header-Only Use

Including `libxs_source.h`, or compiling with `-DLIBXS_SOURCE`, provides the
implementation without linking a library (see the Usage section of the
[overview](index.md#usage)). `LIBXS_BUILD` must not be defined in this case.
