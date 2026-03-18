# Low-Level Utilities and SIMD

Header: `libxs_utils.h`

Target-attribute machinery, ISA feature gates, intrinsic fix-ups, bit-scan operations, and AVX-512 inline helpers.

## Target Attributes

```C
LIBXS_INTRINSICS(TARGET)
```

Expands to a `__attribute__((target(...)))` clause for the given ISA level. Used to compile individual functions for a higher ISA than the baseline without changing compiler flags. The macro is a no-op when the baseline already covers the requested level or when the compiler does not support multi-versioning.

```C
LIBXS_ATTRIBUTE_TARGET(TARGET)
```

Lower-level macro that resolves `TARGET` (an integer ISA constant such as `LIBXS_X86_AVX512`) to the corresponding target string, e.g. `target("avx2,fma,avx512f,...")`.

## ISA Feature Gates

Preprocessor symbols defined when the compiler can generate code for a given ISA:

| Macro | ISA level |
|:------|:----------|
| `LIBXS_INTRINSICS_X86` | x86 baseline (SSE2) |
| `LIBXS_INTRINSICS_SSE3` | SSE3 |
| `LIBXS_INTRINSICS_SSE42` | SSE4.2 |
| `LIBXS_INTRINSICS_AVX` | AVX |
| `LIBXS_INTRINSICS_AVX2` | AVX2 + FMA |
| `LIBXS_INTRINSICS_AVX512` | AVX-512 (F + CD + DQ + BW + VL) |

`LIBXS_STATIC_TARGET_ARCH` records the highest ISA provided by the compiler flags. `LIBXS_MAX_STATIC_TARGET_ARCH` records the highest ISA reachable via target attributes.

## Intrinsic Fix-ups

Portability wrappers for intrinsics whose signatures differ across compilers:

| Macro | Purpose |
|:------|:--------|
| `LIBXS_INTRINSICS_LOADU_SI128` | `_mm_loadu_si128` |
| `LIBXS_INTRINSICS_MM512_LOAD_PS/PD` | Unaligned 512-bit float/double load |
| `LIBXS_INTRINSICS_MM512_STREAM_*` | Non-temporal 512-bit stores |
| `LIBXS_INTRINSICS_MM256_STORE_EPI32` | 256-bit integer store |
| `LIBXS_INTRINSICS_MM512_SET_EPI16` | Portable 512-bit set from 32 int16 values |
| `LIBXS_INTRINSICS_MM512_UNDEFINED_*` | Undefined-value initializers (zero in debug) |
| `LIBXS_INTRINSICS_MM512_EXTRACTI64X4_EPI64` | 256-bit extract from 512-bit |
| `LIBXS_INTRINSICS_MM512_ABS_PS` | Absolute value of packed floats |
| `LIBXS_INTRINSICS_MM512_MASK_I32GATHER_EPI32` | Masked gather |
| `LIBXS_INTRINSICS_MM512_STORE/LOAD_MASK*` | Mask register I/O (8/16/32/64-bit) |
| `LIBXS_INTRINSICS_MM512_CVTU32_MASK*` | Convert uint32 to mask register |

## Bit-Scan Operations

```C
LIBXS_INTRINSICS_BITSCANFWD32(N)   /* count trailing zeros, 32-bit */
LIBXS_INTRINSICS_BITSCANFWD64(N)   /* count trailing zeros, 64-bit */
LIBXS_INTRINSICS_BITSCANBWD32(N)   /* bit index of highest set bit, 32-bit */
LIBXS_INTRINSICS_BITSCANBWD64(N)   /* bit index of highest set bit, 64-bit */
```

Use GCC `__builtin_ctz`/`__builtin_clz` when available, Windows `_BitScanForward`/`_BitScanReverse` intrinsics on MSVC, or portable software fallbacks (`*_SW` variants). All return 0 when `N` is 0.

## Derived Macros

```C
LIBXS_NBITS(N)     /* minimum bits to represent N */
LIBXS_ISQRT2(N)    /* fast power-of-two approximation of sqrt(N) */
```

```C
unsigned int LIBXS_ILOG2(unsigned long long n);   /* ceil(log2(n)) */
```

`LIBXS_ILOG2` is a function (not a macro) and handles the n = 0 case.

## AVX-512 Inline Functions

Available only when `LIBXS_INTRINSICS_AVX512` is defined.

```C
__m512i libxs_mulhi_epu32(__m512i a, __m512i b);   /* inline */
```

Unsigned 32-bit high-multiply for 16 lanes: floor(a * b / 2^32). Emulates the missing `_mm512_mulhi_epu32` via even/odd `_mm512_mul_epu32`.

```C
__m512i libxs_mod_u32x16(__m512i x,
  unsigned int p, unsigned int rcp);               /* inline */
```

Vectorized Barrett reduction: x mod p for 16 uint32 lanes. `rcp` is the Barrett reciprocal from `libxs_barrett_rcp()`. This is the SIMD counterpart of the scalar `libxs_mod_u32`.
