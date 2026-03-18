# CPU Identification

Header: `libxs_cpuid.h`

Portable CPU feature detection for x86-64, AArch64, and RISC-V targets. Returns an ISA level that can be compared numerically — higher values indicate more capable instruction sets.

## ISA Constants

| Constant | Value | Architecture |
|---|---|---|
| `LIBXS_TARGET_ARCH_UNKNOWN` | 0 | Unknown / unsupported |
| `LIBXS_TARGET_ARCH_GENERIC` | 1 | Portable scalar baseline |
| `LIBXS_X86_GENERIC` | 1002 | x86-64 baseline |
| `LIBXS_X86_SSE3` | 1003 | SSE3 |
| `LIBXS_X86_SSE42` | 1004 | SSE4.2 |
| `LIBXS_X86_AVX` | 1005 | AVX |
| `LIBXS_X86_AVX2` | 1006 | AVX2 + FMA |
| `LIBXS_X86_AVX512` | 1100 | AVX-512 (Foundation + VL/BW/DQ) |
| `LIBXS_AARCH64` | 2001 | ARMv8.1 baseline |
| `LIBXS_AARCH64_SVE128` | 2201 | SVE 128-bit |
| `LIBXS_AARCH64_SVE256` | 2301 | SVE 256-bit |
| `LIBXS_AARCH64_SVE512` | 2401 | SVE 512-bit |
| `LIBXS_RV64_MVL128` | 3001 | RISC-V RVV 128-bit |
| `LIBXS_RV64_MVL256` | 3002 | RISC-V RVV 256-bit |
| `LIBXS_RV64_MVL128_LMUL` | 3003 | RISC-V RVV 128-bit (non-unit LMUL) |
| `LIBXS_RV64_MVL256_LMUL` | 3004 | RISC-V RVV 256-bit (non-unit LMUL) |
| `LIBXS_X86_ALLFEAT` | 1999 | x86 sentinel (all features) |
| `LIBXS_AARCH64_ALLFEAT` | 2999 | AArch64 sentinel (all features) |
| `LIBXS_RV64_ALLFEAT` | 3999 | RISC-V sentinel (all features) |

## Types

```C
typedef struct libxs_cpuid_info_t {
  char model[256];   /* CPU model name from OS */
  int constant_tsc;  /* non-zero if TSC is invariant */
} libxs_cpuid_info_t;
```

## Functions

```C
int libxs_cpuid(libxs_cpuid_info_t* info);
```

Detect the ISA level of the current platform. Optionally fills `info` with the CPU model name and TSC capability. Returns a constant from the table above.

```C
const char* libxs_cpuid_name(int id);
```

Returns a human-readable string for the given ISA constant, e.g., `"avx2"`.

```C
int libxs_cpuid_id(const char* name);
```

Translates a name (e.g., `"avx512"`) to the corresponding ISA constant.

```C
int libxs_cpuid_vlen(int id);
```

Returns the SIMD vector length in bytes for the given ISA level (0 for scalar-only targets).
