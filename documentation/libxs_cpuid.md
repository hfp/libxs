# CPU Identification

Header: `libxs_cpuid.h`

Portable CPU feature detection for x86-64, AArch64, and RISC-V targets. Returns an ISA level that can be compared numerically — higher values indicate more capable instruction sets. x86 levels use thermometer ordering: higher numeric value implies all features of lower levels. AVX10/256 sits below AVX512 because it lacks 512-bit vectors.

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
| `LIBXS_X86_AVX10_256` | 1030 | AVX10.1/256: all features, 256-bit max (Sierra Forest) |
| `LIBXS_X86_AVX512` | 1100 | AVX-512 (F + CD + DQ + BW + VL + VNNI) |
| `LIBXS_X86_AVX512_AMX` | 1105 | AVX-512 + AMX (TILE + INT8 + BF16); Sapphire/Granite Rapids |
| `LIBXS_X86_AVX512_INT8` | 1110 | AVX-512 + AVX-VNNI-INT8 (VPDPBUUD/BSSD) |
| `LIBXS_X86_AVX10_512` | 1200 | AVX10.1/512 + AMX + INT8: full feature set |
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
typedef struct libxs_cpuid_t {
  char model[256];   /* CPU model name from OS */
  int constant_tsc;  /* non-zero if TSC is invariant */
} libxs_cpuid_t;
```

## Functions

```C
int libxs_cpuid(libxs_cpuid_t* info);
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

```C
int libxs_cpuid_amx_enable(void);
```

Request AMX tile state (XTILE_DATA) from the OS. Must be called before any AMX tile instruction. Returns `EXIT_SUCCESS` if tiles are ready (or were already enabled), `EXIT_FAILURE` on unsupported platform or OS refusal. Not called automatically by `libxs_cpuid` to avoid the per-thread XSAVE overhead for non-AMX workloads.
