# Ozaki-Scheme Low-Precision GEMM

Intercepts DGEMM, SGEMM, ZGEMM, and CGEMM at link time, replacing
them with low-precision Ozaki schemes (mantissa slicing or CRT). Real
GEMM calls go through the selected scheme; complex GEMM (ZGEMM/CGEMM)
is mapped to real GEMM internally.

## Build

```bash
cd samples/ozaki
make GNU=1 -j $(nproc)
```

LIBXS must be built first from the repository root. When a sibling
`libxstream` directory is detected, the optional OpenCL/GPU path is
compiled in (see OZAKI_OCL below).

## Link-Time Interception

Two link-time variants are built per precision:

- `dgemm-blas.x` / `sgemm-blas.x` -- dynamically linked against BLAS
- `dgemm-wrap.x` / `sgemm-wrap.x` -- linked with `--wrap=` (GNU ld)
- `zgemm-wrap.x` / `cgemm-wrap.x` -- complex GEMM wrappers

Static (`libwrap.a`) and shared (`libwrap.so`) wrapper libraries are
built by default. The shared library can intercept any application:

```bash
LD_PRELOAD=/path/to/libwrap.so ./application
```

The static library requires explicit `--wrap` flags:

```bash
-Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_
```

Test scripts: `test-wrap.sh` exercises all variants, `test-check.sh`
validates both schemes, `test-mhd.sh` tests MHD file-based input.

## Test Driver

```bash
dgemm-wrap.x [A.mhd|M [B.mhd|N [K [TA [TB [ALPHA [BETA [LDA [LDB [LDC]]]]]]]]]]
```

Defaults: M=N=K=257, TA=TB=0, ALPHA=BETA=1.0. The first argument can
be 0 followed by MHD filenames to load A and B matrices from files.
The `zgemm-wrap.x` and `cgemm-wrap.x` drivers call ZGEMM and CGEMM.

## Environment Variables

### Scheme Selection

| Variable        | Default   | Description                                                                        |
|-----------------|-----------|------------------------------------------------------------------------------------|
| OZAKI           | 1         | 0=bypass (original BLAS), 1=mantissa slicing, 2=CRT                                |
| OZAKI_COMPLEX   | (auto)    | Complex dispatch: 0=BLAS, 1=CPU, 2=GPU+fallback. Auto: 0 if OZAKI=0, else 2        |
| OZAKI_N         | (auto)    | Slices (Sch.1: fp64=8, fp32=4) or primes (Sch.2: fp64=16, fp32=9; max 20)          |

### Accuracy

| Variable          | Default   | Description                                                                      |
|-------------------|-----------|----------------------------------------------------------------------------------|
| OZAKI_FLAGS       | 3         | Sch.1 bitmask: 1=Triangular, 2=Symmetrize, 0=full S^2 square                     |
| OZAKI_TRIM        | 0         | Levels to trim (0=exact). ~7 bits/level (Sch.1), ~4 bits/level (Sch.2)           |
| OZAKI_I8          | 0         | Sch.2: signed i8 residues (moduli<=128) instead of u8                            |
| OZAKI_GROUPS      | 0         | Sch.2: K-grouping (0/1=off). Consecutive K panels share one reconstruction       |
| OZAKI_MAXK        | 32768     | Max K per preprocessing pass (0=full K in one pass)                              |
| OZAKI_THRESHOLD   | 12        | Intensity threshold. Bypass when flops/(bytes*thr)<1. 0=always apply             |

### GPU Path (requires LIBXSTREAM)

| Variable    | Default   | Description                                |
|-------------|-----------|--------------------------------------------|
| OZAKI_OCL   | 0         | Enable OpenCL/GPU path (0=off, 1=on)       |
| OZAKI_TM    | (auto)    | GPU output tile height (multiple of 8)     |
| OZAKI_TN    | (auto)    | GPU output tile width (multiple of 16)     |

GPU-specific kernel tuning variables (OZAKI_RTM, OZAKI_RTN, OZAKI_WG,
OZAKI_SG, OZAKI_KU, OZAKI_RC, OZAKI_PB, OZAKI_PREFETCH, OZAKI_BOUNDS,
OZAKI_SCALAR_ACC, OZAKI_TINYTC, OZAKI_DEVPOOL, OZAKI_CACHE) are
documented in the LIBXSTREAM Ozaki README.

### Monitoring and Diagnostics

| Variable        | Default   | Description                                                                |
|-----------------|-----------|----------------------------------------------------------------------------|
| OZAKI_VERBOSE   | 0         | 0=silent, 1=stats at exit, N=print every Nth GEMM call                     |
| OZAKI_STAT      | 0         | Track C-matrix (0), A-representation (1), or B-representation (2)          |
| OZAKI_DUMP      | 0         | Dump A/B as MHD files at the given call-count (0=off)                      |
| OZAKI_EPS       | inf       | Dump A/B when epsilon error exceeds threshold                              |
| OZAKI_RSQ       | 0         | Dump A/B when RSQ drops below threshold (updated after dump)               |
| OZAKI_EXIT      | 1         | Exit on accuracy violation after dump. 0=continue                          |
| OZAKI_PROFILE   | 0         | Profile: 0=off, 1=all phases, 2=kernel, 3=preprocess A, 4=preprocess B     |

### Benchmark

| Variable   | Default   | Description                                                                |
|------------|-----------|----------------------------------------------------------------------------|
| CHECK      | 0         | Validate vs BLAS: 0=off, negative=auto-threshold, positive=custom          |
| NREPEAT    | 3         | Number of GEMM calls (first call is warmup when >1)                        |

## Profiling

The OZAKI_PROFILE variable enables per-GEMM timing collected into a
histogram reported at program exit:

```text
OZAKI PROF: 850 DP-GFLOPS/s (17.0 INT8-TOPS/s, 20x)
```

Profile modes select which phase is measured:

| Mode         | CPU                           | GPU                    |
|--------------|-------------------------------|------------------------|
| 1/negative   | All phases (preprocess+dot)   | All profiled kernels   |
| 2            | Kernel only (dot products)    | Dotprod kernel only    |
| 3            | Preprocessing (A+B)           | Preprocess A kernel    |
| 4            | Preprocessing (A+B)           | Preprocess B kernel    |

On the CPU, modes 3 and 4 are equivalent (A and B preprocessing is
interleaved across OpenMP threads).

## Example

```bash
./dgemm-wrap.x 256                          # exact (default flags=3, trim=0)
OZAKI_TRIM=4 ./dgemm-wrap.x 256             # drop 4 least significant diagonals
OZAKI=2 ./dgemm-wrap.x 256                  # CRT scheme (u8 default)
OZAKI=2 OZAKI_GROUPS=4 ./dgemm-wrap.x 4096  # CRT with K-grouping
OZAKI_FLAGS=0 ./dgemm-wrap.x 256            # full S^2 square, no symmetrize
```
