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

| Variable        | Default   | Description                                                       |
|-----------------|-----------|-------------------------------------------------------------------|
| OZAKI           | 2         | 0=bypass (BLAS), 1=mantissa slicing, 2=CRT, 3=adaptive            |
| OZAKI_COMPLEX   | (auto)    | Complex dispatch: 0=BLAS, 1=CPU, 2=GPU+fallback. Auto: 2 if on    |
| OZAKI_N         | (auto)    | Slices (Sch.1: fp64=8, fp32=4) or primes (Sch.2: fp64=16, fp32=9) |

OZAKI=3 (adaptive) starts with Scheme 1 on the first GPU call to
learn the effective cutoff from preprocessing occupancy. Subsequent
calls compare the Scheme-1 pair count (at the cached cutoff) against
the Scheme-2 prime count and pick the cheaper path. On CPU, adaptive
falls back to Scheme 2.

### Accuracy

| Variable          | Default   | Description                                                      |
|-------------------|-----------|------------------------------------------------------------------|
| OZAKI_FLAGS       | 3         | Sch.1 bitmask: 1=Triangular, 2=Symmetrize, 0=full S^2 square     |
| OZAKI_TRIM        | 0         | Levels to trim (0=exact). ~7 bits/level (Sch.1), ~4 bits (Sch.2) |
| OZAKI_I8          | 0         | Sch.2: signed i8 residues (moduli<=128) instead of u8            |
| OZAKI_GROUPS      | 0         | Sch.2: K-grouping (0/1=off). Consecutive K panels, one reconstr. |
| OZAKI_MAXK        | 32768     | Max K per preprocessing pass (0=full K in one pass)              |
| OZAKI_THRESHOLD   | 12        | Intensity threshold. Bypass when flops/(bytes\*thr)<1. 0=always  |

### GPU Path (requires LIBXSTREAM)

| Variable    | Default   | Description                                |
|-------------|-----------|--------------------------------------------|
| OZAKI_OCL   | 0         | Enable OpenCL/GPU path (0=off, 1=on)       |
| OZAKI_TM    | (auto)    | GPU output tile height (multiple of 8)     |
| OZAKI_TN    | (auto)    | GPU output tile width (multiple of 16)     |

GPU-specific kernel tuning variables (OZAKI_RTM, OZAKI_RTN, OZAKI_WG,
OZAKI_SG, OZAKI_KU, OZAKI_RC, OZAKI_PB, OZAKI_HIER, OZAKI_PREFETCH,
OZAKI_SCALAR_ACC, OZAKI_DEVPOOL, OZAKI_CACHE) are documented in the
LIBXSTREAM Ozaki README.

### Monitoring and Diagnostics

| Variable        | Default   | Description                                                       |
|-----------------|-----------|-------------------------------------------------------------------|
| OZAKI_VERBOSE   | 0         | 0=silent, 1=stats at exit, N=print every Nth GEMM call            |
| OZAKI_STAT      | 0         | Track C-matrix (0), A-representation (1), or B-representation (2) |
| OZAKI_DUMP      | 0         | Dump A/B as MHD files at the given call-count (0=off)             |
| OZAKI_EPS       | inf       | Dump A/B when epsilon error exceeds threshold                     |
| OZAKI_RSQ       | 0         | Dump A/B when RSQ drops below threshold (updated after dump)      |
| OZAKI_EXIT      | 1         | Exit on accuracy violation after dump. 0=continue                 |
| OZAKI_PROFILE   | 0         | Profile: 0=off, 1=all, 2=kernel, 3=preprocess A, 4=preprocess B   |

### Benchmark

| Variable   | Default   | Description                                                        |
|------------|-----------|--------------------------------------------------------------------|
| CHECK      | 0         | Validate vs BLAS: 0=off, negative=auto-threshold, positive=custom  |
| NREPEAT    | 3         | Number of GEMM calls (first call is warmup when >1)                |
| EVIL       | 0         | Adversarial exponent-span test (see below)                         |
| OZAKI_DECAY| 0         | Forward-difference decay diagnostic (0=off, nonzero=on)            |

### Adversarial Input (EVIL)

The EVIL variable initializes A and B with controlled exponent
structure for stress-testing accuracy and adaptive slice reduction.
The magnitude sets the exponent span in bits; the sign selects
the distribution:

  EVIL=N  (N>0)   Per-column.  Column j of A is scaled by
                  2^(N\*j/(ncols-1)), column j of B by the
                  inverse.  Product A\*B is well-conditioned.
                  Uniform exponents within each column.

  EVIL=-N (N>0)   Per-element.  Each element gets a pseudorandom
                  exponent in [0,N] via coprime shuffle, with
                  opposite sign for B.  Every row of A spans the
                  full exponent range -- worst case for row-wise
                  alignment and adaptive cutoff.

  EVIL=0          Default shuffle mode (no exponent structure).

The per-column mode (EVIL>0) matches the NVIDIA emulation grading
test (diagonal scaling with D and D^-1).  The per-element mode
(EVIL<0) is adversarial for the adaptive slice-pair reduction:
it forces all slices to be populated in every row.

### Decay Diagnostic (OZAKI_DECAY)

When nonzero, reports the forward-difference decay of int8 slice
buffers along K, M, and N axes (first K-group only, single-
threaded).  Uses libxs_fprint per-axis mode internally.  Output
goes to stderr:

  OZ1[MxNxK] Delta-K: d1=... d2=... ...
  OZ1[MxNxK] Delta-M: d1=... d2=... ...
  OZ1[MxNxK] Delta-N: d1=... d2=... ...

Decaying values indicate exploitable smoothness; growing values
(~2x per order) indicate unstructured data where data-independent
schemes (Ozaki-1, Ozaki-2) are well-matched.

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
./dgemm-wrap.x 256                          # CRT scheme (default)
OZAKI=1 ./dgemm-wrap.x 256                  # mantissa slicing
OZAKI_TRIM=4 OZAKI=1 ./dgemm-wrap.x 256     # drop 4 least significant diagonals
OZAKI=2 OZAKI_GROUPS=4 ./dgemm-wrap.x 4096  # CRT with K-grouping
OZAKI=3 ./dgemm-wrap.x 4096                 # adaptive scheme selection
EVIL=512 ./dgemm-wrap.x 1024                # accuracy grading (wide exponent span)
EVIL=1 OZAKI_PROFILE=1 ./dgemm-wrap.x 1024  # narrow span (shows pair savings)
EVIL=-52 ./dgemm-wrap.x 1024                # per-element span (worst for cutoff)
OZAKI_DECAY=1 OZAKI=1 ./dgemm-wrap.x 256    # forward-difference decay diagnostic
```
