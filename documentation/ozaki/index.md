# Ozaki Scheme
## High-Precision GEMM via Low-Precision Hardware

LIBXS and LIBXSTREAM

---

## What Is Ozaki?

Replaces standard BLAS GEMM with low-precision arithmetic --
transparently, via LD_PRELOAD. No source changes needed.

- **Scheme 1**: Mantissa slicing (int8) -- quadratic in slices
- **Scheme 2**: Chinese Remainder Theorem (CRT) -- linear in primes
- **Adaptive**: Automatic selection based on data characteristics
- Intercepts DGEMM, SGEMM, ZGEMM, and CGEMM

Built-in accuracy tracking tells if results are trustworthy.

---

## How It Works

```text
Application calls DGEMM/SGEMM/ZGEMM/CGEMM
            |
            v
    libwrap.so intercepts the call
            |
            v
    Decompose FP values into int8 slices or u8 residues
            |
            v
    Low-precision dot products (VNNI on CPU, DPAS on GPU)
            |
            v
    Reconstruct full-precision result
```

---

## Build

```bash
git clone https://github.com/hfp/libxstream.git
git clone https://github.com/hfp/libxs.git

cd libxs/samples/ozaki
make GNU=1 -j $(nproc)
```

Produces `libwrap.so`, `libwrap.a`, and test drivers
(`dgemm-wrap.x`, `sgemm-wrap.x`, `zgemm-wrap.x`, `cgemm-wrap.x`).

---

## Quick Sanity Check

```bash
./dgemm-wrap.x 256
```

```text
dgemm('N', 'N', 256/*m*/, 256/*n*/, 256/*k*/,
  1/*alpha*/, 0x7cf7c8a02010/*a*/, 256/*lda*/,
              0x7cf7c8981010/*b*/, 256/*ldb*/,
   1/*beta*/, 0x7cf7c8900010/*c*/, 256/*ldc*/)
OZAKI GEMM: 27.3 ms (1.2 GFLOPS/s)
BLAS GEMM:  0.2 ms (214.3 GFLOPS/s)
...
```

---

## Deploy via LD_PRELOAD

```bash
LD_PRELOAD=/path/to/libwrap.so ./your_application
```

All DGEMM/SGEMM/ZGEMM/CGEMM calls are intercepted automatically.

No recompilation needed.

---

## MPI: Getting LD_PRELOAD Right

The wrapper must reach every MPI rank, not just mpirun itself.

### Wrong (wrapper only in the launcher process)

```bash
LD_PRELOAD=libwrap.so mpirun -np 4 ./app
```

### Correct (wrapper propagated to each rank)

```bash
mpirun -np 4 env LD_PRELOAD=/path/to/libwrap.so ./app
```

---

## SLURM and MPI

* **SLURM**: `srun -n 4 --export=ALL,LD_PRELOAD=/path/to/libwrap.so ./app`
* **OpenMPI**: `mpirun -np 4 -x LD_PRELOAD=/path/to/libwrap.so ./app`
* **Intel MPI**: `mpiexec -n 4 -genv LD_PRELOAD /path/to/libwrap.so ./app`

---

## Validate: Always Check Accuracy First

Before production, enable statistics and set a quality threshold:

```bash
export OZAKI_VERBOSE=1      # print summary at exit
export OZAKI_RSQ=0.95       # dump matrices if R^2 drops below 0.95
```

Run your workload and check the exit summary.

---

## Reading the Statistics Output

```text
DGEMM[3|410814]: linf=7.81597e-14 linf_rel=2.23531e-15 l2_rel=6.0263e-13 eps=5.20675e-16 rsq=1
```

In brackets are the running call-count followed by the PID.

| Metric     | What it tells you              | Healthy value                |
|------------|--------------------------------|------------------------------|
| linf       | Max absolute error             | ~0                           |
| linf_rel   | Max relative error             | < 1e-10 (fp64)               |
| l2_rel     | RMS relative error             | < 1e-12 (fp64)               |
| eps        | Normalized Frobenius error     | < 1e-10 (fp64)               |
| rsq        | R-squared (best single metric) | > 0.99 = good, 1.0 = perfect |

The RSQ is the important number. Investigate if below 0.99.

---

## Monitoring Long-Running Jobs

For HPC jobs that run hours or days, track accuracy over time:

```bash
export OZAKI_VERBOSE=1000   # print stats every 1000th GEMM
export OZAKI_RSQ=0.9        # auto-dump problematic matrices
export OZAKI_EXIT=0         # keep running after threshold violation
```

Watch for rsq degradation -- it indicates ill-conditioned matrices
or insufficient decomposition depth.

---

## Debugging Accuracy Problems

When `OZAKI_RSQ` or `OZAKI_EPS` thresholds are exceeded,
A and B matrices are dumped as MHD files (viewers available).

Reproduce the problem offline:

```bash
./dgemm-wrap.x gemm-292284-0-500-a.mhd gemm-292284-0-500-b.mhd
```

Try increasing accuracy:

- More primes: `OZAKI_N=20` (Scheme 2, default 16)
- Switch scheme: `OZAKI=1` (mantissa slicing)
- More slices: `OZAKI_N=12` (Scheme 1, default 8)

---

## Scheme Selection

| Setting   | Cost model               | Best for                              |
|-----------|--------------------------|---------------------------------------|
| OZAKI=2   | P integer GEMMs (fixed)  | General use, large matrices (default) |
| OZAKI=1   | S\*(S+1)/2 integer GEMMs | Narrow exponent spans                 |
| OZAKI=3\* | Auto-selects 1 or 2      | Repeated calls, unknown data          |

P = number of primes (default 16 for fp64).  
S = number of slices (default 8 for fp64).

\* Auto-selection only on GPU (Ozaki-2 on CPU).

---

## Tuning: Accuracy vs Speed

```bash
export OZAKI_TRIM=4         # drop four precision levels (faster)
export OZAKI_THRESHOLD=0    # apply Ozaki to ALL GEMMs (default: 12)
export OZAKI_N=12           # more slices/primes (more accurate)
```

The `OZAKI_THRESHOLD` controls minimum arithmetic intensity.
GEMMs below the threshold fall through to the original BLAS unchanged.

---

## GPU Offload (LIBXSTREAM)

When built with LIBXSTREAM (OpenCL), a GPU path is available:

```bash
export OZAKI_OCL=1          # enable GPU offload
export OZAKI_OCL=0          # CPU only (default)

export OZAKI=1              # Adaptive cut-off
export OZAKI=2              # CRT scheme (default)
export OZAKI=3              # Auto-select scheme
```

Auto-selecting the scheme uses `OZAKI_CACHE` as a bonus.

---

## GPU Offload Cache

Avoid to re-transfer the A-matrix or the B-matrix.

```bash
export OZAKI_CACHE=1        # A-matrix
export OZAKI_CACHE=2        # B-matrix
export OZAKI_CACHE=3        # A and B
```

The content is finger-printed but can still cause wrong results!

---

## Profiling

```bash
export OZAKI_PROFILE=1      # all phases (preprocessing + kernel)
export OZAKI_PROFILE=2      # kernel only (dot/matrix products)
export OZAKI_PROFILE=3      # preprocessing only
```

At program exit:

```text
OZAKI PROF: 850 DP-GFLOPS/s (17.0 INT8-TOPS/s, 20x)
```

Reports effective GFLOPS/s and derived INT8 throughput.  
Works for both CPU and GPU paths (same histogram).

---

## Complex GEMM (ZGEMM/CGEMM)

Complex GEMM is mapped to real GEMM internally.  
The real GEMM is wrapped automatically.

Can be controlled via `OZAKI_COMPLEX`.

---

## Complete SLURM Example

```bash
#!/bin/bash
#SBATCH -o %x-%j.txt
#SBATCH -J yourapp
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --ntasks-per-node=16
#SBATCH -N 1

export LD_PRELOAD=$HOME/libxs/samples/ozaki/libwrap.so
export OZAKI=2              # CRT scheme (default)
export OZAKI_VERBOSE=1000   # monitor every 1000th GEMM
export OZAKI_RSQ=0.9        # dump if accuracy drops

srun --export=ALL ./yourapp.x workload.inp
```

---

## Troubleshooting

**Segfault on startup?**
The application may statically link BLAS.
Use --wrap instead:

```bash
gcc -o app app.o -lwrap \
  -Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ \
  -Wl,--wrap=zgemm_ -Wl,--wrap=cgemm_ \
  -llapack -lblas
```

**No Ozaki output at all?**
Verify the wrapper is loaded:

```bash
ldd ./app | grep libwrap
```

---

## Checklist

1. Build: `cd libxs/samples/ozaki && make GNU=1 -j`
2. Deploy: `mpirun -np N env LD_PRELOAD=./libwrap.so ./app`
3. Validate: `export OZAKI_VERBOSE=1 OZAKI_RSQ=0.95`
4. Monitor: `export OZAKI_VERBOSE=1000 OZAKI_EXIT=0`
5. Debug: `./dgemm-wrap.x dumped-a.mhd dumped-b.mhd`
6. Tune: `export OZAKI_TRIM=7` or `export OZAKI=1`

---

## Questions?

- Hans Pabst: hans.pabst @ intel.com
- LIBXS: https://github.com/hfp/libxs
- LIBXSTREAM: https://github.com/hfp/libxstream
