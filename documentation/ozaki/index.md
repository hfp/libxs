# Ozaki Scheme
## High-Precision GEMM via Low-Precision Hardware

LIBXS and LIBXSTREAM

---

## What Is Ozaki?

Replaces standard BLAS GEMM with low-precision arithmetic --
transparently, via LD_PRELOAD. No source changes needed.

- **Scheme 1**: Mantissa slicing (int8) -- quadratic in slices
- **Scheme 2**: Chinese Remainder Theorem (CRT) -- linear in primes
- Intercepts DGEMM, SGEMM, ZGEMM, and CGEMM

Built-in accuracy tracking tells you whether results are trustworthy.

---

## How It Works

```
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

```
Ozaki-Scheme Low-Precision GEMM
GEMM: NN M=256 N=256 K=256
ozaki_oz1: nslices=8 block=16x16x16 vnni=1
Time: 0.123 seconds
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

### Wrong (wrapper only loads in the launcher process)

```bash
LD_PRELOAD=libwrap.so mpirun -np 4 ./app
```

### Correct (wrapper propagated to each rank)

```bash
mpirun -np 4 env LD_PRELOAD=/path/to/libwrap.so ./app
```

**OpenMPI**: `mpirun -np 4 -x LD_PRELOAD=/path/to/libwrap.so ./app`

**Intel MPI**: `mpiexec -n 4 -genv LD_PRELOAD /path/to/libwrap.so ./app`

**SLURM**: `srun -n 4 --export=ALL,LD_PRELOAD=/path/to/libwrap.so ./app`

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

```
GEMM: ncalls=90620954 linf=0 linf_rel=0 l2_rel=0 eps=0.000000 rsq=1.000000
```

| Metric     | What it tells you              | Healthy value                |
|------------|--------------------------------|------------------------------|
| ncalls     | Total intercepted GEMMs        | -                            |
| linf       | Max absolute error             | ~0                           |
| linf_rel   | Max relative error             | < 1e-10 (fp64)               |
| l2_rel     | RMS relative error             | < 1e-12 (fp64)               |
| eps        | Normalized Frobenius error     | < 1e-10 (fp64)               |
| rsq        | R-squared (best single metric) | > 0.99 = good, 1.0 = perfect |

The rsq value is the single most important number.
Values below 0.99 need investigation.

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

When OZAKI_RSQ or OZAKI_EPS thresholds are exceeded, the wrapper
dumps the A and B matrices as MHD files automatically.

Reproduce the problem offline:

```bash
./dgemm-wrap.x gemm-292284-0-500-a.mhd gemm-292284-0-500-b.mhd
```

Try increasing accuracy:

- More slices: `OZAKI_N=12` (Scheme 1, default 8)
- Switch scheme: `OZAKI=2` (CRT, often better for large K)
- More primes: `OZAKI_N=20` (Scheme 2, default 16)

---

## Scheme Selection

| Setting   | Cost model            | Best for                      |
|-----------|-----------------------|-------------------------------|
| OZAKI=1   | S*(S+1)/2 dot prods   | Smaller matrices, high acc.   |
| OZAKI=2   | P dot products        | Large matrices, large K       |

S = number of slices (default 8 for fp64).
P = number of primes (default 16 for fp64).

```bash
export OZAKI=1              # Scheme 1 (default)
export OZAKI=2              # Scheme 2 (CRT)
```

---

## Tuning: Accuracy vs Speed

```bash
export OZAKI_TRIM=4         # drop 4 precision levels (faster, less accurate)
export OZAKI_THRESHOLD=0    # apply Ozaki to ALL GEMMs (default: 12)
export OZAKI_N=12           # more slices/primes (slower, more accurate)
```

The OZAKI_THRESHOLD controls minimum arithmetic intensity.
GEMMs below the threshold fall through to the original BLAS unchanged.

---

## GPU Offload (LIBXSTREAM)

When built with LIBXSTREAM (OpenCL), a GPU path is available:

```bash
export OZAKI_OCL=1          # enable GPU offload
export OZAKI_OCL=0          # CPU only (default)
```

Auto-detects Intel XMX hardware (DPAS instructions).
Falls back to CPU transparently if no GPU is available.

For large matrices with Scheme 2, K-grouping helps significantly:

```bash
export OZAKI=2 OZAKI_GROUPS=4
```

---

## Profiling

```bash
export OZAKI_PROFILE=1      # all phases (preprocessing + kernel)
export OZAKI_PROFILE=2      # kernel only (dot products)
export OZAKI_PROFILE=3      # preprocessing only
```

At program exit:

```
OZAKI PROF: 850 DP-GFLOPS/s (17.0 INT8-TOPS/s, 20x)
```

Reports effective GFLOPS/s and derived INT8 throughput.
Works for both CPU and GPU paths (same histogram).

---

## Complex GEMM (ZGEMM/CGEMM)

Complex GEMM is mapped to real GEMM internally.
The real GEMM goes through the Ozaki wrapper automatically.

Controlled via OZAKI_COMPLEX (default: follows OZAKI setting).

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
export OZAKI=2              # CRT scheme
export OZAKI_VERBOSE=1000   # monitor every 1000th GEMM
export OZAKI_RSQ=0.9        # dump if accuracy drops

srun --export=ALL ./yourapp.x workload.inp
```

---

## Troubleshooting

**Segfault on startup?**
The application may statically link BLAS. Use --wrap instead:

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
6. Tune: `export OZAKI_TRIM=4` or `export OZAKI=2`

---

## Questions?

- Hans Pabst: hans.pabst @ intel.com
- LIBXS: https://github.com/hfp/libxs
- LIBXSTREAM: https://github.com/hfp/libxstream
