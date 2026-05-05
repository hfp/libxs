# Ozaki Scheme

## High-Precision GEMM via Low-Precision Hardware

LIBXS and LIBXSTREAM

---

## The Transistor Cost of FP64

Unit     | Transistors | Relative
---------|-------------|---------
FP64 FMA | ~150k       | 1×
FP16 FMA | ~15k        | 0.1×
INT8 MAC | ~5k         | 0.03×

Same die area: ~30× more INT8 MACs than FP64 FMAs.

---

## Fairness: Matrix Engines

- **Fair**: tensor core INT8 vs. tensor core FP64
  (both dense matrix engines, same data-flow model)
- **Unfair**: tensor core INT8 vs. scalar FP64 ALU
  (scalar unit: register-file and scheduling overhead)

Ozaki targets matrix engines: VNNI/AMX on CPU, DPAS/tensor cores on GPU.

---

## Hardware Targets

Target        | Instruction | Operands | Status
--------------|-------------|----------|--------
AVX-512 VNNI  | VPDPBUSD    | u8 × s8  | default
AVX-512 INT8  | VPDPBUUD    | u8 × u8  | auto
Intel Xe DPAS | dpas.8×8    | u8/s8    | default
NVIDIA        | dp4a        | u8/s8    | default

VNNI auto-dispatched on CPU. NVIDIA: dp4a has headroom vs. peak; MMA deferred.

---

## What Is Ozaki?

Replaces BLAS GEMM with low-precision arithmetic.

- **Scheme 1**: Mantissa slicing (int8) — quadratic in slices
- **Scheme 2**: Chinese Remainder Theorem (CRT) — linear in primes
- **Adaptive**: Automatic selection based on data characteristics

Built-in accuracy tracking tells if results are trustworthy.

---

## How It Works

```text
Application calls DGEMM/SGEMM/ZGEMM/CGEMM
            │
            ▼
    libwrap.so intercepts the call (LD_PRELOAD)
            │
            ▼
    Decompose FP values into int8 slices or u8 residues
            │
            ▼
    Low-precision dot products (VNNI on CPU, DPAS on GPU)
            │
            ▼
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

DGEMM/SGEMM/ZGEMM/CGEMM calls are intercepted automatically.

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

- **SLURM**: `srun -n 4 --export=ALL,LD_PRELOAD=/path/to/libwrap.so ./app`
- **OpenMPI**: `mpirun -np 4 -x LD_PRELOAD=/path/to/libwrap.so ./app`
- **Intel MPI**: `mpiexec -n 4 -genv LD_PRELOAD /path/to/libwrap.so ./app`

---

## Validate and Check Accuracy

Before production, enable statistics and set a quality threshold:

```bash
export OZAKI_VERBOSE=1      # print summary at exit
export OZAKI_RSQ=0.95       # dump matrices if R² drops below 0.95
```

Run your workload and check the exit summary.

```text
DGEMM[3|410814]: linf=7.81597e-14 linf_rel=2.23531e-15 l2_rel=6.0263e-13 eps=5.20675e-16 rsq=1
```

In brackets are the call-count followed by the PID.

---

## Reading the Statistics


| Metric   | What it tells you       | Healthy value |
|----------|-------------------------|---------------|
| linf     | Max abs. error          | ≈0            |
| linf_rel | Max rel. error          | <1e-10 (FP64) |
| l2_rel   | RMS rel. error          | <1e-12 (FP64) |
| eps      | Frobenius               | <1e-10 (FP64) |
| rsq      | R² (best single metric) | 1.0 = perfect |

R² is the important number. Investigate if below 0.99.

---

## Monitoring Jobs

For HPC jobs that run hours or days, track accuracy over time:

```bash
export OZAKI_VERBOSE=1000   # print stats every 1000th GEMM
export OZAKI_RSQ=0.9        # auto-dump problematic matrices
export OZAKI_EXIT=0         # keep running after threshold violation
```

Watch for R² degradation — it indicates ill-conditioned matrices
or insufficient decomposition depth.

---

## Debugging Accuracy

When `OZAKI_RSQ` or `OZAKI_EPS` thresholds are exceeded,
A and B matrices are dumped as MHD or PNG files (viewers available).

Reproduce the problem offline:

```bash
./dgemm-wrap.x gemm-292284-0-500-a.mhd gemm-292284-0-500-b.mhd
```

- Dump GEMM call by counter using `OZAKI_DUMP`.
- Try to increase the accuracy; see [Details](https://libxs.readthedocs.io/samples/libxs_ozaki/).

---

## Scheme Selection

| Setting | Cost model              | Best for                              |
|---------|-------------------------|---------------------------------------|
| OZAKI=2 | P integer GEMMs (fixed) | General use, large matrices (default) |
| OZAKI=1 | S·(S+1)/2 integer GEMMs | Narrow exponent spans                 |
| OZAKI=3 | Auto-selects 1 or 2     | Repeated calls, unknown data          |

P = number of primes (default 16 for FP64).
S = number of slices (default 8 for FP64).

Auto-selection only on GPU (Ozaki-2 on CPU).

---

## Tuning: Accuracy vs Speed

```bash
export OZAKI_TRIM=7         # drop seven precision levels (faster)
export OZAKI_THRESHOLD=0    # apply Ozaki to ALL GEMMs (default: 12)
export OZAKI_N=12           # more slices/primes (more accurate)
```

The `OZAKI_THRESHOLD` controls minimum arithmetic intensity.
GEMMs below the threshold fall through to the original BLAS.

---

## GPU Offload (LIBXSTREAM)

When built with LIBXSTREAM, a GPU path is available:

```bash
export OZAKI_OCL=1          # enable GPU offload
export OZAKI_OCL=0          # CPU only (default)

export OZAKI=1              # Adaptive cut-off
export OZAKI=2              # CRT scheme (default)
export OZAKI=3              # Auto-select scheme
```

Auto-selection uses `OZAKI_CACHE` as a bonus.

---

## GPU Offload Cache

Avoid to re-transfer the A-matrix or the B-matrix.

```bash
export OZAKI_CACHE=1        # A-matrix
export OZAKI_CACHE=2        # B-matrix
export OZAKI_CACHE=3        # A and B
```

The content is finger-printed,
but can be still wrong!

---

## Profiling

```bash
export OZAKI_PROFILE=1      # all phases (preprocessing + kernel)
export OZAKI_PROFILE=2      # kernel only (dot/matrix products)
export OZAKI_PROFILE=3      # preprocessing only
```

At program exit:

```text
OZAKI PROF: 850 DP-GFLOPS/s (17.0 INT8-TOPS/s, 20×)
```

Reports effective GFLOPS/s and derived INT8 throughput.
Same histogram for CPU and GPU.

---

## Complex GEMM

Complex GEMM is mapped to real GEMM internally.
The real GEMM is wrapped automatically.

```bash
export OZAKI_COMPLEX=0      # BLAS as linked
export OZAKI_COMPLEX=1      # CPU Ozaki
export OZAKI_COMPLEX=2      # GPU+fallback (default)
```

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

1. **Build**: `cd libxs/samples/ozaki && make GNU=1 -j`
2. **Deploy**: `mpirun -np N env LD_PRELOAD=./libwrap.so ./app`
3. **Validate**: `OZAKI_VERBOSE=1 OZAKI_RSQ=0.95`
4. **Monitor**: `OZAKI_VERBOSE=1000 OZAKI_EXIT=0`
5. **Debug**: `./dgemm-wrap.x dumped-a.mhd dumped-b.mhd`
6. **Tune**: `OZAKI=1 OZAKI_TRIM=7`

---

## The Quadratic Cost — Here or There?

Silicon, Ozaki-1, and Ozaki-2 all scale ~w² with precision width.
Ozaki shifts this cost from **transistors to algorithm** —
the ~30× unit cost margin (INT8 vs. FP64) absorbs it.

Only GEMM (O(n³) compute on O(n²) data) can pay this cost.
Reductions, triangular solves, element-wise ops — still need native FP64.

---

## Hardware Design Implications

Ozaki changes the hardware trade-off:

- **AI-dominated**: minimal FP64, strong LP — viable for HPC GEMM via reconstruction
- **Balanced**: moderate FP64 + LP — how about CPU?
- **HPC-dominated**: wide FP64 — solver-heavy, etc.

FP64 stagnation already due to AI dominance.

---

## When LP Wins & When Not

- Favorable: compute-bound GEMM, LP:DP absorbs overhead (16–36×)
- Unfavorable: small matrices, bandwidth-bound, low LP:DP ratio
- Structural: only GEMM — nothing else has the compute density

---

## Questions?

- Hans Pabst [@ intel.com](mailto:hans.pabst@intel.com)
- [https://github.com/hfp/libxs](https://github.com/hfp/libxs)
- [https://github.com/hfp/libxstream](https://github.com/hfp/libxstream)
