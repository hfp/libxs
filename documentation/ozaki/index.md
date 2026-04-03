# Ozaki Scheme
## High-Precision GEMM via Low-Precision Hardware

LIBXSTREAM and LIBXS

---

## What is Ozaki?

High-precision GEMM using low-precision arithmetic

Two schemes:
- **Scheme 1**: Mantissa slicing (int8) — quadratic in slices
- **Scheme 2**: Chinese Remainder Theorem — linear in primes

Intercepts BLAS GEMM calls transparently via `LD_PRELOAD`

---

## Quick Architecture

```
Application calls DGEMM/SGEMM or ZGEMM/CGEMM
            |
            v
    libwrap.so intercepts
            |
            v
    Decompose to int8 slices/residues
            |
            v
    Low-precision dot products
    (AVX-512 VNNI / GPU DPAS)
            |
            v
    Reconstruct high-precision result
```

---

## Getting Started

```bash
cd $HOME
git clone https://github.com/hfp/libxstream.git
cd libxstream && make -j $(nproc)

cd $HOME
git clone https://github.com/hfp/libxs.git
cd libxs && make -j $(nproc)

cd samples/ozaki && make -j $(nproc)
```

Produces `libwrap.so` (LD_PRELOAD), `libwrap.a` (--wrap),
and test drivers `dgemm-wrap.x`, `sgemm-wrap.x`, `zgemm-wrap.x`, `cgemm-wrap.x`

---

## First Test Run

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

## Deploy: LD_PRELOAD

```bash
export LD_PRELOAD=/path/to/libwrap.so
./your_application
```

All DGEMM/SGEMM/ZGEMM/CGEMM calls intercepted automatically

**No recompilation needed!**

---

## Critical: MPI + LD_PRELOAD

### Wrong (may not work)

```bash
LD_PRELOAD=/path/to/libwrap.so mpirun -np 4 ./app
```

```bash
export LD_PRELOAD=/path/to/libwrap.so
mpirun -np 4 ./app
```

Only `mpirun` gets wrapper, not the actual MPI ranks!

---

## MPI: Correct Usage

```bash
mpirun -np 4 env LD_PRELOAD=/path/to/libwrap.so ./app
```

**OpenMPI**: `mpirun -np 4 -x LD_PRELOAD=/path/to/libwrap.so ./app`

**Intel MPI**: `mpiexec -n 4 -genv LD_PRELOAD /path/to/libwrap.so ./app`

**SLURM**: `srun -n 4 --export=ALL,LD_PRELOAD=/path/to/libwrap.so ./app`

---

## Scheme 1: Mantissa Slicing

Decomposes IEEE-754 mantissa into 7-bit int8 slices

```
fp64 (52 bits) -> 8 slices x 7 bits (default)
fp32 (23 bits) -> 4 slices x 7 bits (default)
```

S*(S+1)/2 pairwise int8 dot products (with default flags)

Uses AVX-512 VNNI or AVX-VNNI-INT8 when available

---

## Scheme 2: CRT

Each element reduced modulo small coprimes (<= 128)

```
fp64 -> 19 modular int8 residues (default, max 20)
fp32 -> 10 modular int8 residues (default, max 12)
```

**Linear cost** in number of primes (vs quadratic for slices)

Reconstruct via Garner's algorithm + Horner evaluation

---

## Complex GEMM (ZGEMM/CGEMM)

3M (Karatsuba) method — 3 real GEMMs instead of 4:

```
P1 = Re(A) * Re(B)
P2 = Im(A) * Im(B)
P3 = (Re+Im)(A) * (Re+Im)(B)

Re(C) = P1 - P2,  Im(C) = P3 - P1 - P2
```

Each real GEMM goes through Ozaki wrapper

---

## GPU Offload (LIBXSTREAM)

If built with LIBXSTREAM (OpenCL support):

```bash
export OZAKI_OCL=1          # Enable GPU (default)
export OZAKI_OCL=0          # Force CPU fallback
export OZAKI_GROUPS=4       # K-grouping for CRT (large matrices)
```

Auto-detects Intel XMX hardware (DPAS). Falls back to CPU if unavailable.

---

## Tuning: Accuracy vs Performance

```bash
export OZAKI=2              # CRT (vs default Scheme 1)
export OZAKI_TRIM=4         # Drop 4 levels (~28 bits, faster)
export OZAKI_THRESHOLD=0    # Apply Ozaki to all GEMMs (default: 12)
```

`OZAKI_THRESHOLD` controls minimum arithmetic intensity —
GEMMs below the threshold fall through to original BLAS

---

## Workload Validation: Setup

Always validate before production deployment:

```bash
export OZAKI_VERBOSE=1      # Summary statistics at exit
export OZAKI_RSQ=0.95       # Dump matrices if R^2 < threshold
```

`OZAKI_VERBOSE=N` prints every N-th GEMM for monitoring long runs

`OZAKI_EXIT=0` continues execution after threshold violation (default: exit)

---

## Workload Validation: Statistics Output

```
GEMM: ncalls=90620954 linf=0 linf_rel=0 l2_rel=0 eps=0.000000 rsq=1.000000
```

| Metric | Meaning | Good value |
|---|---|---|
| `ncalls` | Total intercepted GEMM calls | - |
| `linf` | Max absolute error | ~0 |
| `linf_rel` | Max relative error | < 1e-10 (fp64) |
| `l2_rel` | RMS relative error | < 1e-12 (fp64) |
| `eps` | Normalized error (Frobenius) | < 1e-10 (fp64) |
| `rsq` | Coefficient of determination | ~1.0 (perfect) |

**`rsq`** is the single best metric — values < 0.99 indicate problems

---

## Workload Validation: Monitoring

For long-running HPC jobs, track accuracy drift:

```bash
export OZAKI_VERBOSE=1000   # Print every 1000th GEMM
export OZAKI_RSQ=0.9        # Auto-dump problematic matrices
export OZAKI_EXIT=0         # Don't abort, keep collecting
```

Watch for `rsq` degradation over time — indicates
ill-conditioned matrices or insufficient decomposition components

---

## Debug: Matrix Dumps

When `OZAKI_RSQ` or `OZAKI_EPS` thresholds are exceeded:

Creates `gemm_<slurm-rank>-a.mhd` and `gemm_<slurm-rank>-b.mhd`

Reproduce offline:
```bash
./dgemm-wrap.x gemm-292284-0-a.mhd gemm-292284-0-b.mhd
```

Increase accuracy: `OZAKI_N=16` (more slices) or `OZAKI=2` (CRT)

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
export OZAKI_VERBOSE=1000   # Monitor every 1000th GEMM
export OZAKI_RSQ=0.9        # Dump if wrong

srun --export=ALL ./yourapp.x workload.inp
```

---

## Troubleshooting

**Segfault?** Application may statically link BLAS — use `--wrap` instead:
```bash
gcc -o app app.o -lwrap \
  -Wl,--wrap=dgemm_ -Wl,--wrap=sgemm_ \
  -llapack -lblas
```

**No Ozaki output?** Check wrapper is loaded:
```bash
ldd ./app | grep libwrap
```

---

## Summary

1. Build: `cd libxs/samples/ozaki && make -j`
2. Deploy: `mpirun -np N env LD_PRELOAD=./libwrap.so ./app`
3. Validate: `export OZAKI_VERBOSE=1 OZAKI_RSQ=0.95`
4. Monitor: `export OZAKI_VERBOSE=1000 OZAKI_EXIT=0`
5. Debug: `./dgemm-wrap.x dumped-a.mhd dumped-b.mhd`

---

## Questions?

- Hans Pabst: hans.pabst @ intel.com
- https://github.com/hfp/libxs
- https://github.com/hfp/libxstream

Thank you!
