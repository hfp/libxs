#!/usr/bin/env python3
"""Validate paircnt_dd output against Corrfunc.theory.DD.

Reads the same text catalog, runs Corrfunc with matching bin parameters,
and compares pair counts. Requires: pip install corrfunc numpy

Usage:
  python3 paircnt_validate.py catalog.dat --rmin 0.1 --rmax 25.0 --nbins 20
"""
import argparse
import subprocess
import sys

import numpy as np


def read_text_catalog(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            x, y, z = line.split()[:3]
            pts.append((float(x), float(y), float(z)))
    pts = np.array(pts, dtype=np.float64)
    return pts[:, 0], pts[:, 1], pts[:, 2]


def run_corrfunc(x, y, z, rmin, rmax, nbins, boxsize, periodic):
    from Corrfunc.theory.DD import DD
    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    binfile = edges
    results = DD(1, 1, binfile, x, y, z,
                 periodic=periodic, boxsize=boxsize, verbose=False)
    return results["npairs"], edges


def run_paircnt_dd(catalog, rmin, rmax, nbins, boxsize, exe="./paircnt_dd"):
    cmd = [exe, catalog, "--rmin", str(rmin), "--rmax", str(rmax),
           "--nbins", str(nbins)]
    if boxsize > 0:
        cmd += ["--boxsize", str(boxsize)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"paircnt_dd failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)
    counts = []
    for line in proc.stdout.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            counts.append(int(parts[2]))
    return np.array(counts, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Validate paircnt_dd against Corrfunc")
    parser.add_argument("catalog")
    parser.add_argument("--rmin", type=float, default=0.1)
    parser.add_argument("--rmax", type=float, default=25.0)
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--boxsize", type=float, default=0.0)
    parser.add_argument("--exe", default="./paircnt_dd")
    args = parser.parse_args()

    periodic = args.boxsize > 0

    x, y, z = read_text_catalog(args.catalog)
    print(f"catalog: {len(x)} points", file=sys.stderr)

    corrfunc_counts, edges = run_corrfunc(
        x, y, z, args.rmin, args.rmax, args.nbins, args.boxsize, periodic)

    our_counts = run_paircnt_dd(
        args.catalog, args.rmin, args.rmax, args.nbins, args.boxsize,
        args.exe)

    print(f"{'bin':>4s} {'rmin':>12s} {'rmax':>12s} "
          f"{'corrfunc':>12s} {'ours':>12s} {'diff':>8s}")
    total_cf = 0
    total_ours = 0
    mismatches = 0
    for i in range(args.nbins):
        cf = int(corrfunc_counts[i])
        ours = int(our_counts[i]) if i < len(our_counts) else 0
        diff = ours - cf
        total_cf += cf
        total_ours += ours
        flag = " *" if diff != 0 else ""
        if diff != 0:
            mismatches += 1
        print(f"{i:4d} {edges[i]:12.6e} {edges[i+1]:12.6e} "
              f"{cf:12d} {ours:12d} {diff:8d}{flag}")
    print(f"{'total':>4s} {'':>12s} {'':>12s} "
          f"{total_cf:12d} {total_ours:12d} {total_ours - total_cf:8d}")

    if mismatches == 0:
        print("\nPASS: all bins match exactly", file=sys.stderr)
    else:
        print(f"\n{mismatches}/{args.nbins} bins differ", file=sys.stderr)


if __name__ == "__main__":
    main()
