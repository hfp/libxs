#!/usr/bin/env python3
"""Generate point catalogs for pair-counting validation.

Outputs plain text (x y z per line) by default, suitable for paircnt_dd.
Can also write Fortran fastfood (.ff) format for Corrfunc compatibility.

Usage:
  python3 paircnt_gendata.py --npoints 10000 --boxsize 420.0 --out catalog.dat
  python3 paircnt_gendata.py --npoints 10000 --boxsize 420.0 --out catalog.ff
  python3 paircnt_gendata.py --npoints 10000 --clustered --out clustered.dat
"""
import argparse
import struct
import sys


def generate_uniform(n, boxsize, seed):
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, boxsize, size=(n, 3))
    except ImportError:
        import random
        random.seed(seed)
        return [[random.uniform(0.0, boxsize) for _ in range(3)]
                for _ in range(n)]


def generate_clustered(n, boxsize, nclusters, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1 * boxsize, 0.9 * boxsize,
                          size=(nclusters, 3))
    sigma = boxsize * 0.05
    pts = []
    per_cluster = n // nclusters
    for c in range(nclusters):
        count = per_cluster if c < nclusters - 1 else n - len(pts)
        cluster_pts = rng.normal(centers[c], sigma, size=(count, 3))
        cluster_pts = np.clip(cluster_pts, 0.0, boxsize)
        pts.append(cluster_pts)
    return np.vstack(pts)


def write_text(pts, path):
    with open(path, "w") as f:
        f.write("# x y z\n")
        for p in pts:
            f.write(f"{p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")


def write_ff(pts, path, boxsize):
    """Write Fortran fastfood binary (float32, matches Corrfunc .ff)."""
    import numpy as np
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    idat = np.array([0, n, 0, 0, 0], dtype=np.int32)
    fdat = np.zeros(9, dtype=np.float32)
    fdat[0] = boxsize
    znow = np.float32(0.0)
    with open(path, "wb") as f:
        rec = idat.tobytes()
        f.write(struct.pack("i", len(rec)))
        f.write(rec)
        f.write(struct.pack("i", len(rec)))
        rec = fdat.tobytes()
        f.write(struct.pack("i", len(rec)))
        f.write(rec)
        f.write(struct.pack("i", len(rec)))
        rec = znow.tobytes()
        f.write(struct.pack("i", len(rec)))
        f.write(rec)
        f.write(struct.pack("i", len(rec)))
        for d in range(3):
            col = np.ascontiguousarray(pts[:, d])
            rec = col.tobytes()
            f.write(struct.pack("i", len(rec)))
            f.write(rec)
            f.write(struct.pack("i", len(rec)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate point catalog for pair counting")
    parser.add_argument("--npoints", type=int, default=10000)
    parser.add_argument("--boxsize", type=float, default=420.0)
    parser.add_argument("--clustered", action="store_true")
    parser.add_argument("--nclusters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="catalog.dat")
    args = parser.parse_args()

    if args.clustered:
        pts = generate_clustered(args.npoints, args.boxsize,
                                 args.nclusters, args.seed)
    else:
        pts = generate_uniform(args.npoints, args.boxsize, args.seed)

    if args.out.endswith(".ff"):
        write_ff(pts, args.out, args.boxsize)
    else:
        write_text(pts, args.out)

    print(f"{args.npoints} points -> {args.out} "
          f"(boxsize={args.boxsize}, "
          f"{'clustered' if args.clustered else 'uniform'}, "
          f"seed={args.seed})", file=sys.stderr)


if __name__ == "__main__":
    main()
