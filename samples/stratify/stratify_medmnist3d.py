#!/usr/bin/env python3
"""Stratify one MedMNIST3D volume from a standardized NPZ dataset."""

import argparse
import csv
import sys
from pathlib import Path

import stratify_dense3d


MEDMNIST3D_FLAGS = (
    "organmnist3d",
    "nodulemnist3d",
    "adrenalmnist3d",
    "fracturemnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
)


def require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for MedMNIST3D NPZ input") from error
    return np


def medmnist_npz_candidates(root, flag, size):
    root_path = Path(root).expanduser()
    if size == 28:
        names = ["%s.npz" % flag, "%s_28.npz" % flag]
    else:
        names = ["%s_%d.npz" % (flag, size), "%s.npz" % flag]
    for name in names:
        yield root_path / name


def resolve_npz_path(path, root, flag, size):
    if path:
        result = Path(path).expanduser()
        if not result.is_file():
            raise FileNotFoundError("MedMNIST3D NPZ file not found: %s" % result)
        return result
    for candidate in medmnist_npz_candidates(root, flag, size):
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in medmnist_npz_candidates(root, flag, size))
    raise FileNotFoundError("MedMNIST3D NPZ file not found; tried: %s" % tried)


def load_medmnist3d_volume(path, split, index):
    np = require_numpy()
    image_key = "%s_images" % split
    label_key = "%s_labels" % split
    with np.load(str(path)) as data:
        if image_key not in data:
            raise KeyError("dataset key '%s' not found in %s" % (image_key, path))
        images = data[image_key]
        labels = data[label_key] if label_key in data else None
        if images.ndim != 4:
            raise ValueError("%s has shape %s, expected N,D,H,W" %
                             (image_key, images.shape))
        if index < 0 or index >= images.shape[0]:
            raise IndexError("index %d outside %s with %d samples" %
                             (index, image_key, images.shape[0]))
        array = images[index]
        label = None
        if labels is not None:
            label_array = labels[index]
            if hasattr(label_array, "reshape"):
                label = [int(value) for value in label_array.reshape(-1)]
            else:
                label = [int(label_array)]
    depth, height, width = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    volume = [float(value) for value in array.reshape(-1)]
    return volume, depth, height, width, label


def write_label_csv(path, source, split, index, label):
    with open(path, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["source", "split", "index", "label"])
        writer.writerow([source, split, index,
                         "" if label is None else " ".join(str(value) for value in label)])


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", help="path to a MedMNIST3D NPZ file")
    parser.add_argument("--root", default="~/.medmnist",
                        help="directory containing MedMNIST NPZ files")
    parser.add_argument("--flag", choices=MEDMNIST3D_FLAGS, default="organmnist3d",
                        help="MedMNIST3D dataset flag used with --root")
    parser.add_argument("--size", type=int, default=28,
                        help="MedMNIST image size used with --root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0, help="sample index within the split")
    parser.add_argument("--curve", choices=("hilbert", "morton"), default="hilbert")
    parser.add_argument("--frame", choices=("compact", "canonical"),
                        default="compact",
                        help="sheet framing policy for finite-bit stratification")
    parser.add_argument("--libxs", help="path to libxs shared library")
    parser.add_argument("--out", help="write stratified sheet as an 8-bit PGM image")
    parser.add_argument("--map-csv", help="write source-to-destination map as CSV")
    parser.add_argument("--label-csv", help="write selected MedMNIST label as CSV")
    parser.add_argument("--out-hdf5", help="write stratified sheet and map as HDF5")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    npz_path = resolve_npz_path(args.npz, args.root, args.flag, args.size)
    volume, depth, height, width, label = load_medmnist3d_volume(
        npz_path, args.split, args.index)
    libxs, lib_path = stratify_dense3d.load_libxs(args.libxs)
    sheet, sheet_height, sheet_width, records, map_seconds = stratify_dense3d.stratify(
        libxs, args.curve, depth, height, width, volume, args.frame)
    volume_sum = sum(volume)
    sheet_sum = sum(sheet)
    density = float(len(records)) / float(sheet_height * sheet_width)
    print("libxs: %s" % lib_path)
    print("curve: %s" % args.curve)
    print("frame: %s" % args.frame)
    print("input: %s:%s[%d]" % (npz_path, args.split, args.index))
    print("label: %s" % ("none" if label is None else " ".join(str(value) for value in label)))
    print("source: D=%d H=%d W=%d voxels=%d" %
          (depth, height, width, depth * height * width))
    print("sheet: H=%d W=%d cells=%d density=%.3f" %
          (sheet_height, sheet_width, sheet_height * sheet_width, density))
    print("mapping_time: %.6f s" % map_seconds)
    print("sum: source=%.12g sheet=%.12g absdiff=%.3g" %
          (volume_sum, sheet_sum, abs(volume_sum - sheet_sum)))
    if args.out:
        stratify_dense3d.write_pgm(args.out, sheet, sheet_height, sheet_width)
        print("wrote: %s" % args.out)
    if args.map_csv:
        stratify_dense3d.write_map_csv(args.map_csv, depth, height, width, sheet_width, records)
        print("wrote: %s" % args.map_csv)
    if args.label_csv:
        write_label_csv(args.label_csv, str(npz_path), args.split, args.index, label)
        print("wrote: %s" % args.label_csv)
    if args.out_hdf5:
        stratify_dense3d.write_hdf5(args.out_hdf5, sheet, sheet_height, sheet_width,
                                   records, depth, height, width, args.curve,
                                   args.frame)
        print("wrote: %s" % args.out_hdf5)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))