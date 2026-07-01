#!/usr/bin/env python3
"""Report invariants and locality distortion for 3D-to-2D stratification."""

import argparse
import ctypes
import csv
import sys

import stratify_dense3d


LIBXS_DATATYPE_F64 = 0 | (8 << 4)
LIBXS_FPRINT_MAXORDER = 8


class LibxsFprint(ctypes.Structure):
    _fields_ = [
        ("l2", ctypes.c_double * (LIBXS_FPRINT_MAXORDER + 1)),
        ("l1", ctypes.c_double * (LIBXS_FPRINT_MAXORDER + 1)),
        ("linf", ctypes.c_double * (LIBXS_FPRINT_MAXORDER + 1)),
        ("order", ctypes.c_int),
        ("n", ctypes.c_int),
        ("datatype", ctypes.c_int),
    ]


def configure_fprint(libxs):
    size_ptr = ctypes.POINTER(ctypes.c_size_t)
    libxs.libxs_fprint.argtypes = [ctypes.POINTER(LibxsFprint), ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_int, size_ptr,
                                   size_ptr, ctypes.c_int, ctypes.c_int]
    libxs.libxs_fprint.restype = ctypes.c_int
    libxs.libxs_fprint_diff.argtypes = [ctypes.POINTER(LibxsFprint),
                                        ctypes.POINTER(LibxsFprint),
                                        ctypes.POINTER(ctypes.c_double)]
    libxs.libxs_fprint_diff.restype = ctypes.c_double
    libxs.libxs_fprint_decay.argtypes = [ctypes.POINTER(LibxsFprint)]
    libxs.libxs_fprint_decay.restype = ctypes.c_double


def percentile(sorted_values, percent):
    if not sorted_values:
        return 0.0
    index = int(round((percent / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[index])


def summarize(values, prefix, metrics):
    if values:
        sorted_values = sorted(values)
        total = float(sum(sorted_values))
        count = float(len(sorted_values))
        metrics[prefix + ".count"] = len(sorted_values)
        metrics[prefix + ".min"] = float(sorted_values[0])
        metrics[prefix + ".mean"] = total / count
        metrics[prefix + ".p50"] = percentile(sorted_values, 50.0)
        metrics[prefix + ".p90"] = percentile(sorted_values, 90.0)
        metrics[prefix + ".p99"] = percentile(sorted_values, 99.0)
        metrics[prefix + ".max"] = float(sorted_values[-1])
    else:
        metrics[prefix + ".count"] = 0


def source_index(z_coord, y_coord, x_coord, height, width):
    return (z_coord * height + y_coord) * width + x_coord


def build_maps(records, sheet_width):
    src_to_dst = {}
    dst_to_src = {}
    for record in records:
        src = record[0]
        dst = record[4] * sheet_width + record[5]
        src_to_dst[src] = (record[4], record[5])
        dst_to_src[dst] = (record[1], record[2], record[3])
    return src_to_dst, dst_to_src


def fingerprint(libxs, values, shape, order):
    info = LibxsFprint()
    count = len(values)
    data = (ctypes.c_double * count)(*values)
    dims = (ctypes.c_size_t * len(shape))(*shape)
    result = libxs.libxs_fprint(ctypes.byref(info), LIBXS_DATATYPE_F64,
                                data, len(shape), dims, None, order, 0, 0, 0)
    if result != 0:
        raise RuntimeError("libxs_fprint failed")
    return info


def add_fingerprint_metrics(libxs, metrics, volume, reconstructed, sheet,
                            depth, height, width, sheet_height, sheet_width):
    order = 4
    source = fingerprint(libxs, volume, (width, height, depth), order)
    recovered = fingerprint(libxs, reconstructed, (width, height, depth), order)
    sheet_fp = fingerprint(libxs, sheet, (sheet_width, sheet_height), order)
    metrics["fprint.order"] = min(source.order, sheet_fp.order)
    metrics["fprint.source.decay"] = libxs.libxs_fprint_decay(ctypes.byref(source))
    metrics["fprint.reconstructed.decay"] = libxs.libxs_fprint_decay(ctypes.byref(recovered))
    metrics["fprint.sheet.decay"] = libxs.libxs_fprint_decay(ctypes.byref(sheet_fp))
    metrics["fprint.source_reconstructed.diff"] = libxs.libxs_fprint_diff(
        ctypes.byref(source), ctypes.byref(recovered), None)
    metrics["fprint.source_sheet.diff"] = libxs.libxs_fprint_diff(
        ctypes.byref(source), ctypes.byref(sheet_fp), None)
    for index in range(0, min(source.order, sheet_fp.order) + 1):
        metrics["fprint.source.l2.%d" % index] = source.l2[index]
        metrics["fprint.sheet.l2.%d" % index] = sheet_fp.l2[index]


def profile(values, depth, height, width, axis):
    if axis == "z":
        result = [0.0] * depth
    elif axis == "y":
        result = [0.0] * height
    else:
        result = [0.0] * width
    for z_coord in range(depth):
        for y_coord in range(height):
            for x_coord in range(width):
                value = values[source_index(z_coord, y_coord, x_coord, height, width)]
                if axis == "z":
                    result[z_coord] += value
                elif axis == "y":
                    result[y_coord] += value
                else:
                    result[x_coord] += value
    return result


def max_abs_delta(left, right):
    if not left:
        return 0.0
    return max(abs(a - b) for a, b in zip(left, right))


def reconstruct_volume(sheet, records, sheet_width, count):
    result = [0.0] * count
    for record in records:
        dst = record[4] * sheet_width + record[5]
        result[record[0]] = sheet[dst]
    return result


def source_neighbor_distances(depth, height, width, src_to_dst):
    distances = []
    adjacent = 0
    within2 = 0
    within4 = 0
    for z_coord in range(depth):
        for y_coord in range(height):
            for x_coord in range(width):
                src = source_index(z_coord, y_coord, x_coord, height, width)
                v0, u0 = src_to_dst[src]
                neighbors = []
                if x_coord + 1 < width:
                    neighbors.append(source_index(z_coord, y_coord, x_coord + 1, height, width))
                if y_coord + 1 < height:
                    neighbors.append(source_index(z_coord, y_coord + 1, x_coord, height, width))
                if z_coord + 1 < depth:
                    neighbors.append(source_index(z_coord + 1, y_coord, x_coord, height, width))
                for neighbor in neighbors:
                    v1, u1 = src_to_dst[neighbor]
                    distance = abs(v0 - v1) + abs(u0 - u1)
                    distances.append(distance)
                    if distance == 1:
                        adjacent += 1
                    if distance <= 2:
                        within2 += 1
                    if distance <= 4:
                        within4 += 1
    return distances, adjacent, within2, within4


def sheet_neighbor_source_distances(sheet_height, sheet_width, dst_to_src):
    distances = []
    adjacent = 0
    for v_coord in range(sheet_height):
        for u_coord in range(sheet_width):
            dst = v_coord * sheet_width + u_coord
            if dst not in dst_to_src:
                continue
            z0, y0, x0 = dst_to_src[dst]
            neighbors = []
            if u_coord + 1 < sheet_width:
                neighbors.append(v_coord * sheet_width + u_coord + 1)
            if v_coord + 1 < sheet_height:
                neighbors.append((v_coord + 1) * sheet_width + u_coord)
            for neighbor in neighbors:
                if neighbor not in dst_to_src:
                    continue
                z1, y1, x1 = dst_to_src[neighbor]
                distance = abs(z0 - z1) + abs(y0 - y1) + abs(x0 - x1)
                distances.append(distance)
                if distance == 1:
                    adjacent += 1
    return distances, adjacent


def compute_metrics(libxs, curve, frame, depth, height, width, volume):
    sheet, sheet_height, sheet_width, records, map_seconds = stratify_dense3d.stratify(
        libxs, curve, depth, height, width, volume, frame)
    src_to_dst, dst_to_src = build_maps(records, sheet_width)
    reconstructed = reconstruct_volume(sheet, records, sheet_width, len(volume))
    source_sum = sum(volume)
    sheet_sum = sum(sheet)
    occupied = len(records)
    cells = sheet_height * sheet_width
    metrics = {
        "curve": curve,
        "frame": frame,
        "source.depth": depth,
        "source.height": height,
        "source.width": width,
        "source.voxels": len(volume),
        "sheet.height": sheet_height,
        "sheet.width": sheet_width,
        "sheet.cells": cells,
        "sheet.occupied": occupied,
        "sheet.unused": cells - occupied,
        "sheet.density": float(occupied) / float(cells),
        "mapping.seconds": map_seconds,
        "invariant.source_sum": source_sum,
        "invariant.sheet_sum": sheet_sum,
        "invariant.sum_absdiff": abs(source_sum - sheet_sum),
        "invariant.reconstruction_max_absdiff": max_abs_delta(volume, reconstructed),
        "profile.z_max_absdiff": max_abs_delta(
            profile(volume, depth, height, width, "z"),
            profile(reconstructed, depth, height, width, "z")),
        "profile.y_max_absdiff": max_abs_delta(
            profile(volume, depth, height, width, "y"),
            profile(reconstructed, depth, height, width, "y")),
        "profile.x_max_absdiff": max_abs_delta(
            profile(volume, depth, height, width, "x"),
            profile(reconstructed, depth, height, width, "x")),
    }
    add_fingerprint_metrics(libxs, metrics, volume, reconstructed, sheet,
                            depth, height, width, sheet_height, sheet_width)
    source_distances, adjacent, within2, within4 = source_neighbor_distances(
        depth, height, width, src_to_dst)
    summarize(source_distances, "source_neighbors.sheet_manhattan", metrics)
    if source_distances:
        count = float(len(source_distances))
        metrics["source_neighbors.sheet_adjacent_fraction"] = float(adjacent) / count
        metrics["source_neighbors.sheet_within2_fraction"] = float(within2) / count
        metrics["source_neighbors.sheet_within4_fraction"] = float(within4) / count
    sheet_distances, sheet_adjacent = sheet_neighbor_source_distances(
        sheet_height, sheet_width, dst_to_src)
    summarize(sheet_distances, "sheet_neighbors.source_manhattan", metrics)
    if sheet_distances:
        metrics["sheet_neighbors.source_adjacent_fraction"] = (
            float(sheet_adjacent) / float(len(sheet_distances)))
    return metrics


def write_metrics(metrics, path):
    rows = sorted(metrics.items())
    if path:
        with open(path, "w", newline="") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)
    for key, value in rows:
        print("%s,%s" % (key, value))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("D", "H", "W"),
                        default=(8, 16, 16), help="source volume shape")
    parser.add_argument("--curve", choices=("hilbert", "morton"), default="hilbert")
    parser.add_argument("--frame", choices=("compact", "canonical"),
                        default="compact",
                        help="sheet framing policy for finite-bit stratification")
    parser.add_argument("--libxs", help="path to libxs shared library")
    parser.add_argument("--hdf5", help="read source volume from an HDF5 file")
    parser.add_argument("--hdf5-dataset", default="ECAL",
                        help="HDF5 dataset containing source volumes")
    parser.add_argument("--hdf5-layout", choices=("auto", "dhw", "ndhw", "ncdhw", "ndhwc", "flat"),
                        default="auto", help="HDF5 dataset layout")
    parser.add_argument("--hdf5-reshape", type=int, nargs=3, metavar=("D", "H", "W"),
                        help="reshape selected flat HDF5 data to a 3D volume")
    parser.add_argument("--hdf5-event", type=int, default=0,
                        help="event index for batched HDF5 layouts")
    parser.add_argument("--hdf5-channel", type=int, default=0,
                        help="channel index for channelled HDF5 layouts")
    parser.add_argument("--medmnist3d", action="store_true",
                        help="read source volume from MedMNIST3D root/flag lookup")
    parser.add_argument("--medmnist3d-npz", help="read source volume from a MedMNIST3D NPZ file")
    parser.add_argument("--medmnist3d-root", default="~/.medmnist",
                        help="directory containing MedMNIST3D NPZ files")
    parser.add_argument("--medmnist3d-flag", default="organmnist3d",
                        help="MedMNIST3D dataset flag used with --medmnist3d-root")
    parser.add_argument("--medmnist3d-size", type=int, default=28,
                        help="MedMNIST image size used with --medmnist3d-root")
    parser.add_argument("--medmnist3d-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--medmnist3d-index", type=int, default=0)
    parser.add_argument("--csv", help="write metrics as CSV")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    depth, height, width = args.shape
    if args.hdf5 and (args.medmnist3d or args.medmnist3d_npz):
        raise ValueError("choose either --hdf5 or --medmnist3d-npz, not both")
    if args.hdf5:
        volume, depth, height, width = stratify_dense3d.load_hdf5_volume(
            args.hdf5, args.hdf5_dataset, args.hdf5_layout,
            args.hdf5_event, args.hdf5_channel, args.hdf5_reshape)
    elif args.medmnist3d or args.medmnist3d_npz:
        import stratify_medmnist3d
        npz_path = stratify_medmnist3d.resolve_npz_path(
            args.medmnist3d_npz, args.medmnist3d_root,
            args.medmnist3d_flag, args.medmnist3d_size)
        volume, depth, height, width, _ = stratify_medmnist3d.load_medmnist3d_volume(
            npz_path, args.medmnist3d_split, args.medmnist3d_index)
    else:
        if depth <= 0 or height <= 0 or width <= 0:
            raise ValueError("shape entries must be positive")
        volume = stratify_dense3d.shower_volume(depth, height, width)
    libxs, lib_path = stratify_dense3d.load_libxs(args.libxs)
    configure_fprint(libxs)
    metrics = compute_metrics(libxs, args.curve, args.frame,
                              depth, height, width, volume)
    metrics["libxs"] = lib_path
    write_metrics(metrics, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))