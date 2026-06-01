#!/usr/bin/env python3
"""3D-to-2D stratification sample using libxs through ctypes."""

import argparse
import ctypes
import ctypes.util
import csv
import math
import os
import sys
import time
from pathlib import Path


def local_library_candidates():
    sample_dir = Path(__file__).resolve().parent
    root_dir = sample_dir.parents[1]
    names = ["libxs.so", "libxs.dylib", "xs.dll"]
    for name in names:
        yield root_dir / "lib" / name
    found = ctypes.util.find_library("xs")
    if found:
        yield found


def load_libxs(path):
    candidates = []
    if path:
        candidates.append(path)
    env_path = os.environ.get("LIBXS_LIBRARY")
    if env_path:
        candidates.append(env_path)
    candidates.extend(local_library_candidates())
    last_error = None
    for candidate in candidates:
        try:
            libxs = ctypes.CDLL(str(candidate))
            configure_libxs(libxs)
            return libxs, str(candidate)
        except OSError as error:
            last_error = error
    if last_error is not None:
        raise RuntimeError("unable to load libxs shared library: %s" % last_error)
    raise RuntimeError("unable to locate libxs shared library")


def configure_libxs(libxs):
    uint_ptr = ctypes.POINTER(ctypes.c_uint)
    signature = [uint_ptr, ctypes.c_int, uint_ptr, ctypes.c_int]
    bits_signature = [uint_ptr, ctypes.c_int, ctypes.c_int,
                      uint_ptr, ctypes.c_int, ctypes.c_int]
    rank_signature = [uint_ptr, ctypes.c_int, ctypes.c_int]
    libxs.libxs_stratify_hilbert.argtypes = signature
    libxs.libxs_stratify_hilbert.restype = ctypes.c_int
    libxs.libxs_stratify_morton.argtypes = signature
    libxs.libxs_stratify_morton.restype = ctypes.c_int
    libxs._libxs_stratify_bits_available = False
    try:
        libxs.libxs_stratify_hilbert_bits.argtypes = bits_signature
        libxs.libxs_stratify_hilbert_bits.restype = ctypes.c_int
        libxs.libxs_stratify_morton_bits.argtypes = bits_signature
        libxs.libxs_stratify_morton_bits.restype = ctypes.c_int
        libxs.libxs_hilbert_bits.argtypes = rank_signature
        libxs.libxs_hilbert_bits.restype = ctypes.c_uint64
        libxs.libxs_morton_bits.argtypes = rank_signature
        libxs.libxs_morton_bits.restype = ctypes.c_uint64
        libxs._libxs_stratify_bits_available = True
    except AttributeError:
        pass


def shower_volume(depth, height, width):
    volume = [0.0] * (depth * height * width)
    center_x = 0.45 * (width - 1)
    center_y = 0.55 * (height - 1)
    center_z = 0.35 * (depth - 1)
    sigma_x = max(width / 5.5, 1.0)
    sigma_y = max(height / 6.0, 1.0)
    sigma_z = max(depth / 3.5, 1.0)
    tail_scale = max(depth / 2.0, 1.0)
    for z_coord in range(depth):
        longitudinal = math.exp(-z_coord / tail_scale)
        for y_coord in range(height):
            dy = (y_coord - center_y) / sigma_y
            for x_coord in range(width):
                dx = (x_coord - center_x) / sigma_x
                dz = (z_coord - center_z) / sigma_z
                core = math.exp(-0.5 * (dx * dx + dy * dy + dz * dz))
                ripple = 1.0 + 0.08 * math.sin(0.7 * x_coord + 1.3 * y_coord + z_coord)
                index = (z_coord * height + y_coord) * width + x_coord
                volume[index] = core * longitudinal * ripple
    return volume


def require_h5py():
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required for HDF5 input or output") from error
    return h5py


def require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for HDF5 input or output") from error
    return np


def select_hdf5_volume(dataset, layout, event, channel, reshape):
    if layout == "dhw":
        if dataset.ndim != 3:
            raise ValueError("layout dhw requires a 3D dataset")
        array = dataset[...]
    elif layout == "flat":
        if dataset.ndim != 2:
            raise ValueError("layout flat requires a 2D dataset")
        if reshape is None:
            raise ValueError("layout flat requires --hdf5-reshape D H W")
        array = dataset[event, :]
    elif layout == "ndhw":
        if dataset.ndim != 4:
            raise ValueError("layout ndhw requires a 4D dataset")
        array = dataset[event, :, :, :]
    elif layout == "ncdhw":
        if dataset.ndim != 5:
            raise ValueError("layout ncdhw requires a 5D dataset")
        array = dataset[event, channel, :, :, :]
    elif layout == "ndhwc":
        if dataset.ndim != 5:
            raise ValueError("layout ndhwc requires a 5D dataset")
        array = dataset[event, :, :, :, channel]
    else:
        if dataset.ndim == 3:
            array = dataset[...]
        elif dataset.ndim == 4:
            array = dataset[event, :, :, :]
        elif dataset.ndim == 5 and dataset.shape[1] <= 4:
            array = dataset[event, channel, :, :, :]
        elif dataset.ndim == 5:
            array = dataset[event, :, :, :, channel]
        elif dataset.ndim == 2 and reshape is not None:
            array = dataset[event, :]
        else:
            raise ValueError("unsupported HDF5 dataset rank: %d" % dataset.ndim)
    if reshape is not None:
        elements = reshape[0] * reshape[1] * reshape[2]
        if array.size != elements:
            raise ValueError("selected HDF5 volume has %d values, expected %d" %
                             (array.size, elements))
        array = array.reshape(reshape)
    return array


def load_hdf5_volume(path, dataset_name, layout, event, channel, reshape):
    h5py = require_h5py()
    with h5py.File(path, "r") as in_file:
        if dataset_name not in in_file:
            raise KeyError("dataset '%s' not found in %s" % (dataset_name, path))
        dataset = in_file[dataset_name]
        if event < 0 or (dataset.ndim in (2, 4, 5) and event >= dataset.shape[0]):
            raise IndexError("event index %d is outside dataset shape %s" %
                             (event, dataset.shape))
        if channel < 0:
            raise IndexError("channel index must be non-negative")
        if layout in ("ncdhw", "auto") and dataset.ndim == 5 and dataset.shape[1] <= 4:
            if channel >= dataset.shape[1]:
                raise IndexError("channel index %d is outside dataset shape %s" %
                                 (channel, dataset.shape))
        if layout == "ndhwc" and dataset.ndim == 5 and channel >= dataset.shape[4]:
            raise IndexError("channel index %d is outside dataset shape %s" %
                             (channel, dataset.shape))
        array = select_hdf5_volume(dataset, layout, event, channel, reshape)
    if array.ndim != 3:
        raise ValueError("selected HDF5 volume has shape %s, expected 3D" %
                         (array.shape,))
    depth, height, width = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    volume = [float(value) for value in array.reshape(-1)]
    return volume, depth, height, width


def compact_sheet_shape(count):
    sheet_height = int(math.floor(math.sqrt(float(count))))
    while sheet_height > 1 and count % sheet_height != 0:
        sheet_height -= 1
    sheet_width = (count + sheet_height - 1) // sheet_height
    return sheet_height, sheet_width


def stratify(libxs, curve, depth, height, width, volume, frame="compact"):
    src_coords = (ctypes.c_uint * 3)()
    dst_coords = (ctypes.c_uint * 2)()
    src_bits = max(1, (max(depth, height, width) - 1).bit_length())
    dst_bits = (3 * src_bits + 1) // 2
    use_bits = getattr(libxs, "_libxs_stratify_bits_available", False)
    if curve == "hilbert":
        stratify_fn = (libxs.libxs_stratify_hilbert_bits if use_bits
                       else libxs.libxs_stratify_hilbert)
        rank_fn = libxs.libxs_hilbert_bits if use_bits else None
    else:
        stratify_fn = (libxs.libxs_stratify_morton_bits if use_bits
                       else libxs.libxs_stratify_morton)
        rank_fn = libxs.libxs_morton_bits if use_bits else None
    records = []
    ranked_records = []
    max_u = 0
    max_v = 0
    start = time.perf_counter()
    for z_coord in range(depth):
        for y_coord in range(height):
            for x_coord in range(width):
                src_coords[0] = x_coord
                src_coords[1] = y_coord
                src_coords[2] = z_coord
                if use_bits:
                    result = stratify_fn(src_coords, 3, src_bits,
                                         dst_coords, 2, dst_bits)
                else:
                    result = stratify_fn(src_coords, 3, dst_coords, 2)
                if result != 0:
                    raise RuntimeError("libxs_stratify_%s failed" % curve)
                src_index = (z_coord * height + y_coord) * width + x_coord
                if use_bits:
                    rank = int(rank_fn(src_coords, 3, src_bits))
                else:
                    rank = None
                u_coord = int(dst_coords[0])
                v_coord = int(dst_coords[1])
                if u_coord > max_u:
                    max_u = u_coord
                if v_coord > max_v:
                    max_v = v_coord
                record = (src_index, z_coord, y_coord, x_coord,
                          v_coord, u_coord, volume[src_index])
                records.append(record)
                if rank is not None:
                    ranked_records.append((rank, record))
    map_seconds = time.perf_counter() - start
    if frame == "compact" and use_bits:
        sheet_height, sheet_width = compact_sheet_shape(len(records))
        records = []
        for ordinal, ranked_record in enumerate(sorted(ranked_records)):
            record = ranked_record[1]
            v_coord = ordinal // sheet_width
            u_coord = ordinal % sheet_width
            records.append((record[0], record[1], record[2], record[3],
                            v_coord, u_coord, record[6]))
    else:
        sheet_width = max_u + 1
        sheet_height = max_v + 1
    sheet = [0.0] * (sheet_width * sheet_height)
    used = set()
    for record in records:
        dst_index = record[4] * sheet_width + record[5]
        if dst_index in used:
            raise RuntimeError("stratification collision at destination %d" % dst_index)
        used.add(dst_index)
        sheet[dst_index] = record[6]
    return sheet, sheet_height, sheet_width, records, map_seconds


def write_pgm(path, data, height, width):
    max_value = max(data) if data else 0.0
    scale = 255.0 / max_value if max_value > 0.0 else 0.0
    with open(path, "wb") as out_file:
        out_file.write(("P5\n%d %d\n255\n" % (width, height)).encode("ascii"))
        out_file.write(bytearray(int(max(0.0, min(255.0, value * scale))) for value in data))


def write_map_csv(path, depth, height, source_width, sheet_width, records):
    total = depth * height * source_width
    with open(path, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["src", "z", "y", "x", "dst", "v", "u"])
        for record in records:
            dst_index = record[4] * sheet_width + record[5]
            if record[0] < total:
                writer.writerow([record[0], record[1], record[2], record[3],
                                 dst_index, record[4], record[5]])


def write_hdf5(path, data, height, width, records, depth, source_height,
               source_width, curve, frame):
    h5py = require_h5py()
    np = require_numpy()
    sheet = np.asarray(data, dtype=np.float32).reshape((height, width))
    mapping = np.asarray([[record[0], record[1], record[2], record[3],
                           record[4] * width + record[5], record[4], record[5]]
                          for record in records], dtype=np.uint64)
    with h5py.File(path, "w") as out_file:
        out_file.create_dataset("sheet", data=sheet)
        out_file.create_dataset("map", data=mapping)
        out_file["sheet"].attrs["curve"] = curve
        out_file["sheet"].attrs["frame"] = frame
        out_file["sheet"].attrs["source_shape"] = (depth, source_height, source_width)
        out_file["map"].attrs["columns"] = "src,z,y,x,dst,v,u"


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
    parser.add_argument("--out", help="write stratified sheet as an 8-bit PGM image")
    parser.add_argument("--map-csv", help="write source-to-destination map as CSV")
    parser.add_argument("--out-hdf5", help="write stratified sheet and map as HDF5")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    depth, height, width = args.shape
    if depth <= 0 or height <= 0 or width <= 0:
        raise ValueError("shape entries must be positive")
    libxs, lib_path = load_libxs(args.libxs)
    source = "synthetic"
    if args.hdf5:
        volume, depth, height, width = load_hdf5_volume(
            args.hdf5, args.hdf5_dataset, args.hdf5_layout,
            args.hdf5_event, args.hdf5_channel, args.hdf5_reshape)
        source = "%s:%s" % (args.hdf5, args.hdf5_dataset)
    else:
        volume = shower_volume(depth, height, width)
    sheet, sheet_height, sheet_width, records, map_seconds = stratify(
        libxs, args.curve, depth, height, width, volume, args.frame)
    volume_sum = sum(volume)
    sheet_sum = sum(sheet)
    density = float(len(records)) / float(sheet_height * sheet_width)
    print("libxs: %s" % lib_path)
    print("curve: %s" % args.curve)
    print("frame: %s" % args.frame)
    print("input: %s" % source)
    print("source: D=%d H=%d W=%d voxels=%d" %
          (depth, height, width, depth * height * width))
    print("sheet: H=%d W=%d cells=%d density=%.3f" %
          (sheet_height, sheet_width, sheet_height * sheet_width, density))
    print("mapping_time: %.6f s" % map_seconds)
    print("sum: source=%.12g sheet=%.12g absdiff=%.3g" %
          (volume_sum, sheet_sum, abs(volume_sum - sheet_sum)))
    if args.out:
        write_pgm(args.out, sheet, sheet_height, sheet_width)
        print("wrote: %s" % args.out)
    if args.map_csv:
        write_map_csv(args.map_csv, depth, height, width, sheet_width, records)
        print("wrote: %s" % args.map_csv)
    if args.out_hdf5:
        write_hdf5(args.out_hdf5, sheet, sheet_height, sheet_width, records,
                   depth, height, width, args.curve, args.frame)
        print("wrote: %s" % args.out_hdf5)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))