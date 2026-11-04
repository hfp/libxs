# Pack

Measures bit packing (`libxs_pack.h`) at a fixed width (`libxs_pack`) against
a width per group of values (`libxs_vpack`): bits per value, ratio, and time
per value. Every round trip is verified.

## Build

```bash
cd samples/pack
make
```

## Usage

```bash
./pack.x [nvalues [nbits [range [zeros [nrepeat]]]]]
./pack.x FILE.mhd [FILE.mhd ...] [eps]
```

Without a file, the values are synthetic: signed integers of up to `nbits`
bits whose magnitudes are log-uniform over `range` bits per block of 512
values, with `zeros` percent set to zero.

| Positional | Default | Description                                  |
|------------|---------|----------------------------------------------|
| nvalues    | 1048576 | Number of values                             |
| nbits      | 24      | Width of the fixed format (2 to 63)          |
| range      | 12      | Spread of magnitudes in bits                 |
| zeros      | 10      | Percentage of zeros                          |
| nrepeat    | 10      | Timed passes per measurement, fastest counts |

With MHD files (any element type), the values are measured in blocks of 512
and summed over all files. Integer data is packed as it is. Floating-point
data is quantized to multiples of `eps` (values not above `eps` become zero),
and each block gets the width its largest value needs. A file whose MHD
extension holds a REAL(8) eps, an INT32 number of blocks, and an INT32 shape(3)
per block provides eps and the blocks itself, so a dump of the data to analyze
can carry its own blocking.

The output compares the fixed width with the levels of `libxs_vpack`, with
deltas between neighbors, and for floating-point data with a quantization step
of twice `eps`, plus the bits that a width per value would need. With MHD files,
a second table takes a width per group of 4 to 64 values (computed by the
sample) and splits it into the payload and the cost of storing the widths: one
byte per width, 7 bits per width, a base per 8 groups with 3- or 2-bit offsets
(`base+3`, `base+2`), and the entropy of the widths as a bound. Each row gives
bits per value and the ratio.

A third table estimates the bits of other ways to store a group of 8, without
implementing them: a zero mask with only the nonzeros packed, exceptions (the
values above a regular width patched separately), both, the same for deltas,
per block deltas along each axis of the block (shape from the MHD extension),
per block groups of 8 in each order of the block's axes (and the best order per
block), per block sub-boxes with one width each (a k-d tree,
`libxs_kdtree_partition`), and dyadic trees of 4 to 64 values in stream order.
The row "kd-tree rows, exceptions" is a prototype of such a format: every block
as sub-boxes of whole rows (split at the middle of axis 1 or 2) with exceptions,
encoded, decoded, and checked, with a length word per block. `KDLEAF=n` in the
environment lets it split only into leaves of at least `n` values.

The rows "vpack level 1" to "vpack level 3" pack at the levels of
`libxs_vpack`; the rows vpack deltas and step 2*eps at level 0, the default
(`LIBXS_PACK` in the environment). At level 1, `vpack.f` and `libxs_vunpack`
also decode what the other packed. For floating-point input, the rows "vpack
real" pack the values as they are (level 0): lossless, keeping 52 or 32 leading
bits, and with an absolute error of at most the largest power of two up to
`eps/2`; each checks its error, and its ratio is against 64 bits per value.

With `REPEAT=n` in the environment (default 1), a measurement takes `n` timed passes
and the fastest counts, as with `nrepeat` for synthetic values. The encoding of the
kd-tree prototype takes one pass regardless.

With `RECORD=n` in the environment, every stream is packed in records of `n`
values (CP2K: 1024), each with a length word, and every record decodes into a
reused buffer, which keeps the decoded values in cache as CP2K does.

With a Fortran compiler (unless `FORTRAN=0`), the output also has a group of 8
in Fortran (`vpack.f`): eight interleaved streams in plain array syntax.
