# Pack

Header: `libxs_pack.h`

Bit packing of integer arrays, at a fixed width or at a width chosen per group
of values. Packed data is a stream of 64-bit words, so the packed buffer must
be 8-byte aligned.

## Fixed Width

```C
int libxs_pack(void* packed, const void* values,
  libxs_data_t type, size_t n, int nbits);
int libxs_unpack(void* values, const void* packed,
  libxs_data_t type, size_t n, int nbits);
```

Packs `n` integers of `type` (`LIBXS_DATATYPE_I8` to `I64`, `U8` to `U64`) at
`nbits` each (1 to eight times the type size) into ⌈n·nbits/64⌉ words.
Bits beyond `nbits` are ignored. Unpacking sign-extends signed types from
`nbits` and zero-extends unsigned types. Both return `EXIT_SUCCESS`, or
`EXIT_FAILURE` for a type or width out of range, or for a NULL pointer
with n > 0.

## Variable Width

```C
size_t libxs_vpack(void* packed, const void* values,
  libxs_data_t type, size_t n, int maxbits, int group);
size_t libxs_vpack_bound(size_t n, int maxbits, int group);
size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n);
```

Every group of `group` values (1 to 255; 0 selects `LIBXS_VPACK_GROUP`,
which is 8) is stored at the width its largest magnitude needs, so groups of
small values or zeros cost little. Values are taken as integers of `maxbits`
bits (1 to eight times the type size); signed types fold the sign so that
small magnitudes of either sign stay narrow. `LIBXS_PACK` in the environment
selects the format (read at the first call). With 1 (the default), groups of a
multiple of 8 are stored as 8 interleaved streams of words, the format of
`samples/pack/vpack.f`. With 2, a group of 8 (the default group) may instead
store a smaller regular width and the few values above it at the group's full
width (exceptions, marked by a bitmap) if that takes fewer bytes. With 3, a
group of 8 is written in a format that takes less space for data of mostly
similar magnitudes and unpacks more slowly; a call falls back to the format of
2 wherever it would not fit `libxs_vpack_bound`. Packing and unpacking need the
same setting.
The stream records the group size and the widths, so unpacking needs only `n`
and the number of words available (`npacked`, checked against the stream in
debug builds only). It does not record the type: unpack with a type of the
same signedness as used for packing.

`libxs_vpack` returns the number of words written, at most
`libxs_vpack_bound(n, maxbits, group)`, and `libxs_vunpack` the number of words
consumed. Both return zero for invalid arguments (including NULL pointers);
`libxs_vunpack` also returns zero for an empty stream (`npacked` of zero) and
leaves the values untouched.

From Fortran (`USE libxs`), `libxs_vpack(packed, values, maxbits[, group])`
and `libxs_vunpack(values, packed)` take contiguous `INTEGER(C_LONG_LONG)`
arrays and return the number of words as `INTEGER(C_SIZE_T)`;
`libxs_vpack_bound(n, maxbits, group)` is bound directly.

The [pack sample](samples/libxs_pack.md) measures both formats on synthetic
data or on MHD files.
