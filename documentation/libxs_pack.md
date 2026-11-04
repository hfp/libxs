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
  libxs_data_t type, size_t n, int maxbits, int level);
size_t libxs_vpack_bound(size_t n, int maxbits);
size_t libxs_vunpack(void* values, const void* packed, size_t npacked,
  libxs_data_t type, size_t n);
```

Packs `n` values of `type` (integers as above, or `LIBXS_DATATYPE_F64` and
`F32`) into a stream of words whose size follows the magnitudes, so small values
or zeros cost little. Integers are taken as integers of `maxbits` bits (1 to
eight times the type size); small magnitudes of either sign stay small. For
floating-point types, `maxbits` sets the accuracy: the width of the type (64 or
32, or more) is lossless; fewer bits keep that many leading bits of every value
(at least sign, exponent, and one bit), which bounds the error relative to the
value; zero or less bounds the absolute error by 2^`maxbits`, and values that
cannot be held to it (Inf, NaN, or too large) are kept exactly.

The `level` selects a trade-off: 1 packs and unpacks fastest, 2 takes less
space, and 3 (or more) the least; each step takes more time to pack and to
unpack. Level 0 (or less) takes the default, which is 2 unless `LIBXS_PACK` in
the environment says otherwise (read at the first call).

Unpacking needs only `n` and the number of words available (`npacked`, checked
against the stream in debug builds only), whatever the level. The stream does
not record the type: unpack with a type of the same signedness as used for
packing. The layout of the stream is internal and may change between versions:
packed data is meant to be unpacked by the same library, not to be stored.

`libxs_vpack` returns the number of words written, at most
`libxs_vpack_bound(n, maxbits)` for any level, and `libxs_vunpack` the number
of words consumed. Both return zero for invalid arguments (including NULL
pointers); `libxs_vunpack` also returns zero for an empty stream (`npacked` of
zero) and leaves the values untouched.

From Fortran (`USE libxs`), `libxs_vpack(packed, values, maxbits[, level])`
and `libxs_vunpack(values, packed)` take contiguous `INTEGER(C_LONG_LONG)`,
`REAL(C_DOUBLE)`, or `REAL(C_FLOAT)` values (and `INTEGER(C_LONG_LONG)` words)
and return the number of words as `INTEGER(C_SIZE_T)`;
`libxs_vpack_bound(n, maxbits)` is bound directly.

The [pack sample](samples/libxs_pack.md) measures the trade-offs on synthetic
data or on MHD files.
