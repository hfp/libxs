LIBXS_API_INLINE uint64_t internal_libxs_hilbert_bits(
  const unsigned int coords[], int ndims, int bits_per_dim)
{
  unsigned int x[64];
  uint64_t code = 0;
  int i, level;
  for (i = 0; i < ndims; ++i) x[i] = coords[i];
  /* Skilling: AxestoTranspose -- convert coordinates to transposed form */
  { const unsigned int m = 1u << (bits_per_dim - 1);
    unsigned int p, q, t;
    for (q = m; q > 1; q >>= 1) {
      p = q - 1;
      for (i = 0; i < ndims; ++i) {
        if (0 != (x[i] & q)) {
          x[0] ^= p;
        }
        else {
          t = (x[0] ^ x[i]) & p;
          x[0] ^= t; x[i] ^= t;
        }
      }
    }
    /* Gray encode */
    for (i = 1; i < ndims; ++i) x[i] ^= x[i - 1];
    t = 0;
    for (q = m; q > 1; q >>= 1) {
      if (0 != (x[ndims - 1] & q)) t ^= (q - 1);
    }
    for (i = 0; i < ndims; ++i) x[i] ^= t;
  }
  /* Interleave transposed bits into the code (MSB first) */
  for (level = bits_per_dim - 1; level >= 0; --level) {
    for (i = 0; i < ndims; ++i) {
      code = (code << 1) | ((x[i] >> level) & 1u);
    }
  }
  return code;
}


LIBXS_API uint64_t libxs_hilbert(const unsigned int coords[], int ndims)
{
  uint64_t result = 0;
  if (NULL != coords && 0 < ndims && ndims <= 64) {
    result = internal_libxs_hilbert_bits(coords, ndims, 64 / ndims);
  }
  return result;
}


LIBXS_API uint64_t libxs_hilbert_bits(
  const unsigned int coords[], int ndims, int bits_per_dim)
{
  uint64_t result = 0;
  if (NULL != coords && 0 < ndims && ndims <= 64
    && 0 < bits_per_dim && bits_per_dim <= 32
    && ndims * bits_per_dim <= 64)
  {
    result = internal_libxs_hilbert_bits(coords, ndims, bits_per_dim);
  }
  return result;
}


LIBXS_API_INLINE uint64_t internal_libxs_morton_bits(
  const unsigned int coords[], int ndims, int bits_per_dim)
{
  uint64_t code = 0;
  int bit, d;
  for (bit = bits_per_dim - 1; 0 <= bit; --bit) {
    for (d = ndims - 1; 0 <= d; --d) {
      code = (code << 1) | ((coords[d] >> bit) & 1u);
    }
  }
  return code;
}


LIBXS_API uint64_t libxs_morton(const unsigned int coords[], int ndims)
{
  uint64_t result = 0;
  if (NULL != coords && 0 < ndims && ndims <= 64) {
    result = internal_libxs_morton_bits(coords, ndims, 64 / ndims);
  }
  return result;
}


LIBXS_API uint64_t libxs_morton_bits(
  const unsigned int coords[], int ndims, int bits_per_dim)
{
  uint64_t result = 0;
  if (NULL != coords && 0 < ndims && ndims <= 64
    && 0 < bits_per_dim && bits_per_dim <= 32
    && ndims * bits_per_dim <= 64)
  {
    result = internal_libxs_morton_bits(coords, ndims, bits_per_dim);
  }
  return result;
}


LIBXS_API void libxs_morton_decode(
  uint64_t code, unsigned int coords[], int ndims)
{
  if (NULL != coords && 0 < ndims && ndims <= 64) {
    const int bpd = 64 / ndims;
    int bit, d;
    for (d = 0; d < ndims; ++d) coords[d] = 0;
    for (bit = 0; bit < bpd; ++bit) {
      for (d = 0; d < ndims; ++d) {
        coords[d] |= (unsigned int)(code & 1u) << bit;
        code >>= 1;
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_morton_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim)
{
  if (NULL != coords && 0 < ndims && ndims <= 64
    && 0 < bits_per_dim && bits_per_dim <= 32)
  {
    int bit, d;
    for (d = 0; d < ndims; ++d) coords[d] = 0;
    for (bit = 0; bit < bits_per_dim; ++bit) {
      for (d = 0; d < ndims; ++d) {
        coords[d] |= (unsigned int)(code & 1u) << bit;
        code >>= 1;
      }
    }
  }
}


LIBXS_API void libxs_morton_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim)
{
  internal_libxs_morton_decode_bits(code, coords, ndims, bits_per_dim);
}


LIBXS_API void libxs_hilbert_decode(
  uint64_t code, unsigned int coords[], int ndims)
{
  if (NULL != coords && 0 < ndims && ndims <= 64) {
    const int bpd = 64 / ndims;
    int i, level;
    for (i = 0; i < ndims; ++i) coords[i] = 0;
    for (level = 0; level < bpd; ++level) {
      const int shift = level;
      int d;
      for (d = ndims - 1; 0 <= d; --d) {
        coords[d] |= (unsigned int)(code & 1u) << shift;
        code >>= 1;
      }
    }
    if (1 < ndims) {
      const unsigned int m = 1u << (bpd - 1);
      unsigned int q;
      unsigned int t = coords[ndims - 1] >> 1;
      for (i = ndims - 1; 0 < i; --i) coords[i] ^= coords[i - 1];
      coords[0] ^= t;
      for (q = 2; 0 != q && q <= m; q <<= 1) {
        const unsigned int p = q - 1;
        for (i = ndims - 1; 0 <= i; --i) {
          if (0 != (coords[i] & q)) {
            coords[0] ^= p;
          }
          else {
            t = (coords[0] ^ coords[i]) & p;
            coords[0] ^= t; coords[i] ^= t;
          }
        }
      }
    }
  }
}


LIBXS_API_INLINE void internal_libxs_hilbert_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim)
{
  if (NULL != coords && 0 < ndims && ndims <= 64
    && 0 < bits_per_dim && bits_per_dim <= 32)
  {
    int i, level;
    for (i = 0; i < ndims; ++i) coords[i] = 0;
    for (level = 0; level < bits_per_dim; ++level) {
      const int shift = level;
      int d;
      for (d = ndims - 1; 0 <= d; --d) {
        coords[d] |= (unsigned int)(code & 1u) << shift;
        code >>= 1;
      }
    }
    if (1 < ndims) {
      const unsigned int m = 1u << (bits_per_dim - 1);
      unsigned int q;
      unsigned int t = coords[ndims - 1] >> 1;
      for (i = ndims - 1; 0 < i; --i) coords[i] ^= coords[i - 1];
      coords[0] ^= t;
      for (q = 2; 0 != q && q <= m; q <<= 1) {
        const unsigned int p = q - 1;
        for (i = ndims - 1; 0 <= i; --i) {
          if (0 != (coords[i] & q)) {
            coords[0] ^= p;
          }
          else {
            t = (coords[0] ^ coords[i]) & p;
            coords[0] ^= t; coords[i] ^= t;
          }
        }
      }
    }
  }
}


LIBXS_API void libxs_hilbert_decode_bits(
  uint64_t code, unsigned int coords[], int ndims, int bits_per_dim)
{
  internal_libxs_hilbert_decode_bits(code, coords, ndims, bits_per_dim);
}


LIBXS_API_INLINE int internal_libxs_stratify(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims,
  internal_libxs_sfc_encode_t encode,
  internal_libxs_sfc_encode_bits_t encode_bits,
  internal_libxs_sfc_decode_t decode,
  internal_libxs_sfc_decode_bits_t decode_bits)
{
  int result = EXIT_FAILURE;
  if (NULL != src_coords && NULL != dst_coords && NULL != encode
    && NULL != encode_bits && NULL != decode && 0 < dst_ndims
    && dst_ndims < src_ndims && src_ndims <= 64)
  {
#if defined(LIBXS_PERM_STRATIFY_SIMPLE)
    const uint64_t code = encode(src_coords, src_ndims);
    LIBXS_UNUSED(decode_bits);
    decode(code, dst_coords, dst_ndims);
    result = EXIT_SUCCESS;
#else
    if (NULL != decode_bits) {
      const int src_bits = 64 / src_ndims;
      const int rank_bits = src_bits * src_ndims;
      const int dst_bits = (rank_bits + dst_ndims - 1) / dst_ndims;
      if (0 < dst_bits && dst_bits <= 32) {
        const uint64_t code = encode_bits(src_coords, src_ndims, src_bits);
        decode_bits(code, dst_coords, dst_ndims, dst_bits);
        result = EXIT_SUCCESS;
      }
    }
#endif
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_stratify_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits,
  internal_libxs_sfc_encode_bits_t encode_bits,
  internal_libxs_sfc_decode_bits_t decode_bits)
{
  int result = EXIT_FAILURE;
  if (NULL != src_coords && NULL != dst_coords && NULL != encode_bits
    && NULL != decode_bits && 0 < dst_ndims && dst_ndims < src_ndims
    && src_ndims <= 64 && 0 < src_bits && src_bits <= 32
    && 0 < dst_bits && dst_bits <= 32
    && src_ndims * src_bits <= 64
    && src_ndims * src_bits <= dst_ndims * dst_bits)
  {
    const uint64_t code = encode_bits(src_coords, src_ndims, src_bits);
    decode_bits(code, dst_coords, dst_ndims, dst_bits);
    result = EXIT_SUCCESS;
  }
  return result;
}


LIBXS_API int libxs_stratify_morton(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims)
{
  int result = internal_libxs_stratify(src_coords, src_ndims,
    dst_coords, dst_ndims, libxs_morton, internal_libxs_morton_bits,
    libxs_morton_decode,
    internal_libxs_morton_decode_bits);
  return result;
}


LIBXS_API int libxs_stratify_morton_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits)
{
  int result = internal_libxs_stratify_bits(src_coords, src_ndims, src_bits,
    dst_coords, dst_ndims, dst_bits, internal_libxs_morton_bits,
    internal_libxs_morton_decode_bits);
  return result;
}


LIBXS_API int libxs_stratify_hilbert(
  const unsigned int src_coords[], int src_ndims,
  unsigned int dst_coords[], int dst_ndims)
{
  int result = internal_libxs_stratify(src_coords, src_ndims,
    dst_coords, dst_ndims, libxs_hilbert, internal_libxs_hilbert_bits,
    libxs_hilbert_decode,
    internal_libxs_hilbert_decode_bits);
  return result;
}


LIBXS_API int libxs_stratify_hilbert_bits(
  const unsigned int src_coords[], int src_ndims, int src_bits,
  unsigned int dst_coords[], int dst_ndims, int dst_bits)
{
  int result = internal_libxs_stratify_bits(src_coords, src_ndims, src_bits,
    dst_coords, dst_ndims, dst_bits, internal_libxs_hilbert_bits,
    internal_libxs_hilbert_decode_bits);
  return result;
}
