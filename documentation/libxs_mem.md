# Memory and Hashing Utilities

Header: `libxs_mem.h`

Byte-level memory operations (macros), comparison, hashing, string search, value formatting, and data-shuffling primitives.

## Memory Macros

```C
LIBXS_MEMSET(DST, SRC, SIZE)
```

Set `SIZE` bytes at `DST` to the value `SRC`, unrolled at compile-time.

```C
LIBXS_MEMZERO(DST)
```

Zero all bytes of `*DST` (size derived from `sizeof(*DST)`).

```C
LIBXS_MEMCPY(DST, SRC, SIZE)
```

Copy `SIZE` bytes from `SRC` to `DST`, unrolled at compile time.

```C
LIBXS_ASSIGN(DST, SRC)
```

Copy `sizeof(*SRC)` bytes from `SRC` to `DST`. Asserts `sizeof(*SRC) <= sizeof(*DST)`.

```C
LIBXS_MEMSWP(PTR_A, PTR_B, SIZE)
```

Swap `SIZE` bytes between `PTR_A` and `PTR_B`. Asserts `PTR_A != PTR_B`.

```C
LIBXS_VALUE_ASSIGN(DST, SRC)
LIBXS_VALUE_SWAP(A, B)
```

Operate on L-values directly (address-of is taken internally). `VALUE_ASSIGN` can cast away const qualifiers. `VALUE_SWAP` asserts equal size.

## Offset and Alignment

```C
size_t libxs_offset(size_t ndims, const size_t offset[],
  const size_t shape[], size_t* size);
```

Compute the linear offset of an n-dimensional coordinate within a shape. Optionally returns the total linear size of the shape.

```C
int libxs_aligned(const void* ptr, const size_t* inc,
  int* alignment);
```

Check whether `ptr` is SIMD-aligned (optionally considering the next access at `ptr + *inc`). Optionally writes the pointer alignment in bytes to `*alignment`.

## Comparison

```C
unsigned char libxs_diff(const void* a, const void* b,
  unsigned char size);
```

Compare two short buffers. Returns zero when equal, non-zero otherwise.

```C
unsigned int libxs_diff_n(const void* a, const void* bn,
  unsigned char elemsize, unsigned char stride,
  unsigned int hint, unsigned int count);
```

Compare `a` against `count` elements in `bn`. Returns the first index that matches `a`, or `count` if none match. `hint` supplies the initial search position.

```C
int libxs_memcmp(const void* a, const void* b, size_t size);
```

Boolean variant of the C `memcmp`. Returns zero when equal.

## Hashing

```C
unsigned int libxs_hash(const void* data, unsigned int size,
  unsigned int seed);
unsigned int libxs_hash8(unsigned int data);
unsigned int libxs_hash16(unsigned int data);
unsigned int libxs_hash32(unsigned long long data);
```

Produce a 32-bit hash from a buffer (with seed) or from an 8/16/32-bit key directly. NULL buffer is accepted (returns seed).

```C
unsigned long long libxs_hash_string(const char string[]);
```

64-bit hash of a character string. NULL-string is accepted.

## String Search and Similarity

```C
int libxs_stridist(const char a[], const char b[]);
```

Case-insensitive character-level edit distance (Levenshtein) between two strings. Returns the minimum number of single-character insertions, deletions, or substitutions to transform `a` into `b` (ignoring case). Returns -1 for NULL input.

```C
const char* libxs_stristrn(const char a[], const char b[],
  size_t maxlen);
const char* libxs_stristr(const char a[], const char b[]);
```

Case-insensitive substring search. `stristrn` limits the search to `maxlen` characters. Returns the pointer to the first match, or NULL.

```C
int libxs_strimatch(const char a[], const char b[],
  const char delims[], int* count);
```

Word-level fuzzy matching. Counts how many words in `a` (or `b`) appear in the other string (case-insensitive, symmetric). Optional `delims` define word separators; optional `count` receives the total word count.

```C
typedef enum libxs_strisimilar_t {
  LIBXS_STRISIMILAR_GREEDY,
  LIBXS_STRISIMILAR_TWOOPT,
  LIBXS_STRISIMILAR_DEFAULT = LIBXS_STRISIMILAR_GREEDY
} libxs_strisimilar_t;

int libxs_strisimilar(const char a[], const char b[],
  const char delims[], libxs_strisimilar_t kind, int* order);
```

Word-level similarity score combining edit distance and word-order analysis.

Strings are split into words using the same delimiters as `libxs_strimatch`. Each word in `a` is matched to a word in `b` via minimum-cost bipartite matching, where the cost of a pair is the character-level Levenshtein distance (case-insensitive). Unmatched words (when the strings have different word counts) contribute their full length.

The matching strategy is selected by `kind`:

- `GREEDY` picks the globally cheapest pair first.
- `TWOOPT` refines the greedy result by iteratively swapping pairs whenever a swap reduces total cost.

The optional `order` output receives the number of pairwise inversions among matched words (Kendall tau distance), measuring how much the word order differs (0 means same order). Returns the total edit distance (0 for identical word sets in any order), or -1 for invalid input.

## Value Formatting

```C
size_t libxs_format_value(char buffer[], int buffer_size,
  size_t nbytes, const char scale[], const char* unit, int base);
```

Format a scalar value with SI-style scaling. Example: `libxs_format_value(buf, sizeof(buf), nbytes, "KMGT", "B", 10)` produces a human-readable byte count. Returns the value in the selected unit.

## Matrix Copy and Transposition

```C
void libxs_matcopy(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo);
void libxs_matcopy_task(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo,
  int tid, int ntasks);
```

Copy an m-by-n matrix from `in` to `out` with leading dimensions `ldi` and `ldo`. If `in` is NULL the destination is zeroed. The `_task` variant distributes work across `ntasks` threads (caller passes its `tid`).

```C
void libxs_otrans(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo);
void libxs_otrans_task(void* out, const void* in,
  unsigned int typesize, int m, int n, int ldi, int ldo,
  int tid, int ntasks);
```

Out-of-place matrix transposition. The `_task` variant distributes work across `ntasks` threads.

```C
void libxs_itrans(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo, void* scratch);
void libxs_itrans_task(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo, void* scratch,
  int tid, int ntasks);
```

In-place matrix transposition (square or via scratch buffer). `scratch` can be NULL (auto-allocated).

```C
void libxs_itrans_batch(void* inout, unsigned int typesize,
  int m, int n, int ldi, int ldo,
  int index_base, int index_stride,
  const int stride[], int batchsize,
  int tid, int ntasks);
```

Batch of in-place matrix transpositions (per-thread form).

## Shuffling

```C
int libxs_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat);
```

In-place shuffle of `count` elements. The `shuffle` stride must be co-prime to `count` (NULL selects `libxs_coprime2(count)`). `nrepeat` defaults to 1.

```C
int libxs_shuffle2(void* dst, const void* src, size_t elemsize,
  size_t count, const size_t* shuffle, const size_t* nrepeat);
```

Out-of-place shuffle from `src` to `dst`. If `*nrepeat` is zero an ordinary copy is performed.

```C
size_t libxs_unshuffle(size_t count, const size_t* shuffle);
```

Return the number of `libxs_shuffle2` applications needed to restore the original order.
