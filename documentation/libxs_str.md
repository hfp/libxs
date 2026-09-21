# String Utilities

Header: `libxs_str.h`

Case-insensitive string search, edit distance, word-level
matching and difference, similarity scoring, UTF-8 code point
iteration, and value formatting.

## Substring Search

```C
const char* libxs_stristrn(const char a[], const char b[],
  size_t maxlen);
const char* libxs_stristr(const char a[], const char b[]);
```

Case-insensitive substring search. `stristrn` limits the match
length to `maxlen` characters of `b`. Returns a pointer to the
first match in `a`, or NULL.

## Edit Distance

```C
int libxs_stridist(const char a[], const char b[]);
```

Case-insensitive character-level edit distance between two
strings. Returns the minimum number of single-character
insertions, deletions, substitutions, or transpositions of
adjacent characters to transform `a` into `b` (ignoring case),
i.e., the restricted Damerau-Levenshtein distance. Returns -1
for NULL input.

## Word Matching

```C
int libxs_strimatch(const char a[], const char b[],
  const char delims[], int* count);
```

Word-level fuzzy matching. Counts how many words in `a` (or `b`)
have a match in the other string, where two words match if one
begins the other (case-insensitive), e.g., `Prod` and `Product`.
The result is the smaller of both counts (symmetric). A word
starting with `[` ends the words of its string (device IDs).
Optional `delims` define word separators (default: space, tab,
semicolon, comma, colon, dash). Optional `count` receives the
word count of the string with more words. Returns -1 for invalid
input.

## Word Difference

```C
int libxs_stridiff(const char a[], const char b[],
  const char delims[], int tolerance, int* count);
```

Word-level set difference with edit-distance tolerance. Counts
how many words in the *smaller* string cannot be matched (within
`tolerance` edits) to any word in the larger string. Matching is
greedy (cheapest pair first), case-insensitive, and each word
can be consumed at most once.

Parameters:

- `a`, `b` — input strings (NULL returns -1)
- `delims` — word separator characters (NULL uses default:
  space, tab, semicolon, comma, colon, dash)
- `tolerance` — maximum edit distance (`libxs_stridist`) for two words to
  be considered a match (0 = exact match only, 1 = allows one
  edit such as plural/tense inflection)
- `count` — optional output, receives the word count of the
  larger string

Returns the number of unmatched words from the smaller string.
A return value of 0 means every word in the shorter string found
a match in the longer one. Returns -1 for NULL input.

Properties:

- Symmetric: `stridiff(a, b, ...)` equals `stridiff(b, a, ...)`
  because matching operates from the smaller side.
- Order-independent: word positions do not affect the result.
- Tolerance=0 gives exact word-level multiset difference.
- Tolerance=1 handles common morphological variation (plurals,
  verb tenses, e.g., "thread" matches "threads").

Example — measuring sentence redundancy:

```C
int defect, total;
defect = libxs_stridiff(
  "The cache stores frequently accessed data.",
  "Frequently accessed data is kept in memory.",
  NULL, 1, &total);
/* defect=2 (unmatched: "kept", "memory"), total=7 */
/* redundancy = 1 - defect/total = 0.71 */
```

## Word Similarity

```C
typedef enum libxs_strisimilar_t {
  LIBXS_STRISIMILAR_GREEDY,
  LIBXS_STRISIMILAR_TWOOPT,
  LIBXS_STRISIMILAR_DEFAULT = LIBXS_STRISIMILAR_GREEDY
} libxs_strisimilar_t;

int libxs_strisimilar(const char a[], const char b[],
  const char delims[], libxs_strisimilar_t kind, int* order);
```

Word-level similarity score combining edit distance and
word-order analysis.

Strings are split into words using the same delimiters as
`libxs_strimatch`. Each word in `a` is matched to a word in `b`
via minimum-cost bipartite matching, where the cost of a pair is
the character-level edit distance of `libxs_stridist`
(case-insensitive, an adjacent transposition counting as one edit).
Unmatched words (when the strings have different word counts)
contribute their full length.

The matching strategy is selected by `kind`:

- `GREEDY` — picks the globally cheapest pair first.
- `TWOOPT` — refines the greedy result by iteratively swapping
  pairs whenever a swap reduces total cost.

The optional `order` output receives the number of pairwise
inversions among matched words (Kendall tau distance),
measuring how much the word order differs (0 = same order).

Returns the total edit distance (0 for identical word sets in
any order), or -1 for invalid input.

## Token Extraction

```C
const char* libxs_strtoken(const char str[],
  const char delims[], int index, int* length);
```

Non-destructive access to the `index`-th token in a delimited
string. Tokens are separated by characters in `delims` (default:
comma). Leading and trailing whitespace within each token is
trimmed. Returns a pointer into `str` at the token start, or
NULL if `index` is out of range. Optional `length` receives the
trimmed token length.

## UTF-8 Code Points

```C
size_t libxs_utf8_size(const unsigned char text[], size_t size,
  size_t pos);
unsigned long libxs_utf8_decode(const unsigned char text[],
  size_t size, int* width);
```

Two entry points because the right answer for malformed input
depends on the question being asked.

`libxs_utf8_size` is **lenient**: it reports the width the lead
byte at `text[pos]` claims, clamped to the bytes that remain, and
never less than 1 so a scan always advances. It does not inspect
continuation bytes. Use it to walk text a code point at a time.

`libxs_utf8_decode` is **strict**: it returns the code point and
writes the bytes consumed to `width` (may be NULL). A sequence
that is truncated, or whose continuation bytes are not
continuation bytes, yields the **lead byte** as the value and a
width of 1. Use it when the code point value is wanted.

On well-formed text the two agree, and a scan by either covers the
string exactly once. They diverge only on malformed input, where
the lenient form skips the claimed span and the strict form skips
one byte:

| Input        | `utf8_size` | `utf8_decode`     |
|--------------|-------------|-------------------|
| `41`         | 1           | U+0041, width 1   |
| `C3 A4`      | 2           | U+00E4, width 2   |
| `E2 80 99`   | 3           | U+2019, width 3   |
| `C3 78`      | 2           | U+00C3, width 1   |
| `C3` (alone) | 1 (clamped) | U+00C3, width 1   |

The strict rule exists because a caller that tests a property of
the value — is this a vowel, is this punctuation — must not be
handed a code point assembled from bytes that do not belong to it.

## Value Formatting

```C
size_t libxs_format_value(char buffer[], int buffer_size,
  size_t nbytes, const char scale[], const char* unit, int base);
```

Format a scalar value with SI-style scaling. Example:

```C
libxs_format_value(buf, sizeof(buf), nbytes, "KMGT", "B", 10);
```

produces a human-readable byte count such as "128 KB". Returns
the value in the selected unit so the caller can decide whether
to print the buffer.

## Relationship Between Functions

The string utilities form a hierarchy of comparison granularity:

| Function       | Granularity | Returns             | Use case                |
|----------------|-------------|---------------------|-------------------------|
| `stridist`     | characters  | edit distance       | spelling similarity     |
| `strimatch`    | words       | matched word count  | overlap detection       |
| `stridiff`     | words       | unmatched count     | redundancy measurement  |
| `strisimilar`  | words       | total edit cost     | structural similarity   |

`stridiff` is the word-level analog of the byte-level
`libxs_setdiff` (from `libxs_math.h`). Where `setdiff` counts
unmatched elements in numeric arrays within a tolerance,
`stridiff` counts unmatched words within an edit-distance
tolerance. Both are order-independent and symmetric.
