# String Utilities

Header: `libxs_str.h`

Case-insensitive string search, edit distance, word-level
matching and difference, similarity scoring, and value formatting.

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

Case-insensitive character-level edit distance (Levenshtein)
between two strings. Returns the minimum number of
single-character insertions, deletions, or substitutions to
transform `a` into `b` (ignoring case). Returns -1 for NULL input.

## Word Matching

```C
int libxs_strimatch(const char a[], const char b[],
  const char delims[], int* count);
```

Word-level fuzzy matching. Counts how many words in `a` (or `b`)
appear in the other string (case-insensitive, symmetric).
Optional `delims` define word separators (default: space, tab,
semicolon, comma, colon, dash). Optional `count` receives the
total word count of the larger string. Returns -1 for invalid input.

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

- `a`, `b` -- input strings (NULL returns -1)
- `delims` -- word separator characters (NULL uses default:
  space, tab, semicolon, comma, colon, dash)
- `tolerance` -- maximum Levenshtein distance for two words to
  be considered a match (0 = exact match only, 1 = allows one
  edit such as plural/tense inflection)
- `count` -- optional output, receives the word count of the
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

Example -- measuring sentence redundancy:

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
the character-level Levenshtein distance (case-insensitive).
Unmatched words (when the strings have different word counts)
contribute their full length.

The matching strategy is selected by `kind`:

- `GREEDY` -- picks the globally cheapest pair first.
- `TWOOPT` -- refines the greedy result by iteratively swapping
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
