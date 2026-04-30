# String Utilities

Header: `libxs_str.h`

Case-insensitive string search, edit distance, word-level
similarity scoring, and value formatting.

## Substring Search

```C
const char* libxs_stristrn(const char a[], const char b[],
  size_t maxlen);
const char* libxs_stristr(const char a[], const char b[]);
```

Case-insensitive substring search. stristrn limits the match
length to maxlen characters of b. Returns a pointer to the
first match in a, or NULL.

## Edit Distance

```C
int libxs_stridist(const char a[], const char b[]);
```

Case-insensitive character-level edit distance (Levenshtein)
between two strings. Returns the minimum number of
single-character insertions, deletions, or substitutions to
transform a into b (ignoring case). Returns -1 for NULL input.

## Word Matching

```C
int libxs_strimatch(const char a[], const char b[],
  const char delims[], int* count);
```

Word-level fuzzy matching. Counts how many words in a (or b)
appear in the other string (case-insensitive, symmetric).
Optional delims define word separators (default: space, tab,
semicolon, comma, colon, dash). Optional count receives the
total word count. Returns -1 for invalid input.

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
libxs_strimatch. Each word in a is matched to a word in b via
minimum-cost bipartite matching, where the cost of a pair is
the character-level Levenshtein distance (case-insensitive).
Unmatched words (when the strings have different word counts)
contribute their full length.

The matching strategy is selected by kind:

  GREEDY    Picks the globally cheapest pair first.
  TWOOPT    Refines the greedy result by iteratively swapping
            pairs whenever a swap reduces total cost.

The optional order output receives the number of pairwise
inversions among matched words (Kendall tau distance),
measuring how much the word order differs (0 = same order).

Returns the total edit distance (0 for identical word sets in
any order), or -1 for invalid input.

## Value Formatting

```C
size_t libxs_format_value(char buffer[], int buffer_size,
  size_t nbytes, const char scale[], const char* unit, int base);
```

Format a scalar value with SI-style scaling. Example:

  libxs_format_value(buf, sizeof(buf), nbytes, "KMGT", "B", 10)

produces a human-readable byte count such as "128 KB". Returns
the value in the selected unit so the caller can decide whether
to print the buffer.
