LIBXS_API_INLINE const char* internal_libxs_predict_detect_delims(const char* line)
{
  const char* result = " ";
  if (NULL != strchr(line, ';')) result = ";";
  else if (NULL != strchr(line, ',')) result = ",";
  else if (NULL != strchr(line, '\t')) result = "\t";
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_resolve_col(
  const char* name, const char* header, const char* delims)
{
  int result = -1;
  char* endptr = NULL;
  const long idx = strtol(name, &endptr, 10);
  if (endptr != name && '\0' == *endptr) {
    result = (int)idx;
  }
  else if (NULL != header) {
    const char* p = header;
    int col = 0;
    while ('\0' != *p && 0 > result) {
      const char* field = p;
      size_t flen;
      while ('\0' != *p && NULL == strchr(delims, *p)) ++p;
      flen = (size_t)(p - field);
      if (0 != libxs_striequal(field, flen, name, strlen(name))) result = col;
      if ('\0' != *p) ++p;
      ++col;
    }
  }
  return result;
}


/** Opens the reader on a file; the buffer starts at one chunk. */
LIBXS_API_INLINE int internal_libxs_predict_lines_open(
  internal_libxs_predict_lines_t* reader, FILE* file)
{
  reader->file = file;
  reader->cap = LIBXS_PREDICT_CSV_CHUNK;
  reader->len = reader->pos = 0;
  reader->eof = 0;
  reader->buf = (char*)LIBXS_PREDICT_MALLOC(reader->cap + 1, reader->pool);
  return (NULL != reader->buf) ? EXIT_SUCCESS : EXIT_FAILURE;
}


LIBXS_API_INLINE void internal_libxs_predict_lines_rewind(
  internal_libxs_predict_lines_t* reader)
{
  rewind(reader->file);
  reader->len = reader->pos = 0;
  reader->eof = 0;
}


/**
 * The next line, terminated in place and without its line break, or NULL at the
 * end of the file. A line the buffer holds only the start of moves to the front
 * and the rest of it is read behind it; one longer than the buffer grows it.
 */
LIBXS_API_INLINE char* internal_libxs_predict_lines_next(
  internal_libxs_predict_lines_t* reader)
{
  char* result = NULL;
  int done = 0;
  while (0 == done) {
    char* const start = reader->buf + reader->pos;
    char* const nl = (char*)memchr(start, '\n', reader->len - reader->pos);
    if (NULL != nl || 0 != reader->eof) {
      char* const end = (NULL != nl) ? nl : (reader->buf + reader->len);
      if (start < end || NULL != nl) {
        result = start;
        *end = '\0';
        if (start < end && '\r' == end[-1]) end[-1] = '\0';
      }
      reader->pos = (size_t)(end - reader->buf) + ((NULL != nl) ? 1 : 0);
      done = 1;
    }
    else {
      const size_t rest = reader->len - reader->pos;
      size_t got = 0;
      memmove(reader->buf, start, rest);
      reader->len = rest;
      reader->pos = 0;
      if (reader->len == reader->cap) {
        int pool = 0;
        char* const grown = (char*)LIBXS_PREDICT_MALLOC(
          2 * reader->cap + 1, pool);
        if (NULL != grown) {
          memcpy(grown, reader->buf, reader->len);
          LIBXS_PREDICT_FREE(reader->buf, reader->pool);
          reader->buf = grown;
          reader->pool = pool;
          reader->cap *= 2;
        }
      }
      if (reader->len < reader->cap) {
        got = fread(reader->buf + reader->len, 1, reader->cap - reader->len,
          reader->file);
      }
      reader->len += got;
      if (0 == got) reader->eof = 1;
    }
  }
  return result;
}


LIBXS_PRAGMA_FASTMATH_OFF
/**
 * A decimal field converted where its value needs no rounding decision, and by
 * strtod otherwise. With digits of at most 2^53 and a power of ten that is itself
 * exact, one multiplication or division is the correctly rounded value, which is
 * the double strtod returns (Clinger's fast path). The fast path is taken only
 * for a plain decimal that runs to the end of its field; more digits, another
 * spelling (hexadecimal, inf, nan), or a decimal point other than '.' in the
 * current locale go to strtod. Returns the end of what was converted.
 */
LIBXS_API_INLINE const char* internal_libxs_predict_parse_num(
  const char* field, const unsigned char delim[256], int fast, double* value)
{
  static const double pow10[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
  };
  const char* p = field;
  const char* result = NULL;
  unsigned long long w = 0;
  int e10 = 0, ndigits = 0, neg = 0, exact = fast;
  if ('-' == *p || '+' == *p) neg = ('-' == *p++) ? 1 : 0;
  for (; '0' <= *p && '9' >= *p; ++p, ++ndigits) {
    if (1844674407370955160ULL >= w) w = w * 10 + (unsigned int)(*p - '0');
    else if ('0' == *p) ++e10;
    else exact = 0;
  }
  if ('.' == *p) {
    for (++p; '0' <= *p && '9' >= *p; ++p, ++ndigits) {
      if (1844674407370955160ULL >= w) {
        w = w * 10 + (unsigned int)(*p - '0');
        --e10;
      }
      else if ('0' != *p) exact = 0;
    }
  }
  if (0 < ndigits && ('e' == *p || 'E' == *p)) {
    const char* q = p + 1;
    int eneg = 0, ex = 0, nexp = 0;
    if ('-' == *q || '+' == *q) eneg = ('-' == *q++) ? 1 : 0;
    for (; '0' <= *q && '9' >= *q; ++q, ++nexp) {
      if (100000 > ex) ex = ex * 10 + (*q - '0');
    }
    if (0 < nexp) {
      e10 += (0 != eneg) ? -ex : ex;
      p = q;
    }
  }
  while (0 != w && 0 == w % 10) {
    w /= 10;
    ++e10;
  }
  if (0 < ndigits && 0 != exact && 9007199254740992ULL >= w
    && -22 <= e10 && 22 >= e10 && ('\0' == *p || 0 != delim[(unsigned char)*p]))
  {
    const double v = (double)w;
    *value = (0 > e10) ? (v / pow10[-e10]) : (v * pow10[e10]);
    if (0 != neg) *value = -*value;
    result = p;
  }
  else {
    char* endptr = NULL;
    *value = strtod(field, &endptr);
    result = endptr;
  }
  return result;
}


LIBXS_PRAGMA_FASTMATH_ON


/**
 * Fill the inputs and the outputs of one row in one pass over its fields. A
 * column's slots are chained from first[col] through next[], so a field costs one
 * lookup and one conversion however many slots read it. With missing set, an
 * input that does not parse - or a column the row does not reach at all - is
 * recorded as absent instead of rejecting the whole row. An output is never
 * absent: it leaves nothing to learn from, so that row is still skipped.
 */
LIBXS_API_INLINE int internal_libxs_predict_parse_fields(
  const char* line, const unsigned char delim[256], int fast,
  const int first[], int nfirst, const int next[], int ninputs, int noutputs,
  double vals[], int missing)
{
  int nin = 0, nout = 0, col = 0, ok = 1, i;
  const char* p = line;
  if (0 != missing) {
    for (i = 0; i < ninputs; ++i) vals[i] = internal_libxs_predict_absent();
  }
  while ('\0' != *p && (nin < ninputs || nout < noutputs) && 0 != ok) {
    const char* const field = p;
    while ('\0' != *p && 0 == delim[(unsigned char)*p]) ++p;
    if (col < nfirst && 0 <= first[col]) {
      double v = 0;
      const char* const end = internal_libxs_predict_parse_num(
        field, delim, fast, &v);
      const int valid = (end != field && (end == p || '\0' == *end
        || 0 != delim[(unsigned char)*end])) ? 1 : 0;
      for (i = first[col]; 0 <= i; i = next[i]) {
        if (0 != valid) vals[i] = v;
        if (i < ninputs) {
          if (0 != valid || 0 != missing) ++nin;
          else ok = 0;
        }
        else if (0 != valid) ++nout;
        else ok = 0;
      }
    }
    if ('\0' != *p) ++p;
    ++col;
  }
  return (0 != ok && (nin >= ninputs || 0 != missing) && nout >= noutputs)
    ? 1 : 0;
}


LIBXS_API_INLINE int internal_libxs_predict_tokenize(
  const char spec[], char buf[], const char* tokens[], int maxtokens)
{
  int n = 0;
  const char* p = spec;
  char* dst = buf;
  while ('\0' != *p && n < maxtokens) {
    while (' ' == *p || '\t' == *p) ++p;
    if ('\0' == *p) break;
    tokens[n] = dst;
    while ('\0' != *p && ',' != *p) {
      *dst++ = *p++;
    }
    while (dst > tokens[n] && (' ' == dst[-1] || '\t' == dst[-1])) --dst;
    *dst++ = '\0';
    ++n;
    if (',' == *p) ++p;
  }
  return n;
}


LIBXS_API int libxs_predict_load_csv(libxs_predict_t* model,
  const char filename[], const char delims[],
  const char inputs[], const char outputs[],
  char header[], int header_size, char* delim_out)
{
  libxs_predict_csv_t opts;
  memset(&opts, 0, sizeof(opts));
  opts.delims = delims;
  opts.inputs = inputs;
  opts.outputs = outputs;
  opts.header = header;
  opts.header_size = header_size;
  opts.delim_out = delim_out;
  return libxs_predict_load_csv_opts(model, filename, &opts);
}


LIBXS_API int libxs_predict_load_csv_opts(libxs_predict_t* model,
  const char filename[], const libxs_predict_csv_t* opts)
{
  int result = -1;
  FILE* file;
  const int ninputs = model->ninputs;
  const int noutputs = model->noutputs;
  const char* delims = (NULL != opts) ? opts->delims : NULL;
  const char* inputs = (NULL != opts) ? opts->inputs : NULL;
  const char* outputs = (NULL != opts) ? opts->outputs : NULL;
  char* header = (NULL != opts) ? opts->header : NULL;
  const int header_size = (NULL != opts) ? opts->header_size : 0;
  char* delim_out = (NULL != opts) ? opts->delim_out : NULL;
  const int nrows = (NULL != opts) ? opts->nrows : 0;
  const int nskip = (NULL != opts && 0 < opts->offset) ? opts->offset : 0;
  const int stride = (NULL != opts && 1 < opts->stride) ? opts->stride : 1;
  internal_libxs_predict_lines_t reader;
  LIBXS_ASSERT(NULL != model && NULL != filename);
  file = fopen(filename, "r");
  if (NULL != file && EXIT_SUCCESS != internal_libxs_predict_lines_open(
    &reader, file))
  {
    fclose(file);
    file = NULL;
  }
  if (NULL != file) {
    static char empty[] = "";
    char* line = NULL;
    double vals[128];
    const char* sep = delims;
    const char* input_tokens[64];
    const char* output_tokens[64];
    char tokbuf[2048];
    unsigned char delim[256];
    int idx[128], next[128], local[256];
    int* first = local;
    int i, named = 0, resolved = 1, ncols = 1, nextra = 0, nfirst = 0;
    int first_pool = 0;
    int ni = 0, no = 0;
    /* the fast conversion reads '.' as the decimal point, as strtod does only
     * in a locale that agrees */
    const int fast = ('.' == localeconv()->decimal_point[0]
      && '\0' == localeconv()->decimal_point[1]) ? 1 : 0;
    LIBXS_UNUSED(ni); LIBXS_UNUSED(no);
    LIBXS_ASSERT(ninputs + noutputs <= 128);
    if (NULL != inputs) {
      ni = internal_libxs_predict_tokenize(inputs, tokbuf, input_tokens, 64);
    }
    if (NULL != outputs) {
      no = internal_libxs_predict_tokenize(outputs,
        tokbuf + sizeof(tokbuf) / 2, output_tokens, 64);
    }
    /**
     * A spec that yields no tokens - an empty or all-blank string - is the
     * same request as no spec at all, and must fall back to sequential columns
     * rather than leave the token pointers unread.  The count is otherwise only
     * checked by an assert, so a release build would index uninitialized
     * pointers and report zero rows loaded for what looks like a valid call.
     */
    if (0 >= ni) inputs = NULL;
    if (0 >= no) outputs = NULL;
    LIBXS_ASSERT((NULL == inputs || ni == ninputs)
      && (NULL == outputs || no == noutputs));
    result = 0;
    do line = internal_libxs_predict_lines_next(&reader);
    while (NULL != line && '#' == line[0]);
    if (NULL == line) line = empty;
    if (NULL != header && header_size > 0) {
      size_t hlen = strlen(line);
      if ((int)hlen >= header_size) hlen = (size_t)(header_size - 1);
      memcpy(header, line, hlen);
      header[hlen] = '\0';
    }
    if (NULL == sep && '\0' != line[0]) {
      sep = internal_libxs_predict_detect_delims(line);
    }
    if (NULL == sep) sep = ",";
    memset(delim, 0, sizeof(delim));
    for (i = 0; '\0' != sep[i]; ++i) delim[(unsigned char)sep[i]] = 1;
    { const char* cp = line;
      while ('\0' != *cp) {
        if (0 != delim[(unsigned char)*cp]) ++ncols;
        ++cp;
      }
    }
    for (i = 0; i < ninputs && 0 != resolved; ++i) {
      if (NULL != inputs) {
        char* endptr = NULL;
        strtol(input_tokens[i], &endptr, 10);
        if (endptr == input_tokens[i] || '\0' != *endptr) named = 1;
        idx[i] = internal_libxs_predict_resolve_col(
          input_tokens[i], line, sep);
        if (0 > idx[i]) resolved = 0;
      }
      else idx[i] = i;
    }
    for (i = 0; i < noutputs && 0 != resolved; ++i) {
      if (NULL != outputs) {
        char* endptr = NULL;
        strtol(output_tokens[i], &endptr, 10);
        if (endptr == output_tokens[i] || '\0' != *endptr) named = 1;
        idx[ninputs + i] = internal_libxs_predict_resolve_col(
          output_tokens[i], line, sep);
        if (0 > idx[ninputs + i]) {
          if (0 != named) idx[ninputs + i] = ncols + nextra++;
          else resolved = 0;
        }
      }
      else idx[ninputs + i] = ninputs + i;
    }
    /* each column's slots in a chain, read in one pass over the fields */
    for (i = 0; i < ninputs + noutputs && 0 != resolved; ++i) {
      if (nfirst <= idx[i]) nfirst = idx[i] + 1;
    }
    if ((int)(sizeof(local) / sizeof(*local)) < nfirst) {
      first = (int*)LIBXS_PREDICT_MALLOC((size_t)nfirst * sizeof(int),
        first_pool);
      if (NULL == first) resolved = 0;
    }
    for (i = 0; i < nfirst && 0 != resolved; ++i) first[i] = -1;
    for (i = ninputs + noutputs - 1; 0 <= i && 0 != resolved; --i) {
      next[i] = first[idx[i]];
      first[idx[i]] = i;
    }
    if (0 == named || 0 == resolved) internal_libxs_predict_lines_rewind(&reader);
    if (NULL != delim_out) {
      *delim_out = sep[0];
    }
    { int nadmit = 0; /* admissible rows seen, which stride/offset count */
      while (0 != resolved && (0 >= nrows || result < nrows)
        && NULL != (line = internal_libxs_predict_lines_next(&reader)))
      {
        if ('#' == line[0]) continue;
        if (0 != internal_libxs_predict_parse_fields(line, delim, fast, first,
          nfirst, next, ninputs, noutputs, vals, model->missing_mode))
        {
          const int taken = (nadmit >= nskip)
            && (0 == ((nadmit - nskip) % stride));
          ++nadmit;
          if (0 != taken && EXIT_SUCCESS == libxs_predict_push(
            NULL, model, vals, vals + ninputs))
          {
            ++result;
          }
        }
      }
    }
    if (local != first) LIBXS_PREDICT_FREE(first, first_pool);
    LIBXS_PREDICT_FREE(reader.buf, reader.pool);
    fclose(file);
  }
  return result;
}
