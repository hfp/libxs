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
      if (flen == strlen(name)) {
        const char* hit = libxs_stristrn(field, name, flen);
        if (hit == field) result = col;
      }
      if ('\0' != *p) ++p;
      ++col;
    }
  }
  return result;
}


LIBXS_API_INLINE int internal_libxs_predict_parse_row(
  const char* line, const char* delims, const int idx[], int nidx, double vals[])
{
  int filled = 0, col = 0, ok = 1, i;
  const char* p = line;
  while ('\0' != *p && filled < nidx && 0 != ok) {
    const char* field = p;
    while ('\0' != *p && NULL == strchr(delims, *p)) ++p;
    for (i = 0; i < nidx && 0 != ok; ++i) {
      if (col == idx[i]) {
        char* endptr = NULL;
        const double v = strtod(field, &endptr);
        if (endptr == field || (endptr != p && '\0' != *endptr
          && NULL == strchr(delims, *endptr)))
        {
          ok = 0;
        }
        else {
          vals[i] = v;
          ++filled;
        }
      }
    }
    if ('\0' != *p) ++p;
    ++col;
  }
  return (0 != ok && filled >= nidx) ? 1 : 0;
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
  int result = -1;
  FILE* file;
  const int ninputs = model->ninputs;
  const int noutputs = model->noutputs;
  LIBXS_ASSERT(NULL != model && NULL != filename);
  file = fopen(filename, "r");
  if (NULL != file) {
    char line[4096];
    double vals[128];
    double* inp = vals;
    double* outp = vals + ninputs;
    const char* sep = delims;
    const char* input_tokens[64];
    const char* output_tokens[64];
    char tokbuf[2048];
    int idx[128];
    int i, resolved = 1;
    int ni = 0, no = 0;
    LIBXS_UNUSED(ni); LIBXS_UNUSED(no);
    LIBXS_ASSERT(ninputs + noutputs <= 128);
    if (NULL != inputs) {
      ni = internal_libxs_predict_tokenize(inputs, tokbuf, input_tokens, 64);
    }
    if (NULL != outputs) {
      no = internal_libxs_predict_tokenize(outputs,
        tokbuf + sizeof(tokbuf) / 2, output_tokens, 64);
    }
    LIBXS_ASSERT((NULL == inputs || ni == ninputs)
      && (NULL == outputs || no == noutputs));
    result = 0;
    while (NULL != fgets(line, (int)sizeof(line), file) && '#' == line[0]) {}
    if (NULL != header && header_size > 0) {
      size_t hlen = strlen(line);
      while (hlen > 0 && ('\n' == line[hlen-1] || '\r' == line[hlen-1])) {
        --hlen;
      }
      if ((int)hlen >= header_size) hlen = (size_t)(header_size - 1);
      memcpy(header, line, hlen);
      header[hlen] = '\0';
    }
    if (NULL == inputs || NULL == outputs) {
      for (i = 0; i < ninputs; ++i) {
        idx[i] = (NULL != inputs)
          ? (int)strtol(input_tokens[i], NULL, 10) : i;
      }
      for (i = 0; i < noutputs; ++i) {
        idx[ninputs + i] = (NULL != outputs)
          ? (int)strtol(output_tokens[i], NULL, 10) : ninputs + i;
      }
      if (NULL == sep && '\0' != line[0]) {
        size_t len = strlen(line);
        if (0 < len && '\n' == line[len - 1]) line[--len] = '\0';
        if (0 < len && '\r' == line[len - 1]) line[--len] = '\0';
        sep = internal_libxs_predict_detect_delims(line);
      }
      if (NULL == sep) sep = ",";
      resolved = 0;
    }
    if (0 != resolved && NULL == sep && '\0' != line[0]) {
      size_t len = strlen(line);
      if (0 < len && '\n' == line[len - 1]) line[--len] = '\0';
      if (0 < len && '\r' == line[len - 1]) line[--len] = '\0';
      sep = internal_libxs_predict_detect_delims(line);
      { int ncols = 1, nextra = 0;
        const char* cp = line;
        while ('\0' != *cp) { if (NULL != strchr(sep, *cp)) ++ncols; ++cp; }
        for (i = 0; i < ninputs && 0 != resolved; ++i) {
          idx[i] = internal_libxs_predict_resolve_col(input_tokens[i], line, sep);
          if (0 > idx[i]) resolved = 0;
        }
        for (i = 0; i < noutputs && 0 != resolved; ++i) {
          idx[ninputs + i] = internal_libxs_predict_resolve_col(
            output_tokens[i], line, sep);
          if (0 > idx[ninputs + i]) {
            idx[ninputs + i] = ncols + nextra;
            ++nextra;
          }
        }
      }
      if (0 == resolved) {
        rewind(file);
        for (i = 0; i < ninputs; ++i) {
          idx[i] = (NULL != inputs)
            ? (int)strtol(input_tokens[i], NULL, 10) : i;
        }
        for (i = 0; i < noutputs; ++i) {
          idx[ninputs + i] = (NULL != outputs)
            ? (int)strtol(output_tokens[i], NULL, 10) : ninputs + i;
        }
      }
    }
    else if (0 != resolved) {
      if (NULL == sep) sep = ";";
      for (i = 0; i < ninputs; ++i) {
        idx[i] = (NULL != inputs)
          ? (int)strtol(input_tokens[i], NULL, 10) : i;
      }
      for (i = 0; i < noutputs; ++i) {
        idx[ninputs + i] = (NULL != outputs)
          ? (int)strtol(output_tokens[i], NULL, 10) : ninputs + i;
      }
    }
    if (NULL != delim_out) {
      *delim_out = sep[0];
    }
    while (NULL != fgets(line, (int)sizeof(line), file)) {
      size_t len;
      if ('#' == line[0]) continue;
      len = strlen(line);
      while (0 < len && ('\n' == line[len - 1] || '\r' == line[len - 1])) line[--len] = '\0';
      if (0 != internal_libxs_predict_parse_row(line, sep, idx, ninputs, inp)
        && 0 != internal_libxs_predict_parse_row(line, sep, idx + ninputs, noutputs, outp))
      {
        if (EXIT_SUCCESS == libxs_predict_push(NULL, model, inp, outp)) {
          ++result;
        }
      }
    }
    fclose(file);
  }
  return result;
}
