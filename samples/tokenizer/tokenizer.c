#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOKEN_BYTES 8
#define TOKEN_PAYLOAD_BYTES 7
#define TOKEN_LEN_MASK 0x07u
#define TOKEN_FLAG_BREAK 0x08u
#define TOKEN_FLAG_COPY 0x80u
#define TOKEN_MIN_COPY 3
#define TOKEN_MAX_COPY 7
#define TOKEN_MAX_DISTANCE 65535u
#define DEFAULT_INPUT "token tokenization tokenizer token"

typedef struct token64_t {
  unsigned char raw[TOKEN_BYTES];
} token64_t;

typedef struct token_stream_t {
  token64_t* data;
  size_t size;
  size_t capacity;
} token_stream_t;

static int stream_reserve(token_stream_t* stream, size_t capacity);
static int stream_push(token_stream_t* stream, const token64_t* token);
static void stream_destroy(token_stream_t* stream);
static int read_stdin(unsigned char** data, size_t* size);
static int join_args(int argc, char* argv[], unsigned char** data, size_t* size);
static int tokenize_bytes(const unsigned char* text, size_t size,
  token_stream_t* stream);
static int decode_bytes(const token_stream_t* stream,
  unsigned char** text, size_t* size);
static void print_tokens(const token_stream_t* stream);
static size_t next_codepoint_size(const unsigned char* text, size_t size,
  size_t pos);
static int prefer_break(unsigned char prev, unsigned char next);
static size_t choose_literal_len(const unsigned char* text, size_t size,
  size_t pos);
static int find_copy_match(const unsigned char* text, size_t size, size_t pos,
  size_t* best_len, size_t* best_dist, int* prefer_break_flag);
static void init_literal(token64_t* token, const unsigned char* text,
  size_t length, int prefer_break_flag);
static void init_copy(token64_t* token, size_t length, size_t distance,
  int prefer_break_flag);
static int token_is_copy(const token64_t* token);
static size_t token_len(const token64_t* token);


int main(int argc, char* argv[])
{
  token_stream_t stream;
  unsigned char* input = NULL;
  unsigned char* decoded = NULL;
  size_t input_size = 0, decoded_size = 0;
  int result = EXIT_FAILURE;

  stream.data = NULL;
  stream.size = 0;
  stream.capacity = 0;

  if (1 < argc && 0 == strcmp(argv[1], "-")) {
    result = read_stdin(&input, &input_size);
  }
  else if (1 < argc) {
    result = join_args(argc, argv, &input, &input_size);
  }
  else {
    input_size = strlen(DEFAULT_INPUT);
    input = (unsigned char*)malloc(input_size + 1);
    if (NULL != input) {
      memcpy(input, DEFAULT_INPUT, input_size + 1);
      result = EXIT_SUCCESS;
    }
  }

  if (EXIT_SUCCESS == result) {
    result = tokenize_bytes(input, input_size, &stream);
  }
  if (EXIT_SUCCESS == result) {
    result = decode_bytes(&stream, &decoded, &decoded_size);
  }
  if (EXIT_SUCCESS == result) {
    printf("input-bytes: %lu\n", (unsigned long)input_size);
    printf("tokens: %lu\n", (unsigned long)stream.size);
    print_tokens(&stream);
    printf("decoded: %.*s\n", (int)decoded_size, (const char*)decoded);
    if (decoded_size == input_size && 0 == memcmp(decoded, input, input_size)) {
      printf("roundtrip: ok\n");
      result = EXIT_SUCCESS;
    }
    else {
      fprintf(stderr, "roundtrip: mismatch\n");
      result = EXIT_FAILURE;
    }
  }
  else {
    fprintf(stderr, "tokenizer64: failed\n");
  }

  free(decoded);
  free(input);
  stream_destroy(&stream);
  return result;
}


static int stream_reserve(token_stream_t* stream, size_t capacity)
{
  int result = EXIT_FAILURE;
  if (NULL != stream) {
    if (capacity <= stream->capacity) result = EXIT_SUCCESS;
    else {
      token64_t* data = (token64_t*)realloc(stream->data,
        capacity * sizeof(token64_t));
      if (NULL != data) {
        stream->data = data;
        stream->capacity = capacity;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static int stream_push(token_stream_t* stream, const token64_t* token)
{
  int result = EXIT_FAILURE;
  if (NULL != stream && NULL != token) {
    const size_t capacity = (0 != stream->capacity) ? (2 * stream->capacity) : 16;
    if (stream->size == stream->capacity) result = stream_reserve(stream, capacity);
    else result = EXIT_SUCCESS;
    if (EXIT_SUCCESS == result) {
      stream->data[stream->size] = *token;
      ++stream->size;
    }
  }
  return result;
}


static void stream_destroy(token_stream_t* stream)
{
  if (NULL != stream) {
    free(stream->data);
    stream->data = NULL;
    stream->size = 0;
    stream->capacity = 0;
  }
}


static int read_stdin(unsigned char** data, size_t* size)
{
  int result = EXIT_FAILURE;
  unsigned char* buffer = NULL;
  size_t used = 0, capacity = 0;
  int ch = 0;
  if (NULL != data && NULL != size) {
    while (EOF != (ch = getchar())) {
      if (used == capacity) {
        size_t next_capacity = (0 != capacity) ? (2 * capacity) : 64;
        unsigned char* next = (unsigned char*)realloc(buffer, next_capacity + 1);
        if (NULL == next) {
          free(buffer);
          buffer = NULL;
          used = 0;
          capacity = 0;
          break;
        }
        buffer = next;
        capacity = next_capacity;
      }
      buffer[used] = (unsigned char)ch;
      ++used;
    }
    if (NULL != buffer || 0 == used) {
      if (NULL == buffer) buffer = (unsigned char*)malloc(1);
      if (NULL != buffer) {
        buffer[used] = 0;
        *data = buffer;
        *size = used;
        result = EXIT_SUCCESS;
      }
    }
  }
  return result;
}


static int join_args(int argc, char* argv[], unsigned char** data, size_t* size)
{
  int result = EXIT_FAILURE;
  size_t total = 0;
  int i;
  unsigned char* buffer;
  if (NULL != data && NULL != size && 1 < argc) {
    for (i = 1; i < argc; ++i) total += strlen(argv[i]) + ((1 < i) ? 1 : 0);
    buffer = (unsigned char*)malloc(total + 1);
    if (NULL != buffer) {
      size_t offset = 0;
      for (i = 1; i < argc; ++i) {
        size_t length = strlen(argv[i]);
        if (1 < i) {
          buffer[offset] = ' ';
          ++offset;
        }
        memcpy(buffer + offset, argv[i], length);
        offset += length;
      }
      buffer[offset] = 0;
      *data = buffer;
      *size = total;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static int tokenize_bytes(const unsigned char* text, size_t size,
  token_stream_t* stream)
{
  int result = EXIT_SUCCESS;
  size_t pos = 0;
  if (NULL == text || NULL == stream) result = EXIT_FAILURE;
  while (EXIT_SUCCESS == result && pos < size) {
    token64_t token;
    size_t best_len = 0, best_dist = 0;
    int prefer_break_flag = 0;
    if (0 != find_copy_match(text, size, pos,
      &best_len, &best_dist, &prefer_break_flag))
    {
      init_copy(&token, best_len, best_dist, prefer_break_flag);
      result = stream_push(stream, &token);
      pos += best_len;
    }
    else {
      best_len = choose_literal_len(text, size, pos);
      if (0 == best_len) best_len = 1;
      if (pos + best_len < size) {
        prefer_break_flag = prefer_break(text[pos + best_len - 1], text[pos + best_len]);
      }
      init_literal(&token, text + pos, best_len, prefer_break_flag);
      result = stream_push(stream, &token);
      pos += best_len;
    }
  }
  return result;
}


static int decode_bytes(const token_stream_t* stream,
  unsigned char** text, size_t* size)
{
  int result = EXIT_FAILURE;
  size_t total = 0;
  size_t i;
  unsigned char* buffer;
  if (NULL != stream && NULL != text && NULL != size) {
    for (i = 0; i < stream->size; ++i) total += token_len(stream->data + i);
    buffer = (unsigned char*)malloc(total + 1);
    if (NULL != buffer) {
      size_t offset = 0;
      for (i = 0; i < stream->size; ++i) {
        const token64_t* token = stream->data + i;
        const size_t length = token_len(token);
        if (0 != token_is_copy(token)) {
          const size_t distance = (size_t)token->raw[1]
            | ((size_t)token->raw[2] << 8);
          size_t j;
          if (0 == distance || distance > offset) {
            free(buffer);
            return EXIT_FAILURE;
          }
          for (j = 0; j < length; ++j) {
            buffer[offset + j] = buffer[offset - distance + j];
          }
        }
        else {
          memcpy(buffer + offset, token->raw + 1, length);
        }
        offset += length;
      }
      buffer[total] = 0;
      *text = buffer;
      *size = total;
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


static void print_tokens(const token_stream_t* stream)
{
  size_t i;
  if (NULL != stream) {
    for (i = 0; i < stream->size; ++i) {
      const token64_t* token = stream->data + i;
      const size_t length = token_len(token);
      const int prefer_break_flag = (0 != (token->raw[0] & TOKEN_FLAG_BREAK));
      if (0 != token_is_copy(token)) {
        const unsigned int distance = (unsigned int)token->raw[1]
          | ((unsigned int)token->raw[2] << 8);
        printf("  %02lu copy    len=%lu dist=%u break=%d\n",
          (unsigned long)i, (unsigned long)length, distance, prefer_break_flag);
      }
      else {
        size_t j;
        printf("  %02lu literal len=%lu break=%d text=\"",
          (unsigned long)i, (unsigned long)length, prefer_break_flag);
        for (j = 0; j < length; ++j) {
          const unsigned char ch = token->raw[1 + j];
          if (isprint(ch) && '"' != ch && '\\' != ch) putchar(ch);
          else printf("\\x%02X", (unsigned int)ch);
        }
        printf("\"\n");
      }
    }
  }
}


static size_t next_codepoint_size(const unsigned char* text, size_t size,
  size_t pos)
{
  size_t result = 1;
  if (NULL != text && pos < size) {
    const unsigned char lead = text[pos];
    if (0xC0u <= lead && lead < 0xE0u) result = 2;
    else if (0xE0u <= lead && lead < 0xF0u) result = 3;
    else if (0xF0u <= lead && lead < 0xF8u) result = 4;
    if (size - pos < result) result = size - pos;
  }
  return result;
}


static int prefer_break(unsigned char prev, unsigned char next)
{
  int result = 0;
  const int prev_space = (0 != isspace(prev));
  const int next_space = (0 != isspace(next));
  const int prev_punct = (0 != ispunct(prev));
  const int next_punct = (0 != ispunct(next));
  if (prev_space || next_space || prev_punct || next_punct) {
    result = 1;
  }
  else if ((0 != isalpha(prev) && 0 != isdigit(next))
    || (0 != isdigit(prev) && 0 != isalpha(next)))
  {
    result = 1;
  }
  else if (0 != islower(prev) && 0 != isupper(next)) {
    result = 1;
  }
  return result;
}


static size_t choose_literal_len(const unsigned char* text, size_t size,
  size_t pos)
{
  size_t offset = 0;
  size_t best = 0;
  if (NULL != text && pos < size) {
    while (offset < TOKEN_PAYLOAD_BYTES && pos + offset < size) {
      const size_t step = next_codepoint_size(text, size, pos + offset);
      if (offset + step > TOKEN_PAYLOAD_BYTES) break;
      offset += step;
      best = offset;
      if (pos + offset < size && 1 == step) {
        if (0 != prefer_break(text[pos + offset - 1], text[pos + offset])) break;
      }
      if (1 == step && (0 != isspace(text[pos + offset - 1])
        || 0 != ispunct(text[pos + offset - 1])))
      {
        break;
      }
    }
  }
  return best;
}


static int find_copy_match(const unsigned char* text, size_t size, size_t pos,
  size_t* best_len, size_t* best_dist, int* prefer_break_flag)
{
  int result = 0;
  size_t distance;
  size_t max_distance;
  if (NULL == text || NULL == best_len || NULL == best_dist
    || NULL == prefer_break_flag || pos >= size)
  {
    return 0;
  }
  *best_len = 0;
  *best_dist = 0;
  *prefer_break_flag = 0;
  max_distance = (pos < TOKEN_MAX_DISTANCE) ? pos : TOKEN_MAX_DISTANCE;
  for (distance = 1; distance <= max_distance; ++distance) {
    size_t current_len = 0;
    while (current_len < TOKEN_MAX_COPY && pos + current_len < size
      && text[pos - distance + current_len] == text[pos + current_len])
    {
      ++current_len;
      if (pos + current_len < size
        && 1 < next_codepoint_size(text, size, pos + current_len - 1)
        && pos + current_len < size
        && 0x80u <= text[pos + current_len] && text[pos + current_len] < 0xC0u)
      {
        --current_len;
        break;
      }
    }
    while (TOKEN_MIN_COPY <= current_len
      && pos + current_len < size
      && 0x80u <= text[pos + current_len] && text[pos + current_len] < 0xC0u)
    {
      --current_len;
    }
    if (current_len >= TOKEN_MIN_COPY && current_len > *best_len) {
      *best_len = current_len;
      *best_dist = distance;
      if (pos + current_len < size) {
        *prefer_break_flag = prefer_break(text[pos + current_len - 1],
          text[pos + current_len]);
      }
      result = 1;
      if (TOKEN_MAX_COPY == current_len) break;
    }
  }
  return result;
}


static void init_literal(token64_t* token, const unsigned char* text,
  size_t length, int prefer_break_flag)
{
  if (NULL != token) {
    memset(token->raw, 0, sizeof(token->raw));
    token->raw[0] = (unsigned char)(length & TOKEN_LEN_MASK);
    if (0 != prefer_break_flag) token->raw[0] |= TOKEN_FLAG_BREAK;
    if (NULL != text && 0 != length) memcpy(token->raw + 1, text, length);
  }
}


static void init_copy(token64_t* token, size_t length, size_t distance,
  int prefer_break_flag)
{
  if (NULL != token) {
    memset(token->raw, 0, sizeof(token->raw));
    token->raw[0] = (unsigned char)(TOKEN_FLAG_COPY | (length & TOKEN_LEN_MASK));
    if (0 != prefer_break_flag) token->raw[0] |= TOKEN_FLAG_BREAK;
    token->raw[1] = (unsigned char)(distance & 0xFFu);
    token->raw[2] = (unsigned char)((distance >> 8) & 0xFFu);
  }
}


static int token_is_copy(const token64_t* token)
{
  int result = 0;
  if (NULL != token && 0 != (token->raw[0] & TOKEN_FLAG_COPY)) result = 1;
  return result;
}


static size_t token_len(const token64_t* token)
{
  size_t result = 0;
  if (NULL != token) result = (size_t)(token->raw[0] & TOKEN_LEN_MASK);
  return result;
}
