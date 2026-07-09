#include <libxs/libxs_token.h>

#define DEFAULT_INPUT "token tokenization tokenizer token"

static int read_stdin(unsigned char** data, size_t* size);
static int join_args(int argc, char* argv[], unsigned char** data, size_t* size);
static void print_tokens(const libxs_token_stream_t* stream);


int main(int argc, char* argv[])
{
  libxs_token_stream_t stream;
  unsigned char* input = NULL;
  unsigned char* decoded = NULL;
  size_t input_size = 0, decoded_size = 0;
  int result = EXIT_FAILURE;

  libxs_token_stream_init(&stream);

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
    result = libxs_token_stream_encode(&stream, input, input_size);
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_token_stream_decode(&stream, &decoded, &decoded_size);
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
    fprintf(stderr, "tokenizer: failed\n");
  }

  free(decoded);
  free(input);
  libxs_token_stream_release(&stream);
  return result;
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


static void print_tokens(const libxs_token_stream_t* stream)
{
  size_t i;
  if (NULL != stream) {
    for (i = 0; i < stream->size; ++i) {
      const libxs_token_t* token = stream->data + i;
      libxs_token_info_t info;
      libxs_token_info(token, &info);
      if (0 != info.is_copy) {
        printf("  %02lu copy    len=%lu dist=%u break=%d\n",
          (unsigned long)i, (unsigned long)info.length,
          (unsigned int)info.distance, info.has_break);
      }
      else {
        size_t j;
        printf("  %02lu literal len=%lu break=%d text=\"",
          (unsigned long)i, (unsigned long)info.length, info.has_break);
        for (j = 0; j < (size_t)info.length; ++j) {
          const unsigned char ch = token->raw[1 + j];
          if (isprint(ch) && '"' != ch && '\\' != ch) putchar(ch);
          else printf("\\x%02X", (unsigned int)ch);
        }
        printf("\"\n");
      }
    }
  }
}
