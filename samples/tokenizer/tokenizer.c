#include <libxs/libxs_token.h>

#define DEFAULT_INPUT "Who is Alice? Alice saw 123."

static int read_stdin(unsigned char** data, size_t* size);
static int join_args(int argc, char* argv[], unsigned char** data, size_t* size);
static void print_tokens(const libxs_token_stream_t* stream);
static int verify_stream(const unsigned char* input, size_t input_size,
  int granularity, const char* name);


int main(int argc, char* argv[])
{
  unsigned char* input = NULL;
  size_t input_size = 0;
  int result = EXIT_FAILURE;
  libxs_init();

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
    result = verify_stream(input, input_size,
      LIBXS_TOKEN_GRANULARITY_NATIVE, "native");
  }
  if (EXIT_SUCCESS == result) {
    result = verify_stream(input, input_size,
      LIBXS_TOKEN_GRANULARITY_WORD, "word");
  }
  if (EXIT_SUCCESS == result) {
    result = verify_stream(input, input_size,
      LIBXS_TOKEN_GRANULARITY_SYLLABLE, "syllable");
  }
  if (EXIT_SUCCESS != result) fprintf(stderr, "tokenizer: failed\n");

  free(input);
  libxs_finalize();
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
  size_t token_pos = 0;
  if (NULL != stream) {
    while (token_pos < stream->size) {
      size_t payload_size = 0;
      size_t ncells = libxs_token_span(stream->data, stream->size,
        token_pos, &payload_size);
      const libxs_token_t* token = stream->data + token_pos;
      if (0 == ncells) break;
      printf("  %02lu kind=%d bytes=%lu cells=%lu sentence=%d\n",
        (unsigned long)token_pos, libxs_token_kind(token),
        (unsigned long)payload_size, (unsigned long)ncells,
        libxs_token_is_sentence_end(token + ncells - 1));
      token_pos += ncells;
    }
  }
}


static int verify_stream(const unsigned char* input, size_t input_size,
  int granularity, const char* name)
{
  int result = EXIT_FAILURE;
  libxs_token_stream_t stream;
  libxs_tokenizer_t* tokenizer = libxs_tokenizer_create(granularity);
  unsigned char* decoded = NULL;
  size_t decoded_size = 0;
  libxs_token_stream_init(&stream);
  if (NULL != tokenizer
    && EXIT_SUCCESS == libxs_token_stream_encode(tokenizer, &stream,
      input, input_size)
    && EXIT_SUCCESS == libxs_token_stream_decode(&stream, &decoded,
      &decoded_size)
    && decoded_size == input_size
    && 0 == memcmp(decoded, input, input_size))
  {
    printf("%s: input-bytes=%lu cells=%lu\n", name,
      (unsigned long)input_size, (unsigned long)stream.size);
    print_tokens(&stream);
    result = EXIT_SUCCESS;
  }
  else {
    fprintf(stderr, "%s: roundtrip mismatch\n", name);
  }
  free(decoded);
  libxs_tokenizer_destroy(tokenizer);
  libxs_token_stream_release(&stream);
  return result;
}
