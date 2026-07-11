#include <libxs/libxs_token.h>

#define DEFAULT_INPUT "Who is Alice? Alice saw 123."

static int read_stdin(unsigned char** data, size_t* size);
static int join_args(int argc, char* argv[], unsigned char** data, size_t* size);
static void print_tokens(const libxs_token_stream_t* stream);
static void print_tokens_with_text(const libxs_token_stream_t* stream,
  const libxs_lexicon_t* lexicon);


int main(int argc, char* argv[])
{
  libxs_token_stream_t stream;
  libxs_lexrule_t lexrules[96];
  libxs_lexicon_t* lexicon = NULL;
  unsigned char* input = NULL;
  size_t input_size = 0;
  int lexrule_count = 0;
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
    lexicon = libxs_lexicon_create();
    lexrule_count = libxs_lexrule_defaults(lexrules, 96);
    if (NULL == lexicon || lexrule_count <= 0) result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    result = libxs_token_stream_encode(lexicon, &stream,
      input, input_size, lexrules, lexrule_count, 1);
  }
  if (EXIT_SUCCESS == result) {
    printf("input-bytes: %lu\n", (unsigned long)input_size);
    printf("tokens: %lu\n", (unsigned long)stream.size);
    print_tokens(&stream);
    printf("vocab: %u\n", libxs_lexicon_size(lexicon));
    print_tokens_with_text(&stream, lexicon);
  }
  else {
    fprintf(stderr, "tokenizer: failed\n");
  }

  free(input);
  libxs_lexicon_destroy(lexicon);
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
      printf("  %02lu id=%u len=%lu flags=0x%04X break=%d sentence=%d\n",
        (unsigned long)i, info.id, (unsigned long)info.length,
        info.flags, info.has_break, info.is_sentence);
    }
  }
}


static void print_tokens_with_text(const libxs_token_stream_t* stream,
  const libxs_lexicon_t* lexicon)
{
  size_t token_pos;
  if (NULL != stream && NULL != lexicon) {
    for (token_pos = 0; token_pos < stream->size; ++token_pos) {
      const libxs_token_t* token = stream->data + token_pos;
      int length = 0;
      unsigned int flags = 0;
      const char* text = libxs_lexicon_text(lexicon, token->id,
        &length, &flags);
      printf("  %02lu id=%u flags=0x%04X len=%u text=\"%.*s\"\n",
        (unsigned long)token_pos, token->id,
        (unsigned int)token->flags, (unsigned int)token->length, length,
        (NULL != text) ? text : "");
    }
  }
}
