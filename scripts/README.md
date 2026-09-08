# Scripts

## Build Support

**tool_source.sh** — Generate `libxs_source.h` (header-only amalgamation). Called by `make headers`.

**tool_version.sh** — Extract version from Git tags. Generates `libxs_version.h`.

**libxs.pc.in** — Template for the installed `.pc` file.

## Code Quality

**tool_analyze.sh** — Run cppcheck on `src/`.

**tool_checkabi.sh** — Compare exported symbols against a reference.

**tool_clangformat.sh** — Run clang-format and shellcheck.

**tool_normalize.sh** — Strip trailing whitespace, fix line endings.

## Documentation

**md2pptx.py** — Convert Markdown to PowerPoint. Splits on `---` rules or `##` headings (auto-detected). Requires `python-pptx` and `lxml`.

## Development

**tool_cpuinfo.sh** — Print CPU topology (sockets, cores, threads).

**tool_checkenvars.sh** — List environment variables used in the source tree (`--list`), or check that the prefixed ones are documented.

**tool_pexec.sh** — Parallel command execution with CPU affinity (`-h` for options).

**tool_gitaddforks.sh** — Add GitHub forks as Git remotes.

**tool_gitprune.sh** — Aggressive Git housekeeping (gc, fsck, repack).
