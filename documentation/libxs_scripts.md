# Scripts

`tool_normalize.sh`, `tool_checkstruct.py`, and `tool_checkenvars.sh` are policy files: they are maintained in LIBXS and copied into dependent projects by `make documentation`, hence a change belongs in LIBXS.

## Build Support

**tool_source.sh** — Generate `libxs_source.h` (header-only amalgamation). Called by `make headers`.

**tool_version.sh** — Print the version: `version.txt`, plus the branch and the number of commits since the last release tag (Git). With `-1`, it emits `libxs_version.h`.

**tool_pkgversion.sh** — Update the RPM spec and the Debian changelog to the version in `version.txt`. Called by `make`.

**libxs.pc.in** — Template for the installed `.pc` file.

**libxsConfig.cmake**, **libxsConfigVersion.cmake.in** — CMake package files installed by `make`. **libxsConfig.cmake.in** — the same for the CMake build.

**debian/**, **fedora/libxs.spec** — Debian and RPM packaging.

## Code Quality

**tool_analyze.sh** — Run cppcheck on `src/`.

**tool_checkabi.sh** — Compare exported symbols against a reference.

**tool_clangformat.sh** — Run `clang-format`, or a newer `clang-format-N` if one is found, with the given arguments.

**tool_normalize.sh** — Run the rules of `.pre-commit-config.yaml` on the tree, or on the given directory. `--install` installs the Git hook.

**tool_checkstruct.py** — Check the structural rules of the contribution policy (`--list` names the checks). Open findings are listed in `tool_checkstruct.todo`.

## Documentation

**deck2typst.py** — Render a slide deck (`documentation/<topic>/index.md`, reveal.js Markdown as `make mkslides` serves it) to a landscape PDF with selectable text, through Pandoc and Typst. Figures come from the deck's `assets/`; a slide too dense to fit is scaled down and reported. Requires `pandoc` and `pip install typst` (or a `typst` executable via `--typst`).

## Development

**tool_cpuinfo.sh** — Print CPU topology (sockets, cores, threads).

**tool_checkenvars.sh** — List environment variables used in the source tree (`--list`), or check that the prefixed ones are documented. Names not documented yet are listed in `tool_checkenvars.todo`.

**tool_pexec.sh** — Parallel command execution with CPU affinity (`-h` for options).

**tool_gitaddforks.sh** — Add GitHub forks as Git remotes.

**tool_gitprune.sh** — Aggressive Git housekeeping (gc, fsck, repack).
