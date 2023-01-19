# Overview

LIBXS's collection of [scripts](https://github.com/hfp/libxs/tree/main/scripts) consists of Python and Bash scripts falling into two categories:

* Configuration scripts
* Development tools

Scripts related to configuring LIBXS are distributed with source code archives. Development tools mostly for software development purpose and are (indirectly) used by contributors, but some scripts are distributed by source code archives as well. The latter are mostly related to running tests (indirectly used by upstream maintainers, e.g., of Linux distributions).

## Development Tools

* `tool_analyze.sh`: Runs compiler based static analysis based on Clang or GCC.
* `tool_changelog.sh`: Rephrases the history of LIBXS's checked-out repository to consist as a changelog grouped by contributors.
* `tool_checkabi.sh`: Extracts exported/visible functions and other symbols (public interface) from built LIBXS and compares against a recorded state. The purpose is to acknowledge and confirm for instance removed functionality (compatibility). This includes functions only exported to allow interaction between LIBXS's different libraries. However, it currently falls short of recognizing changes to the signature of functions (arguments).
* `tool_cpuinfo.sh`: Informs about the system the script is running on, i.e., the number of CPU sockets (packages), the number of CPU cores, the number of CPU threads, the number of threads per CPU core (SMT), and the number of NUMA domains. The script is mainly used to parallelize tests during development. However, this script is distributed because test related scripts are not only of contributor's interest (`tool_test.sh`).
* `tool_envrestore.sh`: Restores environment variables when running tests (`tool_test.sh`).
* `tool_getenvars.sh`: Attempts to collect environment variables used in LIBXS's code base (`getenv`). This script is distributed.
* `tool_gitaddforks.sh`: Collects forks of LIBXS and adds them as Git-remotes, which can foster collaboration (development).
* `tool_gitauthors.sh`: Collects authors of LIBXS from history of the checked-out repository.
* `tool_gitprune.sh`: Performs garbage collection of the checked-out repository (`.git folder`). The script does not remove files, i.e., it does not run `git clean`.
* `tool_inspector.sh`: Wrapper script when running a binary to detect potential memory leaks or data races.
* `tool_normalize.sh`: Detects simple code patters banned from LIBXS's source code.
* `tool_perflog.sh`: Extracts performance information produced by certain examples (driver code), e.g., [LIBXS-DNN tests](https://github.com/hfp/libxs-dnn/tree/main/tests).
* `tool_pexec.sh`: Reads standard input and attempts to execute every line (command) on a per CPU-core basis, which can help to parallelize tests on a per-process basis.
* `tool_report.py`: Core developer team can collect a performance history of a certain CI-collection (Buildkite pipeline).
* `tool_scan.sh`: Core developer team can scan the repository based on a list of keywords.
* `tool_test.sh`: 
* `tool_version.sh`: Determines LIBXS's version from the history of the checked-out repository (Git). With respect to LIBXS's patch version, the information is not fully accurate given a non-linear history.
