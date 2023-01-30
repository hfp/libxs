# Overview

LIBXS's collection of [scripts](https://github.com/hfp/libxs/tree/main/scripts) consists of Python and Bash scripts falling into two categories:

* Configuration scripts
* Development tools

Scripts related to configuring LIBXS are distributed with source code archives. Development tools mostly for software development purpose and are (indirectly) used by contributors, but some scripts are distributed by source code archives as well. The latter are mostly related to running tests (indirectly used by upstream maintainers, e.g., of Linux distributions).

## Development Tools

### Parallel Execution

The script `tool_pexec.sh` allows to execute commands read from standard input (see `-h` or `--help`). The execution may be concurrent on a per-command basis. The level of parallelism is determined automatically but can be adjusted (oversubscription, nested parallelism). By default, a separate logfile is written for every executed command which can be disabled (`-o /dev/null`). File I/O can become a bottleneck on distributed filesystems (e.g., NFS), or generally hinders nested parallelism (`-o /dev/null -k`).

Every line of standard input denotes a separate command:

```bash
seq 100 | xargs -I{} echo "echo \"{}\"" \
        | tool_pexec.sh
```

The script can consider an allow-list which permits certain error codes. Allow-lists can be generated automatically (`-u`).
