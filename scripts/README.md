# Overview

LIBXS's collection of [scripts](https://github.com/hfp/libxs/tree/main/scripts) consists of Python and Bash scripts falling into two categories:

* Configuration scripts
* Development tools

Scripts related to configuring LIBXS are distributed with source code archives. Development tools mostly for software development purpose and are (indirectly) used by contributors, but some scripts are distributed by source code archives as well. The latter are mostly related to running tests (indirectly used by upstream maintainers, e.g., of Linux distributions).

## Development Tools

### Parallel Execution

The script `tool_pexec.sh` can execute commands read from standard input (see `-h` or `--help`). The execution may be concurrent on a per-command basis. The level of parallelism is determined automatically but can be adjusted (oversubscription, nested parallelism). By default, a separate logfile is written for every executed command which can be disabled (`-o /dev/null`). File I/O can become a bottleneck on distributed filesystems (e.g., NFS), or generally hinders nested parallelism (`-o /dev/null -k`).

Every line of standard input denotes a separate command:

```bash
seq 100 | xargs -I{} echo "echo \"{}\"" \
        | tool_pexec.sh
```

The script considers an allow-list which permits certain error codes. Allow-lists can be automatically generated (`-u`).

#### Performance Report

The script `tool_report.py` collects performance results given in two possible formats: <span>(1)&#160;native</span> "telegram" format, and <span>(2)&#160;JSON</span> format. The script aims to avoid encoding domain knowledge. In fact, the collected information is not necessarily performance data but a time series in general. Usually, the script is not executed directly but launched using a wrapper supplying the authorization token and further adapting to the execution environment (setup):

```bash
#!/usr/bin/env bash

# authorization token
TOKEN=0123456789abcdef0123456789abcdef01234567

PYTHON=$(command -v python3)
if [ ! "${PYTHON}" ]; then
  PYTHON=$(command -v python)
fi

if [ "${PYTHON}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  NAME=$(basename "$0" .sh)
  SCRT=${NAME}.py

  if [ "${HERE}" ] && [ -e "${HERE}/${SCRT}" ]; then
    ${PYTHON} "${HERE}/${SCRT}" --token "${TOKEN}" "$@"
  elif [ "${LIBXSROOT}" ] && [ -e "${LIBXSROOT}/scripts/${SCRT}" ]; then
    ${PYTHON} "${LIBXSROOT}/scripts/${SCRT}" --token "${TOKEN}" "$@"
  elif [ "${REPOREMOTE}" ] && [ -e "${REPOREMOTE}/libxs/scripts/${SCRT}" ]; then
    ${PYTHON} "${REPOREMOTE}/libxs/scripts/${SCRT}" --token "${TOKEN}" "$@"
  else
    >&2 echo "ERROR: missing ${SCRT}!"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
```

The following flow is established:

1. Connect to a specified pipeline (online) or load a logfile directly (offline).
2. Populate an instance (JSON-block or telegram) under a "build number", "category", and "case".
3. Plot "execution time" over the history of build numbers.

There are several command line options to customize each of the above steps (`--help` or `-h`):

* To only plot data (already collected), use `-i ""` to omit a network connection.
* To query, e.g., ResNet-50 results, use `-y resnet-50` (case-insensitive).
* Multiple results can be combined, i.e., use `-y` (space-separated words).
* To query exactly (single results) use `-x` in addition to `-y`.
* To limit and select a specific "category" (instead of all), use `-s`.
* Select exactly using `-z`, e.g., `-z -s "clx"` (omits, e.g., "clxap").
* Create a PDF (vector graphics have infinite resolution), use `-g myreport.pdf`.
* Adjust pixel resolution, aspect ratio, or density, use `-d 1200x800`.

The level of verbosity (`-v`) can be adjusted (0: quiet, 1: automation, 2: progress). Default verbosity shows progress (downloading results) whereas "automation" allows to further automate reports, e.g., get the filename of the generated plot (errors are generally printed to `stderr`). Loading a logfile into the database directly can serve two purposes: <span>(1)&#160;debugging</span> the supported format like "telegram" or JSON, and <span>(2)&#160;offline</span> operation. The latter can be also useful if for instance a CI-agents produces a log, i.e., it can load into the database right away. Command line options also allow for "exact placement" (`-j`) by specifying the build number supposed to take the loaded data (data is appended by default, i.e., it is assumed to be a new build or the build number is incremented). In general, data is not duplicated underneath a build of the category or the actual data matches an existing entry.

Examples (omit `-i ""` if downloading results is desired):

* Plot ResNet-50 results from CI-pipeline "tpp-libxs" for "clx" systems:  
  `scripts/tool_report.sh -p tpp-libxs -i "" -y "resnet-50" -z -s "clx"`.
* Like above request, but only FP32 results:  
  `scripts/tool_report.sh -p tpp-libxs -i "" -x -y "ResNet-50 (fwd, mb=1, f32)" -z -s "clx"`.
* Like above request, but alternatively ("all" operator is also default):  
  `scripts/tool_report.sh -p tpp-libxs -i "" -u "all" -y "resnet f32" -z -s "clx"`.
* Plot ResNet-50 results from CI-pipeline "tpp-plaidml":  
  `scripts/tool_report.sh -p tpp-plaidml -i "" -r "duration_per_example,1000,ms"`
* Plot "GFLOP/s" for "conv2d_odd_med" from CI-pipeline "tpp-plaidml":  
  `scripts/tool_report.sh -p tpp-plaidml -i "" -y "conv2d_odd_med" -r "gflop"`
* Plot "tpp-mlir" pipeline (reference benchmarks):  
  `scripts/tool_report.sh -p tpp-mlir -i "" -y "" -r "ref"`
* Plot "tpp-mlir" pipeline (MLIR benchmarks):  
  `scripts/tool_report.sh -p tpp-mlir -i "" -y "" -r "mlir"`
* Plot "tpp-mlir" pipeline (MLIR benchmarks without "simple_copy"):  
  `scripts/tool_report.sh -p tpp-mlir -i "" -u "not" -y "simply_copy" -r "mlir"`
