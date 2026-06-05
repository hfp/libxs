#!/usr/bin/env python3
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
"""MkDocs hooks for building FORD (Fortran) docs and mkslides presentations.

Both tools generate standalone HTML that does not need processing by MkDocs.
Running them in on_post_build (after MkDocs has finished) avoids writing into
docs_dir during the build, which would otherwise trigger a redundant rebuild.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("mkdocs.hooks")

FORD_PROJECT = "libxs_fortran.md"
FORD_OUTPUT = "fortran"
SLIDES_DIRS = ("ozaki", "predict")

_built = False


def _build_ford(docs_dir, site_dir):
    """Build FORD Fortran documentation into the site output."""
    project_path = os.path.join(docs_dir, FORD_PROJECT)

    try:
        import ford
    except ImportError:
        log.warning("FORD not installed, skipping Fortran docs")
        return

    if not os.path.isfile(project_path):
        log.warning("FORD project file not found: %s", project_path)
        return

    log.info("Building FORD documentation from %s", FORD_PROJECT)
    try:
        with open(project_path, "r") as f:
            proj_docs = f.read()
        proj_docs, proj_data = ford.load_settings(
            proj_docs, docs_dir, FORD_PROJECT
        )
        proj_data, proj_docs = ford.parse_arguments(
            {}, proj_docs, proj_data, docs_dir
        )
        if proj_data and proj_docs:
            proj_data.output_dir = Path(site_dir) / FORD_OUTPUT
            ford.main(proj_data, proj_docs)
            log.info("FORD documentation built successfully")
        else:
            log.warning("FORD returned empty settings or docs")
    except Exception:
        log.exception("FORD build failed")


def _build_slides(docs_dir, site_dir):
    """Build mkslides presentations into the site output."""
    if not shutil.which("mkslides"):
        log.warning("mkslides not installed, skipping slides")
        return

    for slides_dir in SLIDES_DIRS:
        slides_src = os.path.join(docs_dir, slides_dir)
        slides_dst = os.path.join(site_dir, slides_dir)

        if not os.path.isdir(slides_src):
            continue

        log.info("Building slides from %s into %s", slides_src, slides_dst)
        try:
            result = subprocess.run(
                ["mkslides", "build", slides_src, "-d", slides_dst],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                log.info("Slides built successfully")
            else:
                log.warning(
                    "mkslides exited with %d: %s",
                    result.returncode, result.stderr.strip(),
                )
        except Exception:
            log.exception("mkslides build failed")


def on_post_build(config):
    """Build FORD and mkslides output into the MkDocs site directory."""
    global _built
    if _built:
        return
    _built = True
    docs_dir = config.docs_dir if config else "documentation"
    site_dir = config.site_dir if config else "site"
    _build_ford(docs_dir, site_dir)
    _build_slides(docs_dir, site_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from types import SimpleNamespace

    config = SimpleNamespace(docs_dir="documentation", site_dir="site")
    on_post_build(config)
