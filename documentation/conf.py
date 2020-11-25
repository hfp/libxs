###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXS library.                                     #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxs/                              #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
import sphinx_rtd_theme
import os

project = 'LIBXS'
copyright = '2009-2020, Intel Corporation.'
author = 'Intel Corporation'
user = os.environ.get('USER')

extensions = [
    #"recommonmark",
    "m2r2"
]

master_doc = "index"
source_suffix = [
    ".rst",
    #".md"
]

exclude_patterns = [
    "*-" + user + "-*.md",
    "Thumbs.db",
    ".DS_Store",
    "_build"
]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2
}
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["../.theme"]

templates_path = ["_templates"]
pygments_style = "sphinx"

language = None
