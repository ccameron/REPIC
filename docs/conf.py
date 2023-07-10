#!/usr/bin/env python3
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath('.'))


def convert_readme_to_rest():
    """
    Converts GitHub README (MarkDown) to Sphinx reST format
    """
    #   myst-parser doesn't include titles in toctree nor properly handles images
    #   steps:
    #       1) converts from MarkDown to reST
    #       2) remove redundant REPIC header
    subprocess.run(["""pandoc ../README.md -f gfm -t rst -s -o readme.rst"""],
                   shell=True)
    #   ; perl -i -p0e 's/REPIC\n=====\n\n//se' readme.rst
    #   update local REPIC schematic image
    #   steps:
    #       1) cp imgs/ to sphinx_docs/
    #       2) update overview img reference in readme.rst
    #       3) update icon img reference in readme.rst
    subprocess.run(["""cp -R ../imgs .;
echo ".. image:: imgs/repic_overview.png
  :width: 90%
  :alt: REPIC schematic
  :align: center
|" > tmp.txt;
perl -i -p0e 's/\.\. raw:: html\n\n   <p align=(.|\n)*\.png">\n   <\/p>\n/`cat tmp.txt`/se' readme.rst;
echo ".. image:: imgs/repic_icon.png
  :width: 25%
  :alt: REPIC icon" > tmp.txt;
 perl -i -p0e 's/\.\. raw:: html\n\n   <img width=.*\.png">\n/`cat tmp.txt`/se' readme.rst;
rm tmp.txt"""],
                   shell=True)


def get_version():
    """
    Pulls REPIC version from Git files
    """
    with open("../repic/__version__.py", 'rt') as f:
        version = f.readlines()[-1].split()[-1].replace('"', '')

    return version


convert_readme_to_rest()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'REPIC'
copyright = '2023, Christopher JF Cameron'
author = 'Christopher JF Cameron'
release = get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinx.ext.napoleon'
]
autoapi_dirs = ['../repic']
autoapi_type = "python"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
}
html_static_path = ['_static']
