#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   original src: https://github.com/navdeep-G/setup.py/blob/master/setup.py
#   modified by: Christopher JF Cameron
#

import os

from setuptools import find_packages, setup

# package meta-data
NAME = "repic"
DESCRIPTION = (
    "REPIC - a consensus methodology for harnessing multiple cryo-EM particle pickers."
)
URL = "https://github.com/ccameron/REPIC"
EMAIL = "christopher.cameron@yale.edu"
AUTHOR = "Christopher JF Cameron"
REQUIRES_PYTHON = ">=3.8.16"

# required packages
REQUIRED = [
    "matplotlib>=3.2.2",
    "mrcfile>=1.4.3",
    "networkx>=2.8.4",
    "numpy>=1.24.2",
    "pandas>=2.0.2",
    "scipy>=1.10.0",
    "tqdm>=4.65.0",
]

work_dir = os.path.abspath(os.path.dirname(__file__))

# import README and use as the long-description
try:
    with open(os.path.join(work_dir, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# load the package's __version__.py module as a dictionary
about = {}
project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(work_dir, project_slug, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        ("docs", ["docs/cryolo.md", "docs/deeppicker.md", "docs/topaz.md"]),
        (
            "docs/patches/deeppicker",
            [
                "docs/patches/deeppicker/autoPick.py",
                "docs/patches/deeppicker/autoPicker.py",
                "docs/patches/deeppicker/dataLoader.py",
                "docs/patches/deeppicker/deepModel.py",
                "docs/patches/deeppicker/starReader.py",
                "docs/patches/deeppicker/train.py",
            ],
        ),
    ],
    entry_points={
        "console_scripts": ["repic=repic.main:main"],
    },
    install_requires=REQUIRED,
    license="BSD-3-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
