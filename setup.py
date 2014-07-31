#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause

from distutils.core import setup

setup(
    name="mempamal",
    version="0.1.0",
    description="MEMPAMAL: Means for EMbarrassingly PArallel MAchine Learning",
    author="Benoit Da Mota",
    author_email="damota.benoit@gmail.com",
    packages=["mempamal", "mempamal.datasets"],
    long_description="""A set of python helpers to produce and run embarrassingly parallel machine learning workflows""",
    classifiers=[])
