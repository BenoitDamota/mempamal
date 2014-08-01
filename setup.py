#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
import setuptools

setuptools.setup(
    name="mempamal",
    version="0.1.0",
    description="MEMPAMAL: Means for EMbarrassingly PArallel MAchine Learning",
    author="Benoit Da Mota",
    author_email="damota.benoit@gmail.com",
    license='BSD 3-clause',
    url='https://github.com/BenoitDamota/mempamal',
    packages=["mempamal", "mempamal.datasets", "mempamal.scripts"],
    long_description="""A set of python helpers to produce and run
    embarrassingly parallel machine learning workflows""",
    install_requires=['numpy', 'scikit-learn'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Unix',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules']
    )
