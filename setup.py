#!/usr/bin/env python
# Author: Benoit Da Mota <damota.benoit@gmail.com>
#
# License: BSD 3 clause
import os
import setuptools

setuptools.setup(
    name="mempamal",
    version="0.1.1",
    description="MEMPAMAL: Means for EMbarrassingly PArallel MAchine Learning",
    author="Benoit Da Mota",
    author_email="damota.benoit@gmail.com",
    license='BSD 3-clause',
    url='https://github.com/BenoitDamota/mempamal',
    packages=["mempamal", "mempamal.datasets", "mempamal.scripts"],
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
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
