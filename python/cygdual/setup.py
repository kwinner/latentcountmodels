#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

libgdual_dir = '/Users/sheldon/projects/latentcountmodels/c'

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cygdual",
                             sources=["cygdual.pyx", libgdual_dir + "/libgdual.c"],
                             include_dirs=[numpy.get_include()])],
)
