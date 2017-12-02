from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os.path import realpath, dirname
import numpy

libgdual_dir = dirname(dirname(dirname(realpath(__file__)))) + "/c"

setup(
    name='lsgdual',
    version='',
    packages=['lsgdual'],
    package_dir={'': ''},
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cygdual",
                             sources=["cygdual.pyx", libgdual_dir + "/gdual.c"],
                             include_dirs=[numpy.get_include(), libgdual_dir])],
    url='',
    license='',
    author='Kevin Winner',
    author_email='kwinner@cs.umass.edu',
    description='Package for manipulating generalized dual numbers using a log-sign representation.'
)
