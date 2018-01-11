from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os.path import realpath, dirname
import numpy
import sysconfig

libgdual_dir = dirname(dirname(realpath(__file__))) + "/c"

_DEBUG = False
_DEBUG_LEVEL = 0

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Wall"]
extra_link_args = ["-lfftw3l"]
if _DEBUG:
    extra_compile_args += ["-g3", "-O0", "-DDEBUG=%s" % _DEBUG_LEVEL, "-UNDEBUG"]
else:
    extra_compile_args += ["-DNDEBUG", "-O3"]

setup(
    name='lsgdual',
    version='',
    packages=['lsgdual'],
    package_dir={'': ''},
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            "cygdual",
            sources=[
                "cygdual.pyx",
                libgdual_dir + "/gdual.c"
            ],
            include_dirs=[
                numpy.get_include(),
                libgdual_dir
            ],
            define_macros=[('LONG_DOUBLE_INTERNALS', '1')],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    url='',
    license='',
    author='Kevin Winner',
    author_email='kwinner@cs.umass.edu',
    description='Package for manipulating generalized dual numbers using a log-sign representation.'
)
