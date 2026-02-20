# cython: language_level=3

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
from Cython.Compiler import Options
import os
import sys
import numpy as np

os.environ['CFLAGS'] = "-flto -g0"

Options.cimport_from_pyx = True
Options.docstrings = False

extensions = [
    Extension(
        name="parser",
        sources=["parser.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="fastmatch",
        sources=["fastmatch.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="parser",
    version="0.0.1",
    description="a",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "infer_types": True, "emit_code_comments": False},
    ),
)