# cython: language_level=3

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import os
import sys
import platform
import numpy as np

Options.cimport_from_pyx = True
Options.docstrings = False

machine = platform.machine().lower()
is_x86 = machine in {"x86_64", "amd64", "i386", "i686"}
is_apple_arm = sys.platform == "darwin" and machine in {"arm64", "aarch64"}

common_compile_args = ["-O3"]
if is_x86 and not (sys.platform == "darwin"):
    # keep native tuning on Linux x86 but avoid arch flags on macOS ARM
    common_compile_args.append("-march=native")

extensions = [
    Extension(
        name="parser",
        sources=["parser.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        name="deckgen",
        sources=["deckgen.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
    Extension(
        name="fastmatch",
        sources=["fastmatch.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
    ),
]

if is_x86 or is_apple_arm:
    extensions.append(
        Extension(
            name="fastmatch_simd",
            sources=["fastmatch_simd.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=common_compile_args,
        )
    )

setup(
    name="parser",
    version="0.0.1",
    description="a",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "infer_types": True, "emit_code_comments": False},
        annotate=True,
        force=True,
    ),
)
