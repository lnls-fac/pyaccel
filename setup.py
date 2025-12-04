#!/usr/bin/env python-sirius

import pathlib
from setuptools import find_packages, setup


def get_abs_path(relative):
    return str(pathlib.Path(__file__).parent / relative)


with open(get_abs_path("README.md"), "r") as _f:
    _long_description = _f.read().strip()


with open(get_abs_path("VERSION"), "r") as _f:
    __version__ = _f.read().strip()


with open(get_abs_path("requirements.txt"), "r") as _f:
    _requirements = _f.read().strip().split("\n")


setup(
    name='pyaccel',
    version=__version__,
    author='lnls-fac',
    description='High level Accelerator Physics package',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lnls-fac/pyaccel',
    download_url='https://github.com/lnls-fac/pyaccel',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=find_packages(),
    install_requires=_requirements,
    package_data={'pyaccel': ['VERSION', ]},
    include_package_data=True,
    python_requires=">=3.6",
    zip_safe=False,
    )
