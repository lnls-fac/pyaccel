#!/usr/bin/env python-sirius

import pkg_resources
from setuptools import find_packages, setup
from distutils.version import StrictVersion


trackcpp_version = '4.10.5'

try:
    import trackcpp
except ImportError:
    raise RuntimeError("trackcpp package not found")

if StrictVersion(trackcpp.__version__) < StrictVersion(trackcpp_version):
    msg = ("trackcpp package version must be >= " + trackcpp_version +
        " (version installed is " + trackcpp.__version__ + ")")
    raise RuntimeError(msg)


def get_abs_path(relative):
    return pkg_resources.resource_filename(__name__, relative)


with open(get_abs_path("README.md"), "r") as _f:
    _long_description = _f.read().strip()


with open(get_abs_path("VERSION"), "r") as _f:
    __version__ = _f.read().strip()


_requirements = ['', ]
#with open(get_abs_path("requirements.txt"), "r") as _f:
#    _requirements = _f.read().strip().split("\n")


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
