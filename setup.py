#!/usr/bin/env python-sirius

from setuptools import setup
from distutils.version import StrictVersion


trackcpp_version = '0.2.0'

try:
    import trackcpp
except ImportError:
    raise RuntimeError("trackcpp package not found")

if StrictVersion(trackcpp.__version__) < StrictVersion(trackcpp_version):
    msg = ("trackcpp package version must be >= " + trackcpp_version +
        " (version installed is " + trackcpp.__version__ + ")")
    raise RuntimeError(msg)

with open('VERSION', 'r') as _f:
    __version__ = _f.read().strip()

setup(
    name='pyaccel',
    version=__version__,
    author='lnls-fac',
    description='High level Accelerator Physics package',
    url='https://github.com/lnls-fac/pyaccel',
    download_url='https://github.com/lnls-fac/pyaccel',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    packages=['pyaccel'],
    package_data={'pyaccel': ['VERSION', 'data/d_touschek.npz']},
    zip_safe=False
)
