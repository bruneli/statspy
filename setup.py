#!/usr/bin/env python

import sys
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins

DISTNAME = 'statspy'
DESCRIPTION = 'Python module for statistics built on top of NumPy/SciPy'
LONG_DESCRIPTION = open('README.rst').read()
URL = 'http://bruneli.github.io/statspy/'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'http://sourceforge.net/projects/statspy/files/'

builtins.__STATSPY_SETUP__ = True

import statspy
VERSION = statspy.__version__

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('statspy')
    config.add_data_dir('statspy/tests')
    return config

def setup_package():
    metadata = dict(name=DISTNAME,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    download_url=DOWNLOAD_URL,
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: Unix',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.6',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3',
                                 'Programming Language :: Python :: 3.2',
                                 ])
    if (len(sys.argv) >= 2
        and ('--help' in sys.argv[1:] or sys.argv[1]
             in ('--help-commands', 'egg_info', '--version', 'clean'))):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        metadata['version'] = VERSION
    else:
        from numpy.distutils.core import setup
        metadata['configuration'] = configuration
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
