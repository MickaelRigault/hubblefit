#! /usr/bin/env python
# -*- coding: utf-8 -*-

DESCRIPTION = "Hubblizer: Fit and standardize SN data ."
LONG_DESCRIPTION = """ Tools to fit the hubble diagram from SNeIa including flexible standardization. """

DISTNAME = 'hubblefit'
AUTHOR = 'Mickael Rigault based on early development from Nicolas Chotard and Stephen Bailey'
MAINTAINER = 'Mickael Rigault' 
MAINTAINER_EMAIL = 'm.rigault@ipnl.in2p3.fr'
URL = 'https://github.com/MickaelRigault/hubblefit/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/MickaelRigault/hubblefit/tarball/0.3'
VERSION = '0.3.0'

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []        
    try:
        import astropy
    except ImportError:
        install_requires.append('astropy')

    try:
        import modefit
    except ImportError:
        install_requires.append('modefit')

    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    if _has_setuptools:
        packages = find_packages()
        print(packages)
    else:
        # This should be updated if new submodules are added
        packages = ['hubblefit']

    setup(name=DISTNAME,
          author=AUTHOR,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          packages=packages,
          package_data={'hubblefit': []},
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 2.7',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
      )
