#!/usr/bin/env python

from setuptools import setup, find_packages

__version__ = '0.0.1'
__author__ = "Sebastian Eppner, Graham Herdman, Maud Grol"
__license__ = "MIT"

setup(
    name='deepdiva',
    version=__version__,
    author=__author__,
    description='Deepdiva project for deep-learning synthesizer programmer using the vst plugin DIVA',
    licence=__license__,
    packages=find_packages(where='src',
                           include=['deepdiva']),
    package_dir={"": "src"}
)
