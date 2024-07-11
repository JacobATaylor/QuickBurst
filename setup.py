"""
Setup script for QuickBurst
can install like this: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    author='Jacob Taylor, Rand Burnette',
    name='QuickBurst',
    version='0.9.0',
    install_requires=[
         'numpy',
         'enterprise_extensions',
         'numba',
         'h5py',
    ],
    python_requires='>=3.7',
    packages=find_packages(include=['QuickBurst']),
    long_description=open('README.md').read(),
    )
