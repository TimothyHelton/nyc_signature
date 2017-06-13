#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import open
import os.path as osp
import re

from setuptools import setup, find_packages


with open('nyc_signature/__init__.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

here = osp.abspath(osp.dirname(__file__))
with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nyc_signature',
    version=version,
    description='New York City Signature Canvasing Project',
    author='Timothy Helton',
    author_email='timothy.j.helton@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Build Tools',
        ],
    keywords='common tools utility',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=[
        'bokeh',
        'geocoder',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'requests',
        'seaborn',
        ],
    package_dir={'nyc_signature': 'nyc_signature'},
    include_package_data=True,
    )


if __name__ == '__main__':
    pass
