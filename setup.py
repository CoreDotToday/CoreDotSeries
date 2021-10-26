#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.1.2', ]

test_requirements = ['pytest>=3', ]

setup(
    author="CoreDotToday",
    author_email='engine@core.today',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Library for time series",
    entry_points={
        'console_scripts': [
            'series=series.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='coredotseries',
    name='coredotseries',
    packages=find_packages(include=['series', 'series.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/coredottoday/CoreDotSeries.git',
    version='0.1.1',
    zip_safe=False,
)
