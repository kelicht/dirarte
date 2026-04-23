from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dirarte',
    version='0.1.0',
    install_requires=requirements,
    description='A Python package for distributionally robust algorithmic recourse for tree ensembles.',
    packages=find_packages(),
)