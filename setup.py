from setuptools import setup
from setuptools import find_packages

setup(
    name='glmcc',
    version='0.1.1',
    author='Takuma Furukawa',
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'scipy']
)