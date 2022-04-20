from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '1.1.3'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='mpes',
    version=__version__,
    description='Distributed data processing routines for multidimensional photoemission spectroscopy (MPES)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mpes-kit/mpes',
    download_url='https://github.com/mpes-kit/mpes/tarball/' + __version__,
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='R. Patrick Xian, Laurenz Rettig',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='xrpatrick@gmail.com, rettig@fhi-berlin.mpg.de'
)
