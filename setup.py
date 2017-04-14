from setuptools import setup
from setuptools import find_packages

MAJOR_VERSION = '0'
MINOR_VERSION = '2'
MICRO_VERSION = '36'
VERSION = "{}.{}.{}".format(MAJOR_VERSION, MINOR_VERSION, MICRO_VERSION)

setup(name='xtoy',
      version=VERSION,
      description='get xtoyed predictions from raw data',
      url='https://github.com/kootenpv/xtoy',
      author='Pascal van Kooten',
      author_email='kootenpv@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'scikit-learn>=0.18.1',
          'deap',
          'bitstring'
      ],
      classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: Microsoft',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Unix',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development',
          'Topic :: Utilities'
      ],
      keywords=['data science', 'machine learning', 'genetic programming'],
      zip_safe=False,
      platforms='any')
