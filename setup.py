from setuptools import setup

setup(name='nwDiff',
      version='0.0.4',
      description='Provides classes to simulate simple diffusion on networks and heuristics for time observables',
      url='https://www.github.com/benmaier/GillEpi',
      author='Benjamin F. Maier',
      author_email='bfmaier@physik.hu-berlin.de',
      license='MIT',
      packages=['nwDiff'],
      install_requires=[
          'numpy',
          'networkx',
      ],
      zip_safe=False)
