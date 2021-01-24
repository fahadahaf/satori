#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='satori',
      version='0.2',
      description='A self-attention model for inferring regulatory interactions',
      author='Fahad Ullah and Asa Ben-Hur',
      author_email='asa@cs.colostate.edu',
      maintainer='Fahad Ullah',
      maintainer_email='fahadullahkhattak@gmail.com',
      url='https://github.com/fahadahaf/satori_v2',
      packages=find_packages(),
      install_requires=[l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()],
      entry_points={
          'console_scripts': [
              'satori=satori:main',

          ],
      },
      )