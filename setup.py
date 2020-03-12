#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Python tool to predict tstar, given a Taup file and a travel time

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2020
:license:
    None
'''
from setuptools import setup, find_packages

setup(name='mqs_reports',
      version='0.1',
      description='Python tool to predict tstar given.',
      url='https://github.com/sstaehler/meas_Q',
      author='Simon Stähler',
      author_email='staehler@erdw.ethz.ch',
      license='None',
      packages=find_packages(),
      install_requires=['obspy', 'numpy', 'scipy'])
