# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:21:37 2023

@author: mofeli
"""

from setuptools import setup, find_packages

setup(
    name='E2E_PPG_Pipeline',
    version='0.1.0',  # Update with the appropriate version number
    description='End-to-End PPG Processing Pipeline for Wearables: From Quality Assessment and Motion Artifacts Removal to HR/HRV Feature Extraction',
    author='Mohammad Feli',
    author_email='mohammad.feli@utu.fi',
    packages=find_packages(),
    install_requires=[
        'heartpy==1.2.7',
        'joblib==1.1.0',
        'matplotlib==3.5.2',
        'more_itertools==10.1.0',
        'neurokit2==0.2.1',
        'numpy==1.21.5',
        'pandas==1.4.4',
        'scikit-learn==1.0.2',
        'scipy==1.9.1',
        'tensorflow==2.10.1',
        'torch==1.9.1',
        'torchvision==0.10.1',
        'wfdb==4.1.2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
