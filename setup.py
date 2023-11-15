# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:21:37 2023

@author: mofeli
"""

from setuptools import setup, find_packages

setup(
    name='E2E_PPG_Pipeline',
    version='0.1.0',  # Update with the appropriate version number
    description='End-to-End PPG Processing Pipeline',
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
    long_description='''\
    # E2E PPG Pipeline

    End-to-End PPG Processing Pipeline for Wearables: From Quality Assessment and Motion Artifacts Removal to HR/HRV Feature Extraction. It is a comprehensive package designed for extracting accurate Heart Rate (HR) and Heart Rate Variability (HRV) data from Photoplethysmogram (PPG) signals.

    This project provides a robust pipeline for processing PPG signals, extracting reliable HR and HRV parameters. The pipeline encompasses various stages, including filtering, signal quality assessment (SQA), signal reconstruction, peak detection and interbeat interval (IBI) extraction, and HR and HRV computation.
    ## Preprocessing
    The input raw PPG signal undergoes filtering to remove undesired frequencies. A second-order Butterworth bandpass filter is employed, allowing frequencies within a specific range to pass while rejecting frequencies outside.
    
    ## Signal Quality Assessment (SQA)
    SQA involves identifying clean and noisy parts within PPG signals. Our SQA approach requires PPG signals in a fixed length, which necessitates segmenting the input signals. To this end, we apply a moving window segmentation technique, where the PPG signals are divided into overlapping segments, each spanning 30 seconds, by sliding a window over the signal. The SQA process includes PPG feature extraction and classification, employing a one-class support vector machine model to distinguish between "Reliable" and "Unreliable" segments.
    
    ## Noise Reconstruction
    Noisy parts within PPG signals, shorter than a specified threshold, are reconstructed using a deep convolutional generative adversarial network (GAN). The GAN model includes a generator trained to produce synthetic clean PPG segments and a discriminator to differentiate real from synthetic signals. The reconstruction process is applied iteratively to ensure denoising.
    
    ## Peak Detection and IBI Extraction
    Systolic peaks in PPG signals are identified using a deep-learning-based method with dilated Convolutional Neural Networks (CNN) architecture. PPG peak detection enables the extraction of IBI values that serve as the basis for obtaining HR and HRV. IBI represents the time duration between two consecutive heartbeats and is computed by measuring the time interval between systolic peaks within the PPG signals.
    
    ## HR and HRV Extraction
    HR and HRV parameters are computed from the extracted IBIs. A variety of metrics are calculated, including:

    - HR: Heart rate
    - MeanNN: Mean of the RR intervals
    - SDNN: Standard deviation of the RR intervals
    - ... (list continues for various HRV metrics)

    ## Installation

    You can install the package using pip:

    ```
    pip install E2E_PPG_Pipeline
    ```

    ## Usage

    Install the required packages available in `requirements.txt`. Load your PPG signal and call the `HRV_Extraction()` function in `E2E_PPG_Pipeline.py` to extract HR and HRV parameters. Define the sample rate and window length for HR and HRV extraction.

    ## License

    This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
    ''',
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
