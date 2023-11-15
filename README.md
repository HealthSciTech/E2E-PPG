[![PyPI](https://img.shields.io/pypi/v/neurokit2.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/E2E-PPG-Pipeline/0.1.0/)



# End-to-End PPG Processing Pipeline for Wearables: From Quality Assessment and Motion Artifacts Removal to HR/HRV Feature Extraction (E2E-PPG)

Welcome to the PPG Signal Processing Pipeline, a comprehensive package designed for extracting accurate Heart Rate (HR) and Heart Rate Variability (HRV) data from Photoplethysmogram (PPG) signals.

## Introduction

This project provides a robust pipeline for processing PPG signals, extracting reliable HR and HRV parameters. The pipeline encompasses various stages, including filtering, signal quality assessment (SQA), signal reconstruction, peak detection and interbeat interval (IBI) extraction, and HR and HRV computation.

<img src="https://github.com/HealthSciTech/E2E-PPG/assets/67778755/896be83f-4709-4444-bac9-2fef0449f739" alt="overview" width="800" height="200">

## Preprocessing
The input raw PPG signal undergoes filtering to remove undesired frequencies. A second-order Butterworth bandpass filter is employed, allowing frequencies within a specific range to pass while rejecting frequencies outside.


## Signal Quality Assessment (SQA)
SQA involves identifying clean and noisy parts within PPG signals. Our SQA approach requires PPG signals in a fixed length, which necessitates segmenting the input signals. To this end, we apply a moving window segmentation technique, where the PPG signals are divided into overlapping segments, each spanning 30 seconds, by sliding a window over the signal. The SQA process includes PPG feature extraction and classification, employing a one-class support vector machine model to distinguish between "Reliable" and "Unreliable" segments.


<img src="https://github.com/HealthSciTech/E2E-PPG/assets/67778755/c0ffee6c-f7b5-4d27-9f34-34cb86a698b5" alt="seg" width="300" height="300">

<img src="https://github.com/HealthSciTech/E2E-PPG/assets/67778755/f63e40d3-74b3-497b-ac91-dc940e669f03" alt="sample" width="400" height="300">



## Noise Reconstruction

Noisy parts within PPG signals, shorter than a specified threshold, are reconstructed using a deep convolutional generative adversarial network (GAN). The GAN model includes a generator trained to produce synthetic clean PPG segments and a discriminator to differentiate real from synthetic signals. The reconstruction process is applied iteratively to ensure denoising.

<img src="https://github.com/HealthSciTech/E2E-PPG/assets/67778755/bb00f079-7341-4ac9-84e2-553eb6a62672" alt="rec-arch" width="550" height="300">
<br />
<br />
<br />
<img src="https://github.com/HealthSciTech/E2E-PPG/assets/67778755/8cf57fa6-94fc-4906-b4c3-8416fffced4e" alt="rec-iter" width="600" height="200">
<br />
<br />
<br />
<img src="https://github.com/HealthSciTech/E2E-PPG/assets/67778755/ef0ce7aa-ab34-4176-a0a3-9192c7bd94de" alt="rec-iter" width="480" height="300">




## Peak Detection and IBI Extraction
Systolic peaks in PPG signals are identified using a deep-learning-based method with dilated Convolutional Neural Networks (CNN) architecture. PPG peak detection enables the extraction of IBI values that serve as the basis for obtaining HR and HRV. IBI represents the time duration between two consecutive heartbeats and is computed by measuring the time interval between systolic peaks within the PPG signals. 
![200779269-c0cfc80a-cb53-4dc7-91e3-7b7590977e7f](https://github.com/HealthSciTech/E2E-PPG/assets/67778755/82ba92d8-b012-4202-8e17-127b0a5df4e5)


## HR and HRV Extraction
HR and HRV parameters are computed from the extracted IBIs. A variety of metrics are calculated, including:

- HR: Heart rate
- MeanNN: Mean of the RR intervals
- SDNN: Standard deviation of the RR intervals
- ... (list continues for various HRV metrics)

## Usage

Install the required packages available in `requirements.txt`. 

See the `Example.py` file for usage details. Load your PPG signal and call the `HRV_Extraction()` function in `E2E_PPG_Pipeline.py` to extract HR and HRV parameters. Define the sample rate and window length for HR and HRV extraction.

## Citation
If you use our work in your research, please consider citing the following papers:

1. **An energy-efficient semi-supervised approach for on-device photoplethysmogram signal quality assessment**

  ```
  @article{feli2023energy,
  title={An energy-efficient semi-supervised approach for on-device photoplethysmogram signal quality assessment},
  author={Feli, Mohammad and Azimi, Iman and Anzanpour, Arman and Rahmani, Amir M and Liljeberg, Pasi},
  journal={Smart Health},
  volume={28},
  pages={100390},
  year={2023},
  publisher={Elsevier}
  }
  ```

2. **Ppg signal reconstruction using deep convolutional generative adversarial network**

  ```
 @inproceedings{wang2022ppg,
  title={Ppg signal reconstruction using deep convolutional generative adversarial network},
  author={Wang, Yuning and Azimi, Iman and Kazemi, Kianoosh and Rahmani, Amir M and Liljeberg, Pasi},
  booktitle={2022 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={3387--3391},
  year={2022},
  organization={IEEE}
  }
  ```

3. **Robust ppg peak detection using dilated convolutional neural networks**

  ```
  @article{kazemi2022robust,
  title={Robust ppg peak detection using dilated convolutional neural networks},
  author={Kazemi, Kianoosh and Laitala, Juho and Azimi, Iman and Liljeberg, Pasi and Rahmani, Amir M},
  journal={Sensors},
  volume={22},
  number={16},
  pages={6054},
  year={2022},
  publisher={MDPI}
  }
  ```


## Contributing

We welcome contributions to enhance and extend the capabilities of the PPG Signal Processing Pipeline. Please review our [Contribution Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the terms of the [MIT License](LICENSE.md).

## Acknowledgments

We express our gratitude to [mention any third-party libraries or resources] for their invaluable contributions to this project.
