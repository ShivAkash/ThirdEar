# ThirdEar

## Real-Time Audio Pattern Detection for Situational Awareness in Headphone Users

ThirdEar is a real-time audio pattern detection system designed to improve situational awareness for users wearing headphones. The system continuously analyzes ambient audio and identifies relevant environmental sounds such as door knocks, alarms, sirens, glass breaking, fireworks, and crying babies.

The project combines lightweight deep learning, audio signal processing, and personalized audio pattern detection to provide low-latency, privacy-oriented sound awareness.

## Research Publication

This project is associated with the following research paper:

**Real-Time Audio Pattern Detection for Enhancing Situational Awareness in Headphone Users Using Deep Learning**

Presented at the **7th International Conference on Innovative Trends in Information Technology (ICITIIT 2026)**.

[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11499663)

## Problem

Headphones can reduce a user's awareness of important sounds in their surrounding environment. This can result in users missing sounds such as:

* Door knocks
* Alarms
* Emergency sirens
* Glass breaking
* Fireworks
* Crying babies
* Personalized speech or calling patterns

ThirdEar addresses this problem by continuously analyzing short audio segments and identifying sounds that may require the user's attention.



## Machine Learning Pipeline

The audio processing pipeline consists of the following stages:

1. Capture a short audio segment.
2. Apply preprocessing and windowing.
3. Compute the Short-Time Fourier Transform (STFT).
4. Convert the frequency representation into a Mel-spectrogram.
5. Feed the Mel-spectrogram into the CNN.
6. Obtain class probabilities from the softmax output.
7. Apply a confidence threshold.
8. Trigger an alert when the detection satisfies the decision criteria.

The model uses a **64-dimensional Mel filter bank** on short audio frames.

## Model Architecture

The classifier is designed to provide a balance between classification performance and computational cost.

| Component             | Configuration             |
| --------------------- | ------------------------- |
| Input                 | Mel-spectrogram           |
| Convolutional Block 1 | 32 filters, 3 x 3 kernel  |
| Convolutional Block 2 | 64 filters, 3 x 3 kernel  |
| Convolutional Block 3 | 128 filters, 3 x 3 kernel |
| Dense Layer 1         | 256 units                 |
| Dense Layer 2         | 256 units                 |
| Output                | Softmax                   |
| Dropout               | 0.3                       |
| Trainable Parameters  | ~920K                     |

The relatively small parameter count allows the model to be considered for CPU-based and edge deployment.

## Dataset

The research implementation uses a hybrid dataset containing approximately **3,500 audio samples**.

The dataset combines publicly available environmental sound datasets with custom-recorded personalized audio.

### Environmental Sound Classes

* Door knock
* Clock alarm
* Siren
* Fireworks
* Glass breaking
* Crying baby

### Background Noise

* Fan
* Traffic
* Typing

### Personalized Audio

Custom speech recordings are used to support personalized audio pattern detection.

The hybrid dataset allows the system to evaluate environmental sound recognition while also considering user-specific audio patterns.

## Results

The research implementation produced the following results:

| Metric                          |    Result |
| ------------------------------- | --------: |
| Average Classification Accuracy |     82.4% |
| Precision                       |     80.7% |
| Recall                          |     79.9% |
| CNN Inference Time              |  ~9–11 ms |
| End-to-End Latency              |   ~120 ms |
| Confidence Threshold            |      0.65 |
| Alert Cooldown                  | 2 seconds |
| Trainable Parameters            |     ~920K |

The approximate processing breakdown is:

| Stage                       |    Time |
| --------------------------- | ------: |
| Audio preprocessing         |  ~12 ms |
| Feature extraction          |  ~18 ms |
| CNN inference               |  ~10 ms |
| Decision and alert handling |   ~5 ms |
| End-to-end pipeline         | ~120 ms |

The reported end-to-end latency includes the complete audio capture, processing, inference, decision, and alert pipeline.

## Decision Mechanism

To reduce false or repeated notifications, ThirdEar uses a confidence threshold and cooldown mechanism.

```text
if confidence >= 0.65:
    trigger alert
else:
    ignore prediction
```

A two-second cooldown prevents repeated alerts when the same sound continues across multiple audio frames.

## Privacy

ThirdEar is designed with local processing as a primary consideration.

The system aims to:

* Process audio locally where possible
* Avoid continuous cloud-based audio transmission
* Avoid persistent storage of raw audio
* Use lightweight models suitable for edge inference
* Minimize the amount of user audio retained by the system

This architecture is particularly relevant for applications that continuously monitor audio.

## Repository Structure

```text
ThirdEar/
|
+-- audioset_tagging_cnn/
|   +-- Audio classification and model implementation
|
+-- frontend/
|   +-- Frontend application
|
+-- requirementsall.txt
|
+-- README.md
```

## Technology Stack

### Machine Learning

* Python
* TensorFlow
* Keras
* NumPy
* Librosa
* Convolutional Neural Networks

### Audio Processing

* Short-Time Fourier Transform (STFT)
* Mel-spectrograms
* Mel filter banks
* Digital signal processing
* Real-time audio classification

### Frontend

* React
* TypeScript

## Installation

Clone the repository:

```bash
git clone https://github.com/ShivAkash/ThirdEar.git
cd ThirdEar
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the Python dependencies:

```bash
pip install -r requirementsall.txt
```

Refer to the individual directories for component-specific setup and execution instructions.

## Research Contributions

The project focuses on the following areas:

* Real-time environmental sound classification
* Lightweight CNN architectures for audio recognition
* Personalized audio pattern detection
* Low-latency CPU inference
* Privacy-oriented audio processing
* Edge deployment of audio classification models
* Situational awareness for headphone users

## Future Work

Potential areas for further development include:

* INT8 model quantization
* Fully offline inference
* Android deployment
* iOS deployment
* Windows and macOS support
* Personalized sound enrollment
* Speaker-aware detection
* Adaptive confidence thresholds
* Additional environmental sound classes
* Improved robustness under background noise
* Power-efficient continuous audio processing
* Hardware-accelerated inference
* Integration with gaming and communication applications

## Citation

If you use this project or its research findings, please cite the associated publication.

```bibtex
@inproceedings{thirdEar2026,
  title={Real-Time Audio Pattern Detection for Enhancing Situational Awareness in Headphone Users Using Deep Learning},
  booktitle={7th International Conference on Innovative Trends in Information Technology (ICITIIT)},
  year={2026}
}
```

For the exact bibliographic metadata, refer to the official IEEE Xplore record:

https://ieeexplore.ieee.org/abstract/document/11499663


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Project Status

ThirdEar is an ongoing research and development project. The current repository contains the implementation associated with the research work, with additional work planned toward cross-platform deployment and fully offline inference.
