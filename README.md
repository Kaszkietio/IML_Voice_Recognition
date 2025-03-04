# IML Voice Recognition

## Overview

This repository contains an various models exploration for a voice recognition system using machine learning techniques. The project is part of an Introduction to Machine Learning (IML) course and focuses on processing and classifying voice data using modern deep learning models.

## Features

- **Voice Data Processing**: Preprocessing and feature extraction from audio recordings.
- **Machine Learning Models**: Implementation of deep learning models for voice recognition.
- **Evaluation Metrics**: Performance assessment using accuracy, precision, recall, and F1-score.
- **Visualization**: Graphical representation of model training and performance.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.11
- TensorFlow / Keras
- NumPy
- Pandas
- Librosa (for audio processing)
- Matplotlib & Seaborn (for visualization)

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kaszkietio/IML_Voice_Recognition.git
cd IML_Voice_Recognition
pip install -r requirements.txt
```

### Dataset

Models were trained on:
- [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge)
- [DAPS](https://www.kaggle.com/datasets/psyreddy07/daps-data)

## Project Structure

```
IML_Voice_Recognition/
│-- data/              # Dataset and audio files
│-- models/            # Trained models and checkpoints
│-- src/               # Preprocessing, training, evaluation scripts
│-- notebooks/         # Jupyter notebooks for analysis
│-- requirements.txt   # Dependencies
│-- README.md          # Project documentation
```

## License

This project is licensed under the MIT License.

