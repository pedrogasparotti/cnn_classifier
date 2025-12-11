# CNN Classifier for Bridge Damage Detection via Drive-by Monitoring

> **Supervised Convolutional Neural Network for Highway Bridge Structural Health Monitoring Using Indirect (Drive-by) Acceleration Data**

This repository contains the implementation of a 1D Convolutional Neural Network (CNN) for classifying bridge health conditions based on acceleration signals from vehicles crossing the structure. The methodology uses **indirect monitoring** (drive-by method), where sensors mounted on vehicles capture dynamic responses during bridge crossings.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Development Guide](#development-guide)
- [Citation](#citation)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This project addresses **scour-induced damage detection** in highway bridges using machine learning. Scour (erosion around bridge foundations) is responsible for over 60% of bridge failures in the United States. Traditional inspection methods are costly, disruptive to traffic, and often fail to detect hidden damage below water level.

### Key Contributions

- **Supervised CNN architecture** optimized for 1D acceleration signals
- **Drive-by monitoring approach** using vehicle-mounted sensors
- **Multi-class classification**: Healthy, 5% damage, 10% damage
- **Operational variability handling**: vehicle properties, speed, road roughness
- **100% accuracy** on holdout test set (numerical simulations)
- **Minimum sample size analysis** for training efficiency

---

## Project Structure

```
cnn_classifier/
│
├── src/
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── model_builder.py            # CNN architecture definition
│   ├── hyperparameter_tuning.py   # Bayesian optimization
│   ├── training.py                 # Model training with callbacks
│   ├── evaluation.py               # Metrics and visualization
│   └── utils.py                    # Helper functions
│
├── keras_tuner_dir/                # Hyperparameter search results
│   └── 1d_cnn_hyper_tuning/
│
├── keras_tuner_dir_bayesian/       # Bayesian optimization results
│   └── 1d_cnn_bayesian_tuning/
│
├── cross_val_results/              # K-fold validation outputs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset analysis
│   ├── 02_model_training.ipynb     # Training pipeline
│   └── 03_results_analysis.ipynb   # Results visualization
│
├── data/                           # Dataset (see Dataset section)
│   ├── train/
│   │   ├── healthy_baseline.npy
│   │   ├── damage_5percent.npy
│   │   └── damage_10percent.npy
│   ├── test/
│   └── holdout/
│
├── models/                         # Saved trained models
│   └── best_cnn_model.h5
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── README.md                       # This file
└── LICENSE                         # License information
```

---

## Installation

### Prerequisites

- Python 3.8 or higher

### Step 1: Clone the Repository

```bash
git clone https://github.com/pedrogasparotti/cnn_classifier.git
cd cnn_classifier
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
keras-tuner==1.3.5
scipy==1.11.1
joblib==1.3.1
```
---

## Dataset

### Data Generation

Acceleration data is generated using **VBI-2D** (Vehicle-Bridge Interaction 2D), a MATLAB-based finite element simulation tool developed by Cantero (2024).

**Reference:**
> Cantero, D. (2024). VBI-2D – Road vehicle-bridge interaction simulation tool and verification framework for Matlab. *SoftwareX*, 26, 101725.

### Vehicle Model

- **Type**: 5-axle articulated truck-trailer
- **Total mass**: 35,000 kg
- **DOFs**: 14 (vertical displacements + rotations)
- **Speed range**: 19-21 m/s (variable)
- **Tire stiffness**: 1.6×10⁶ to 3.9×10⁶ N/m
- **Suspension stiffness**: 3.6×10⁵ to 2.5×10⁶ N/m

### Bridge Model

- **Type**: Simply supported Euler-Bernoulli beam
- **Length**: 25 m
- **Material**: Reinforced concrete
- **Young's modulus**: 3.5×10¹⁰ Pa
- **Mass per unit length**: 18,358 kg/m
- **Second moment of area**: 1.3901 m⁴

### Damage Scenarios

| Scenario | Description | Foundation Stiffness | Samples |
|----------|-------------|---------------------|---------|
| **Healthy** | No scour damage | K_f = 3.44×10⁸ Pa | 600 |
| **5% Damage** | Minor scour (β=0.95) | K_eff = 0.95 × K_f | 600 |
| **10% Damage** | Moderate scour (β=0.90) | K_eff = 0.90 × K_f | 600 |

### Environmental & Operational Variabilities (EOVs)

- **Road profile**: ISO 8608 Class A (randomly generated)
- **Vehicle properties**: Random variations (±10%) in mass, stiffness, damping
- **Measurement noise**: 5% additive Gaussian noise
- **Sampling rate**: 1000 Hz
- **Signal length**: 1,936 points (resampled to fixed length)

### Download Dataset

```bash
# The dataset is hosted on Google Drive
# Download link: https://drive.google.com/drive/folders/1sOhhGaJ1dINsPp3AZqT4m_remHE7Bi_9?usp=sharing

# Place downloaded files in:
./data/train/
./data/test/
./data/holdout/
```

--

### Acknowledgments

- **CORE (Centro de Otimização e Confiabilidade em Engenharia)** - Federal University of Santa Catarina
- **VBI-2D Development Team** - For providing the simulation tool
- **Reviewers** - For valuable feedback on the methodology

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Pedro Vinícius Gasparotti de Souza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Links

- **Dataset**: https://drive.google.com/drive/folders/1sOhhGaJ1dINsPp3AZqT4m_remHE7Bi_9?usp=sharing
- **VBI-2D Repository**: https://github.com/DanielCanteroNTNU/VBI-2D
- **Lab Website**: [CORE/UFSC](http://core.ufsc.br/)

---