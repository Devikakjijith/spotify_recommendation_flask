# SoundSpace | Spotify Music Analytics & Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

SoundSpace is an unsupervised machine learning application and interactive web platform built with Flask, scikit-learn, and pandas. It analyzes over **230,000+ tracks** from Spotify, applies **Principal Component Analysis (PCA)** for dimensionality reduction, performs **K-Means clustering**, and delivers a real-time **content-based song recommendation engine** powered by cosine similarity in the latent audio feature space.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Project Structure](#project-structure)
- [Dataset Details](#dataset-details)
- [Installation & Setup](#installation--setup)
- [Model Training & Persistence](#model-training--persistence)
- [Running the Application](#running-the-application)
- [Application Pages](#application-pages)
- [Evaluation & Performance Metrics](#evaluation--performance-metrics)
- [Tech Stack](#tech-stack)

---

## Overview

Traditional song recommendation systems often rely on collaborative filtering (user listening history), which suffers from the cold-start problem. **SoundSpace** uses an **unsupervised content-based approach**, analyzing the intrinsic acoustic properties of audio tracks—such as danceability, energy, acousticness, tempo, valence, and loudness.

By reducing high-dimensional audio features into an optimal 7-dimensional PCA space ($\ge 95\%$ variance retained), SoundSpace enables fast, accurate similarity searches and real-time inference on both existing library tracks and user-crafted audio profiles.

---

## Key Features

- **Model Persistence & Fast Startup**: Pre-trained pipelines (`StandardScaler`, `PCA`, `KMeans`) are serialized using `joblib`. The web application boots in ~1 second with zero repeated training overhead, while auto-training on initial launch if saved artifacts are absent.
- **Dual-Mode Recommendation Engine**:
  1. **Library Selection**: Browse curated top tracks across genres or search any track from the dataset to find acoustically similar songs.
  2. **Custom Audio Profiling (Real-Time Inference)**: Adjust acoustic sliders (danceability, tempo, energy, valence, etc.). The inputs are standardized and projected into PCA latent coordinates (PC1–PC7) on the fly to find matching tracks.
- **Comprehensive Exploratory Data Analysis (EDA)**: High-resolution visualizations including feature histograms, correlation heatmaps, feature boxplots, and scatter plots.
- **PCA & Clustering Performance Dashboard**: Live metrics displaying cumulative explained variance, reconstruction Mean Squared Error (MSE), dimensionality reduction percentage, and K-Means silhouette evaluation.
- **Responsive UI**: Custom CSS interface with instant table rendering, active model status badges, and PCA projection chips.

---

## Machine Learning Architecture

```
                      Raw Spotify Dataset (232k+ Tracks)
                                     │
                                     ▼
                     Data Cleaning & Median Imputation
                                     │
                                     ▼
                   Feature Scaling (StandardScaler)
                                     │
                                     ▼
             Principal Component Analysis (PCA: 9 -> 7 PCs)
              ├── Retains >= 95% Cumulative Variance
              ├── Dimensionality Reduction: 22.2%
              └── Reconstruction MSE: ~0.043
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        K-Means Clustering                      Latent Feature Space
    (k=3, Silhouette = 0.3740)                  (232k x 7 Projections)
                                                         │
                                                         ▼
                                               Cosine Similarity Engine
                                               ├── Library Track Search
                                               └── Real-Time Custom Input
```

---

## Project Structure

```
spotify_unsupervised2/
│
├── app.py                   # Main Flask web application and routing
├── train_model.py           # Model training, evaluation, and serialization script
├── requirements.txt         # Project dependencies
├── SpotifyFeatures.csv      # Dataset (232,725 songs with acoustic features)
├── model.pkl                # Root serialized model bundle
├── .gitignore               # Ignored files (virtual environment, cache, etc.)
│
├── models/                  # Persisted model artifacts
│   ├── scaler.pkl           # Trained StandardScaler
│   ├── pca_model.pkl        # Fitted PCA model (7 components)
│   ├── kmeans_model.pkl     # Fitted KMeans model
│   └── spotify_model.pkl    # Complete model & metadata bundle
│
├── templates/               # Jinja2 HTML templates
│   ├── base.html            # Navigation and base layout
│   ├── dataset.html         # Dataset overview and sample view
│   ├── eda.html             # EDA visualizations and correlation charts
│   ├── recommendations.html # Recommendation engine (Library & Custom modes)
│   └── pca.html             # PCA metrics, variance scree plot, and interpretation
│
└── static/                  # Static assets
    └── style.css            # Custom responsive styles and UI theme
```

---

## Dataset Details

The project utilizes `SpotifyFeatures.csv`, containing **232,725 tracks** across diverse genres. 

Key numeric audio features analyzed:
- **`acousticness`** (0.0 to 1.0): Confidence measure of whether the track is acoustic.
- **`danceability`** (0.0 to 1.0): Suitability of the track for dancing.
- **`energy`** (0.0 to 1.0): Perceptual measure of intensity and activity.
- **`instrumentalness`** (0.0 to 1.0): Predicts whether a track contains no vocals.
- **`liveness`** (0.0 to 1.0): Detects the presence of an audience in the recording.
- **`loudness`** (-60 to 0 dB): Overall loudness of a track.
- **`speechiness`** (0.0 to 1.0): Presence of spoken words in a track.
- **`tempo`** (BPM): Overall estimated tempo in beats per minute.
- **`valence`** (0.0 to 1.0): Musical positiveness conveyed by the track.

---

## Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone <repository-url>
cd spotify_unsupervised2
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Model Training & Persistence

You can train and serialize the models at any time by executing [`train_model.py`](train_model.py):

```bash
python train_model.py
```

This script:
1. Loads and validates `SpotifyFeatures.csv`.
2. Standardizes audio features with `StandardScaler`.
3. Fits `PCA`, computing cumulative variance to select components reaching $\ge 95\%$ variance threshold.
4. Fits a `KMeans` model and computes silhouette score.
5. Saves individual models (`models/scaler.pkl`, `models/pca_model.pkl`, `models/kmeans_model.pkl`) and the bundled package (`models/spotify_model.pkl` / `model.pkl`).

*Note: If `app.py` is started without existing `.pkl` files, it will automatically call `train_and_save_model()` on first launch.*

---

## Running the Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Application Pages

| Route | Page | Description |
| :--- | :--- | :--- |
| `/` or `/dataset` | **Dataset Overview** | Displays dataset metrics (total tracks, features, duplicate counts), column data types, missing value statistics, and preview tables. |
| `/eda` | **Exploratory Data Analysis** | Displays distribution histograms, feature correlation matrix heatmap, boxplots for outliers, and danceability vs. energy scatter plots. |
| `/recommendations` | **Song Recommendations** | Dual-tab interface supporting both library song selection and custom audio attribute sliders with live PCA coordinate projections. |
| `/pca` | **PCA & Performance** | Detailed model performance metrics, variance scree plot, 2D PCA song projection scatter plot, and saved model artifact status. |

---

## Evaluation & Performance Metrics

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Original Features** | 9 | Audio attributes extracted from dataset |
| **Reduced Components** | 7 | Principal components selected ($\ge 95\%$ variance threshold) |
| **Variance Retained** | **95.64%** | Total information preserved in reduced space |
| **Dimensionality Reduction** | **22.22%** | Reduction in feature space complexity |
| **Reconstruction MSE** | **0.04358** | Mean squared error between original and reconstructed scaled features |
| **K-Means Silhouette Score** | **0.3740** | Cluster cohesion and separation in latent space |
| **Application Startup Time** | **~1.0s** | Instant boot using pre-computed serialized artifacts |

---

## Tech Stack

- **Backend**: [Python 3](https://www.python.org/), [Flask](https://flask.palletsprojects.com/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (`StandardScaler`, `PCA`, `KMeans`, `cosine_similarity`)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- **Data Visualization**: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)
- **Model Serialization**: [joblib](https://joblib.readthedocs.io/)
- **Frontend**: HTML5, Jinja2 templates, Custom Responsive CSS3

