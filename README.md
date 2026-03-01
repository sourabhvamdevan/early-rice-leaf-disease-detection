# Rice Leaf Disease Detector

A machine learning web application that detects rice leaf diseases from uploaded images. Built with Streamlit and deployed on Streamlit Community Cloud.

---

## Live Demo

The app is hosted at:
```
https://early-rice-leaf-a11.streamlit.app/
```

---

## Overview

This application allows users to upload a photo of a rice leaf and receive an instant disease diagnosis. The model classifies the leaf into one of three disease categories, displays a confidence score, provides a description of the detected disease, and recommends a treatment plan.

As seen in the application output, a sample prediction correctly identified Leaf smut with 67.1% confidence. The sidebar confirms the model loaded successfully and lists all three detectable classes with their respective severity levels.

---

## Disease Classes

| Disease | Severity | Description |
|---|---|---|
| Bacterial leaf blight | High | Caused by Xanthomonas oryzae. Produces water-soaked lesions that turn yellow along leaf margins. |
| Brown spot | Moderate | Caused by Cochliobolus miyabeanus. Produces brown oval lesions with yellow halos on leaves. |
| Leaf smut | Low | Caused by Entyloma oryzae. Forms small angular black spots on both surfaces of the leaf blade. |

---

## Model Architecture

Feature extraction and classification are handled by a two-stage pipeline.

**Stage 1 - Feature Extraction**

EfficientNetB0 pretrained on ImageNet is used as a fixed feature extractor. The final pooling layer outputs a 1280-dimensional feature vector for each image.

**Stage 2 - Voting Classifier**

A soft-voting ensemble of three classifiers trained on the extracted features:
- SVM with RBF kernel (C=1.0)
- Random Forest (100 estimators)
- XGBoost (100 estimators)

The soft voting strategy averages the predicted class probabilities from all three classifiers before making the final decision, which produces the per-class confidence scores shown in the bar chart and pie chart in the application.

---

## Dataset

**Source:** vbookshelf/rice-leaf-diseases (Kaggle)

The dataset contains 120 images across 3 disease classes. EfficientNetB0 feature extraction with an ensemble classifier performs well on limited data due to the rich pretrained representations from ImageNet.

---

## Application Features

- Image upload supporting JPG, JPEG, and PNG formats
- Predicted disease label with confidence percentage
- Severity rating for the detected disease
- Disease description and recommended treatment
- Early detection alert when confidence falls below 60%
- Per-class confidence bar chart
- Probability distribution pie chart
- Wavelet feature decomposition viewer using db1 wavelet at level 2

---

## Screenshots

**Prediction Results**

The interface shows the predicted disease, confidence score, severity, and number of classes evaluated. Below the metrics, a description card provides disease information and treatment recommendations. The confidence scores section displays both a horizontal bar chart for per-class comparison and a pie chart for probability distribution.

Sample output from the live application:
- Predicted Disease: Leaf smut
- Confidence: 67.1%
- Severity: Low
- Classes Evaluated: 3

---

## Project Structure

```
rice-disease-detector/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── classifier.pkl
    ├── classes.pkl
    ├── label_map.pkl
    └── efficientnet_extractor.h5
```

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/rice-disease-detector.git
cd rice-disease-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Training the Model

To retrain the model from scratch, open `final_codee_updated.ipynb` in Google Colab and run both cells. The notebook will:

1. Install all required packages
2. Download the dataset via KaggleHub
3. Extract EfficientNetB0 features image by image
4. Train the Voting Classifier
5. Save all model files to a `models/` folder
6. Zip and download the folder to your machine

Place the extracted `models/` folder in the project root before running the app.

---

## Deployment on Streamlit Community Cloud

1. Push all files including the `models/` folder to a public GitHub repository
2. Go to share.streamlit.io and sign in with GitHub
3. Click New App and select the repository
4. Set the main file path to `app.py`
5. Click Deploy and wait 3 to 5 minutes for the build to complete

Note: large `.h5` and `.pkl` files should be tracked with Git LFS if they exceed 50MB.

---

## Docker Deployment

```bash
docker-compose up --build
```

Open http://localhost:8501 in your browser.

---

## Requirements

```
streamlit==1.35.0
numpy==1.26.4
opencv-python-headless==4.9.0.80
Pillow==10.3.0
matplotlib==3.8.4
PyWavelets==1.6.0
scikit-learn==1.4.2
xgboost==2.0.3
tensorflow-cpu==2.16.1
```

---

## Tech Stack

- Python 3.10
- Streamlit
- TensorFlow / Keras
- scikit-learn
- XGBoost
- OpenCV
- PyWavelets
- Docker
