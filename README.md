# REAL-TIME ISL FINGER SPELLING TRANSLATOR

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-023F7F?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0099BC?style=for-the-badge&logo=google&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Project Overview

This project is a **real-time webcam-based Indian Sign Language (ISL) finger spelling translator** that recognizes hand gestures for alphabets using **MediaPipe hand landmarks** and a trained **Artificial Neural Network (ANN)** classifier.

The system captures video from a webcam, detects a hand, extracts 21 hand landmarks, converts them into normalized numerical features, and predicts the corresponding alphabet in real time. Predicted letters are then combined over time to form words.

### Main Features

- Real-time webcam capture using OpenCV
- Hand landmark detection using MediaPipe
- Keypoint-based feature extraction
- ANN-based alphabet classification
- Live prediction with confidence score
- Word building from consecutive letter predictions
- Modular Python project structure for cleaner submission

## Dataset Information

This project uses the **Indian Sign Language (ISLRTC referred)** dataset from Kaggle. The dataset contains Indian Sign Language images for English alphabets **A-Z** and numbers **0-9**. [Kaggle dataset link](https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred). 

### Dataset Source
- **Name:** Indian Sign Language (ISLRTC referred)
- **Platform:** Kaggle
- **Link:** https://www.kaggle.com/datasets/atharvadumbre/indian-sign-language-islrtc-referred
- **Classes available:** Alphabets A-Z and digits 0-9

### Dataset Usage in This Project
For this project, only the **alphabet gestures (A-Z)** are used for training the finger spelling translator.

## Project Workflow

1. Collect or download ISL image data.
2. Use MediaPipe to detect hand landmarks from each image.
3. Extract 63 features 
4. Normalize keypoints relative to the wrist.
5. Save processed features into a CSV dataset.
6. Train an ANN classifier on the processed dataset.
7. Save the trained model, scaler, and label encoder.
8. Run the webcam application for real-time prediction.
9. Combine stable predictions into words.

## Project Structure

```text
IndianSignLanguage/
├── app.py                      # Main webcam application
├── setup.py                    # Package setup file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── logger.py                   # Logging configuration
├── exception.py                # Custom exception handling
├── utils.py                    # Utility functions
│
├── data/                       # Dataset files and processed CSV
├── logs/                       # Log files
├── models/                     # Saved model artifacts
│   ├── isl_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
├── components/                 # Training and data preparation logic
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── model_trainer.py
│   ├── prepare_data.ipynb
│   └── training.ipynb
│
├── pipeline/                   # Real-time prediction pipeline
│   ├── __init__.py
│   └── predict_pipeline.py
│
├── prepare_data.py             # Script to prepare dataset
└── train.py                    # Script to train the model
```

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib
- TQDM

## Environment Setup

These steps assume the evaluator has no prior context and is setting up the project from scratch.

### Prerequisites

Make sure the following are installed:

- Python 3.10 or Python 3.11
- pip
- Webcam
- Git (optional, if cloning from GitHub)

### Step 1: Clone the Repository

```bash
git clone <your-repository-link>
cd IndianSignLanguage
```

If the project is already downloaded as a ZIP, extract it and open the project folder in terminal.

### Step 2: Create a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install the Project in Editable Mode

If not already handled by `-e .` in `requirements.txt`, run:

```bash
pip install -e .
```

## Configuration

Before running the project, check the following:

- Ensure the `models/` folder exists.
- Ensure trained files are available:
  - `models/isl_model.pkl`
  - `models/label_encoder.pkl`
  - `models/scaler.pkl`
- If these files are not available, first run data preparation and training.

## Step-by-Step Execution

### Option 1: Run the Full Pipeline from Scratch

#### Step A: Prepare Dataset Features

This script processes gesture images, extracts MediaPipe landmarks, and creates a CSV file for training.

```bash
python prepare_data.py
```

Expected output:
- Processed CSV file saved in `data/`
- Keypoint features for each gesture sample

#### Step B: Train the Model

This script trains the ANN classifier and saves the trained artifacts.

```bash
python train.py
```

Expected output:
- `models/isl_model.pkl`
- `models/label_encoder.pkl`
- `models/scaler.pkl`

#### Step C: Run the Real-Time Webcam App

```bash
python app.py
```

This launches the webcam-based finger spelling translator.

### Option 2: Run Only the Real-Time App

If trained model files are already present in `models/`, run:

```bash
python app.py
```

## How to Use the App

1. Start the app using `python app.py`.
2. Allow the webcam to open.
3. Show one hand clearly in front of the camera.
4. Make an ISL alphabet gesture.
5. Hold the gesture steady until the prediction is confirmed.
6. The predicted letter will be displayed on screen.
7. Continue showing letters to build a word.
8. Use the on-screen controls if implemented:
   - `Q` to quit
   - `Backspace` to delete last character
   - `Space` to finalize a word


## Notebooks

The notebooks are kept as experimentation evidence and project development support.

- `components/prepare_data.ipynb` — landmark extraction and dataset preparation experiments
- `components/training.ipynb` — training and evaluation experiments

The final executable submission should use the `.py` modules and scripts instead of relying only on notebooks.
