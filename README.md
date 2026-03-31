# Real-time Indian Sign Language Fingerspelling Translator

A webcam-based computer vision project that recognizes Indian Sign Language (ISL) alphabet hand gestures in real time using MediaPipe hand landmarks and an ANN classifier.

## Project Overview

This project performs real-time fingerspelling recognition for ISL alphabet gestures (A-Z). The system detects a hand from webcam frames, extracts 21 hand landmarks using MediaPipe, converts them into 63 normalized features, and predicts the corresponding letter using a trained Artificial Neural Network (ANN).

The application also builds words and simple sentences by confirming stable predictions over time.

## Features

- Real-time webcam-based hand gesture recognition
- MediaPipe hand landmark extraction
- 63-feature normalized hand keypoint representation
- ANN-based alphabet classification
- Confidence thresholding for stable predictions
- Hold-to-confirm mechanism for letter addition
- Word and sentence building in real time
- Modular project structure with reusable Python components

## Project Structure

```text
INDIANSIGNLANGUAGE/
├── data/
│   ├── raw_images/
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── isl_keypoints.csv
├── logs/
├── models/
│   ├── isl_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── src/
│   ├── components/
│   │   ├── data_preparation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── app.py
├── prepare_data.py
├── train.py
├── requirements.txt
├── setup.py
└── README.md
```

## Dataset Preparation

The dataset consists of alphabet-wise folders containing gesture images for A-Z.  
MediaPipe is used in static image mode to detect one hand per image.  
For each valid image:

- 21 hand landmarks are extracted
- Each landmark contributes x, y, z values
- Total features = 21 x 3 = 63
- Features are normalized relative to the wrist and scaled by hand size
- The resulting dataset is saved as `data/isl_keypoints.csv`

## Model Training

The classifier is trained using:

- `StandardScaler` for feature scaling
- `LabelEncoder` for converting A-Z into class labels
- `MLPClassifier` from scikit-learn as the ANN model

### Model configuration

- Hidden layers: `(256, 128, 64)`
- Activation: `relu`
- Optimizer: `adam`
- Early stopping enabled
- Validation fraction: `0.1`

### Reported performance

- Dataset size: 5277 samples
- Number of classes: 26
- Test accuracy: 96.69%

## Real-Time Inference Flow

1. Webcam frame is captured using OpenCV
2. Hand landmarks are detected using MediaPipe
3. 63 normalized keypoint features are extracted
4. Features are scaled using the saved scaler
5. ANN predicts the letter and probability scores
6. If confidence is above threshold, the prediction is shown
7. If the same letter is held for a fixed duration, it is added to the current word
8. If no hand is detected for some time, the current word is added to the sentence

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare dataset

```bash
python prepare_data.py
```

### 3. Train model

```bash
python train.py
```

### 4. Run the real-time app

```bash
python app.py
```

## Controls

- `Q` -> Quit the application
- `Backspace` -> Remove last letter from current word
- `Space` -> Push current word into sentence

## Limitations

- The project recognizes isolated alphabet gestures, not full continuous sign language sentences
- Performance depends on lighting, hand visibility, and webcam quality
- Some letters may be confused more often than others due to dataset imbalance or visual similarity
- MediaPipe may fail to detect hands in some images or live frames
- Repeated letters in words may need deliberate hand separation between signs

## Future Improvements

- Add temporal smoothing using a sequence model like LSTM
- Add support for dynamic gestures
- Improve repeated-letter handling
- Expand beyond alphabet-level recognition into word-level sign recognition
- Build a Streamlit or web-based interface
