import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

from utils import save_object
from logger import logging
from exception import CustomException


def train_model(
    data_csv="data/isl_keypoints.csv",
    model_dir="models",
):
    try:
        logging.info("Loading dataset...")
        df = pd.read_csv(data_csv)

        print(f"Dataset: {df.shape[0]} rows, {df['label'].nunique()} classes")
        print(f"Classes: {sorted(df['label'].unique())}")

        X = df.drop("label", axis=1).values.astype(np.float32)
        y = df["label"].values

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y_enc,
            test_size=0.2,
            random_state=42,
            stratify=y_enc,
        )

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")
        print("\nTraining ANN...")

        model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=True,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nTest Accuracy: {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        logging.info(f"Test accuracy: {acc * 100:.2f}%")

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "isl_model.pkl")
        enc_path = os.path.join(model_dir, "label_encoder.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        save_object(model_path, model)
        save_object(enc_path, le)
        save_object(scaler_path, scaler)

        print(f"\nModel saved -> {model_path}")
        print(f"Encoder saved -> {enc_path}")
        print(f"Scaler saved -> {scaler_path}")

        return model, le, scaler, acc

    except Exception as e:
        raise CustomException(e, sys)