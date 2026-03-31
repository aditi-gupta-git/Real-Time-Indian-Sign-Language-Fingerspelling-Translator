import os
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
import sys

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Saved object at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)

def extract_hand_keypoints(hand_landmarks):
    """Extract + normalize 21 landmarks → 63 features."""
    kp = []
    for lm in hand_landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z])

    # Subtract wrist (landmark 0) to make position-invariant
    wx, wy, wz = kp[0], kp[1], kp[2]
    kp = [
        kp[i] - wx if i % 3 == 0 else
        kp[i] - wy if i % 3 == 1 else
        kp[i] - wz
        for i in range(len(kp))
    ]

    # Scale by max distance so size-invariant
    distances = [np.sqrt(kp[i]**2 + kp[i+1]**2) for i in range(0, len(kp), 3)]
    max_dist = max(distances) if max(distances) > 0 else 1
    kp = [k / max_dist for k in kp]

    return np.array(kp, dtype=np.float32)