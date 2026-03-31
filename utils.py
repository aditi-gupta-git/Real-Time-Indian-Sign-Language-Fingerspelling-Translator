import os
import sys
import pickle
import numpy as np

from logger import logging
from exception import CustomException


def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Saved object at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def extract_hand_keypoints(hand_landmarks):
    kp = []

    for lm in hand_landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z])

    wx, wy, wz = kp[0], kp[1], kp[2]

    normalized_kp = [
        kp[i] - wx if i % 3 == 0 else
        kp[i] - wy if i % 3 == 1 else
        kp[i] - wz
        for i in range(len(kp))
    ]

    distances = [
        np.sqrt(normalized_kp[i] ** 2 + normalized_kp[i + 1] ** 2)
        for i in range(0, len(normalized_kp), 3)
    ]

    max_dist = max(distances) if max(distances) > 0 else 1.0
    normalized_kp = [k / max_dist for k in normalized_kp]

    return np.array(normalized_kp, dtype=np.float32)