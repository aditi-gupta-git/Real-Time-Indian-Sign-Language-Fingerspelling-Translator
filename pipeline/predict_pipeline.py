import numpy as np
import mediapipe as mp

from utils import load_object, extract_hand_keypoints
from logger import logging

MODEL_PATH = "models/isl_model.pkl"
ENC_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"


class PredictPipeline:
    def __init__(self):
        logging.info("Loading model, encoder, scaler...")

        self.model = load_object(MODEL_PATH)
        self.encoder = load_object(ENC_PATH)
        self.scaler = load_object(SCALER_PATH)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

        logging.info("Prediction pipeline ready")

    def predict_frame(self, frame):
        rgb = frame[:, :, ::-1].copy()
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, 0.0, frame

        for hand_lm in result.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
            )

            kp = extract_hand_keypoints(hand_lm).reshape(1, -1)
            kp_scaled = self.scaler.transform(kp)

            probs = self.model.predict_proba(kp_scaled)[0]
            idx = np.argmax(probs)
            conf = float(probs[idx])
            letter = self.encoder.inverse_transform([idx])[0]

            return letter, conf, frame

        return None, 0.0, frame

    def close(self):
        self.hands.close()