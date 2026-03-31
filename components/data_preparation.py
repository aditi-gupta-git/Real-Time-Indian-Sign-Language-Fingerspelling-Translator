import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

from utils import extract_hand_keypoints
from logger import logging

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def process_dataset(
    image_dataset_dir="data/raw_images",
    output_csv="data/isl_keypoints.csv",
    samples_per_class=300,
):
    all_rows = []
    failed = 0

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:

        for letter in LETTERS:
            folder = os.path.join(image_dataset_dir, letter)

            if not os.path.exists(folder):
                print(f"Skipping {letter} - folder not found: {folder}")
                continue

            images = [
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ][:samples_per_class]

            count = 0

            for img_file in tqdm(images, desc=f"Letter {letter}"):
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.copyMakeBorder(
                    img, 20, 20, 20, 20,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    kp = extract_hand_keypoints(result.multi_hand_landmarks[0])
                    all_rows.append([letter] + kp.tolist())
                    count += 1
                else:
                    failed += 1

            print(f"{letter}: {count}/{len(images)} extracted")
            logging.info(f"{letter}: {count} samples")

    cols = ["label"] + [f"f{i}" for i in range(63)]
    df = pd.DataFrame(all_rows, columns=cols)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved {len(df)} rows -> {output_csv}")
    print(f"Missed: {failed} images (MediaPipe could not detect hand)")
    logging.info(f"Done: {len(df)} rows, {failed} failures")

    return df