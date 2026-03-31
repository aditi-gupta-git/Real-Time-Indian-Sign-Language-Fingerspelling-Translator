import sys, os

project_root = r"D:\Project\IndianSignLanguage"
sys.path.insert(0, project_root)

import cv2
import time
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

CONF_THRESHOLD   = 0.80
SAME_LETTER_SECS = 1.5
SPACE_SECS       = 2.5

def run_app():
    pipeline = PredictPipeline()
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    word         = ""
    sentence     = []
    cur_letter   = None
    letter_start = time.time()
    last_seen    = time.time()
    confirmed    = False

    print("✅ Running — Q to quit | BACKSPACE to delete")
    logging.info("App started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        letter, conf, frame = pipeline.predict_frame(frame)
        now = time.time()

        if letter and conf >= CONF_THRESHOLD:
            last_seen = now
            if letter == cur_letter:
                held = now - letter_start
                if held >= SAME_LETTER_SECS and not confirmed:
                    word     += letter
                    confirmed = True
                    logging.info(f"Confirmed: {letter}")
            else:
                cur_letter   = letter
                letter_start = now
                confirmed    = False
        else:
            if cur_letter and (now - last_seen) >= SPACE_SECS:
                if word:
                    sentence.append(word)
                    word = ""
                cur_letter = None
                confirmed  = False

        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-170), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        if letter and conf >= CONF_THRESHOLD:
            cv2.putText(frame, f"{letter}",
                        (30, h-95),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.5,
                        (0, 255, 150), 4)
            cv2.putText(frame, f"{conf*100:.0f}%",
                        (130, h-95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                        (0, 255, 150), 3)

            held    = min(now - letter_start, SAME_LETTER_SECS)
            bar_w   = int((held / SAME_LETTER_SECS) * 320)
            cv2.rectangle(frame, (30, h-60), (350, h-38), (70, 70, 70), -1)
            cv2.rectangle(frame, (30, h-60), (30+bar_w, h-38),
                          (0, 255, 150) if not confirmed else (255, 200, 0), -1)
            cv2.putText(frame,
                        "Added!" if confirmed else "Hold to confirm",
                        (360, h-42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 1)
        else:
            cv2.putText(frame, "No hand detected",
                        (30, h-100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (80, 80, 80), 2)

        cv2.putText(frame, f"Word: {word}",
                    (30, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
        cv2.putText(frame,
                    f"Sentence: {' '.join(sentence)}",
                    (30, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 255), 2)

        cv2.imshow("ISL Fingerspelling Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 8:
            word = word[:-1]
        elif key == ord(' '):
            if word:
                sentence.append(word)
                word = ""

    cap.release()
    cv2.destroyAllWindows()
    pipeline.close()
    final = ' '.join(sentence)
    print(f"\nFinal sentence: {final}")
    logging.info(f"Final: {final}")

if __name__ == "__main__":
    run_app()