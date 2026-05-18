import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque
import pyttsx3
import threading

# ── Load model ──────────────────────────────────────────────────────────────
model = pickle.load(open("model/gesture_model.pkl", "rb"))

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils
LANDMARK_STYLE  = mp_draw.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3)
CONNECTION_STYLE = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)

# ── TTS engine ───────────────────────────────────────────────────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)   # speaking speed (words per minute)

def speak_async(text):
    """Run TTS in a background thread so the video feed never freezes."""
    def _run():
        engine = pyttsx3.init()          # fresh engine per thread (pyttsx3 is not thread-safe)
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# ── Prediction smoothing ─────────────────────────────────────────────────────
prediction_buffer = deque(maxlen=10)

# ── Sentence builder state ───────────────────────────────────────────────────
sentence        = []          # list of added words
last_added_word = None        # prevents the same word being added twice in a row
stable_counter  = 0
STABLE_THRESHOLD = 20         # frames gesture must be held before it is added
                              # (~20 frames ≈ 0.7 s at 30 fps — tune to taste)

# ── Helper: draw a filled rounded rectangle ──────────────────────────────────
def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(overlay,  (x1 + r, y1 + r), r, color, -1)
    cv2.circle(overlay,  (x2 - r, y1 + r), r, color, -1)
    cv2.circle(overlay,  (x1 + r, y2 - r), r, color, -1)
    cv2.circle(overlay,  (x2 - r, y2 - r), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Controls  →  S = Speak sentence | C = Clear sentence | B = Backspace | Q = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.flip(frame, 1)   # disabled — training data was collected without flip
    h, w, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result    = hands.process(frame_rgb)

    current_prediction = None

    # ── Hand detection & prediction ──────────────────────────────────────────
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Collect 63 landmark values (21 points × x,y,z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            raw_prediction = model.predict([landmarks])[0]
            prediction_buffer.append(raw_prediction)
            final_prediction  = max(set(prediction_buffer), key=prediction_buffer.count)
            current_prediction = final_prediction

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                LANDMARK_STYLE, CONNECTION_STYLE
            )

            # ── Gesture label ────────────────────────────────────────────────
            draw_rounded_rect(frame, 8, 8, 320, 58, 8, (20, 20, 20), alpha=0.6)
            cv2.putText(frame, final_prediction.replace("_", " "),
                        (16, 44), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 180), 2)

            # ── Confidence bar ───────────────────────────────────────────────
            # Needs model trained with probability=True  (see train_model.py)
            try:
                proba      = model.predict_proba([landmarks])[0]
                confidence = float(max(proba)) * 100

                bar_x, bar_y, bar_h = 8, 68, 14
                bar_max_w = 250
                filled_w  = int(confidence / 100 * bar_max_w)

                bar_color = (0, 220, 80) if confidence >= 75 else \
                            (0, 165, 255) if confidence >= 50 else (0, 60, 220)

                draw_rounded_rect(frame, bar_x, bar_y,
                                  bar_x + bar_max_w, bar_y + bar_h, 4, (40, 40, 40), 0.7)
                if filled_w > 6:
                    draw_rounded_rect(frame, bar_x, bar_y,
                                      bar_x + filled_w, bar_y + bar_h, 4, bar_color, 0.9)

                cv2.putText(frame, f"{confidence:.0f}%",
                            (bar_x + bar_max_w + 8, bar_y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            except AttributeError:
                # model was not trained with probability=True — skip bar silently
                pass

            # ── Hold-to-add progress ─────────────────────────────────────────
            if final_prediction != last_added_word:
                stable_counter += 1

                prog_w = int(stable_counter / STABLE_THRESHOLD * 250)
                draw_rounded_rect(frame, 8, 92, 258, 108, 4, (40, 40, 40), 0.7)
                if prog_w > 4:
                    draw_rounded_rect(frame, 8, 92, 8 + prog_w, 108, 4, (80, 180, 255), 0.9)
                cv2.putText(frame, "Hold to add to sentence",
                            (8, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

                if stable_counter >= STABLE_THRESHOLD:
                    sentence.append(final_prediction.replace("_", " "))
                    last_added_word = final_prediction
                    stable_counter  = 0
            else:
                stable_counter = 0   # same as last added word — don't add again

    else:
        # No hand in frame — reset the hold counter so user must re-hold
        stable_counter     = 0
        last_added_word    = None
        prediction_buffer.clear()

    # ── Sentence display panel ───────────────────────────────────────────────
    panel_y = h - 90
    draw_rounded_rect(frame, 0, panel_y, w, h, 0, (15, 15, 15), alpha=0.75)

    cv2.putText(frame, "SENTENCE",
                (12, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    sentence_text = " ".join(sentence) if sentence else "— start signing —"
    # If sentence is very long, show only the last portion
    if len(sentence_text) > 55:
        sentence_text = "..." + sentence_text[-52:]

    cv2.putText(frame, sentence_text,
                (12, panel_y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

    # Controls hint
    hint = "  S=Speak    C=Clear    B=Backspace    Q=Quit"
    cv2.putText(frame, hint,
                (12, panel_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)

    # ── Keyboard controls ────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s') or key == ord('S'):
        if sentence:
            text_to_speak = " ".join(sentence)
            print(f"Speaking: {text_to_speak}")
            speak_async(text_to_speak)
        else:
            print("Nothing to speak yet.")

    elif key == ord('c') or key == ord('C'):
        sentence.clear()
        last_added_word = None
        stable_counter  = 0
        print("Sentence cleared.")

    elif key == ord('b') or key == ord('B'):
        if sentence:
            removed = sentence.pop()
            last_added_word = sentence[-1].replace(" ", "_") if sentence else None
            print(f"Removed: {removed}")

    cv2.imshow("Gesture → Speech", frame)

cap.release()
cv2.destroyAllWindows()