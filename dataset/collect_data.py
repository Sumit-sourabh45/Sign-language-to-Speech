import cv2
import mediapipe as mp
import csv
import os

gesture_name = input("Enter gesture name: ")
num_samples = 70

if not os.path.exists("dataset"):
    os.makedirs("dataset")

file_path = f"dataset/{gesture_name}.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0

with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)

    while count < num_samples:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                writer.writerow(landmarks)
                count += 1

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Collecting {gesture_name}: {count}/{num_samples}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()