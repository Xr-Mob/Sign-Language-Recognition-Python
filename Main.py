import cv2 as cv
import numpy as np
import os
import mediapipe as mp
from matplotlib import pyplot as plt
import time
from collections import deque
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -------- Setup -------- #
mp_holistic = mp.solutions.holistic
drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequence = 30
sequence_length = 30
model_path = 'sign_language_model.h5'

# -------- MediaPipe Helpers -------- #
def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv.cvtColor(image, cv.COLOR_RGB2BGR), results

def render_formatted_landmarks(image, results):
    drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            drawing.DrawingSpec(color=(55, 44, 230), thickness=1, circle_radius=1),
                            drawing.DrawingSpec(color=(55, 44, 230), thickness=1, circle_radius=1))
    drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            drawing.DrawingSpec(color=(120, 150, 230), thickness=1, circle_radius=1),
                            drawing.DrawingSpec(color=(120, 110, 230), thickness=1, circle_radius=1))
    drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# -------- Step 1: Data Collection -------- #
if not os.path.exists(DATA_PATH):
    print("Starting data collection...")
    for action in actions:
        for sequence in range(no_sequence):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    capture = cv.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequence):
                for frame_num in range(sequence_length):
                    ret, frame = capture.read()
                    image, results = mediapipe_detection(frame, holistic)
                    render_formatted_landmarks(image, results)

                    if frame_num == 0:
                        cv.putText(image, 'STARTING COLLECTION', (120, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv.LINE_AA)
                        cv.putText(image, f'Collecting {action} Seq {sequence}', (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                        cv.imshow('Feed', image)
                        cv.waitKey(2000)
                    else:
                        cv.putText(image, f'Collecting {action} Seq {sequence}', (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                        cv.imshow('Feed', image)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)

                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
    capture.release()
    cv.destroyAllWindows()
    print("Data collection completed.")
else:
    print("Data already collected. Skipping data collection.")

# -------- Step 2: Train Model if Not Already Trained -------- #
if not os.path.exists(model_path):
    print("Training model...")
    label_map = {label: num for num, label in enumerate(actions)}  
    expected_frame_shape = 1662  # 33*4 + 468*3 + 21*3 + 21*3

    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequence):
            window = []
            valid_sequence = True
            for frame_num in range(sequence_length):
                file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if not os.path.exists(file_path):
                    valid_sequence = False
                    break
                res = np.load(file_path)
                if res.shape[0] != expected_frame_shape:
                    valid_sequence = False
                    break
                window.append(res)

            if valid_sequence:
                sequences.append(window)
                labels.append(label_map[action])
            else:
                print(f"Skipping corrupted or incomplete sequence: {action} {sequence}")

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

    model.save(model_path)
    print("Model trained and saved.")
else:
    print("Model already trained. Loading model...")
    model = tf.keras.models.load_model(model_path)

# -------- Step 3: Real-Time Prediction -------- #
print("Starting real-time sign language recognition...")
sequence = deque(maxlen=30)
sentence = []
threshold = 0.8

cap = cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        render_formatted_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)
            res = model.predict(input_data)[0]
            predicted_action = actions[np.argmax(res)]

            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or predicted_action != sentence[-1]:
                    sentence.append(predicted_action)
                    sentence = sentence[-3:]

            cv.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
            cv.putText(image, ' '.join(sentence), (3, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow('Real-Time Sign Recognition', image)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
