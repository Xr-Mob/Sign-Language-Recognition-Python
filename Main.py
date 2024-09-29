import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
drawing = mp.solutions.drawing_utils

def mediapipe_dectection(image,model):
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

def render_landmarks(image,results):
    drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def render_formatted_landmarks(image,results):
    drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            drawing.DrawingSpec(color=(55, 44, 230), thickness=1, circle_radius=1),
                            drawing.DrawingSpec(color=(55, 44, 230), thickness=1, circle_radius=1)
                            )
    
    drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                            drawing.DrawingSpec(color=(120, 150, 230), thickness=1, circle_radius=1),
                            drawing.DrawingSpec(color=(120, 110, 230), thickness=1, circle_radius=1)
                            )
    drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                            )
    drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                            drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                            )
    
def extract_keypoints(results):
    pose = np.array([[result.x, result.y, result.z, result.visibility] for result in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros((33*4)) 
    face = np.array([[result.x, result.y, result.z] for result in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros((468*4))
    lh = np.array([[result.x, result.y, result.z] for result in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros((21*3))
    rh = np.array([[result.x, result.y, result.z] for result in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros((21*3))
    return np.concatenate([pose,face,lh,rh])

capture = cv.VideoCapture(0)
#Set Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():

        #Getting feed from camera
        retBool, frame = capture.read()

        #Make detection
        image, results = mediapipe_dectection(frame,holistic)
        print(results)

        #Draw Landmarks
        render_formatted_landmarks(image,results)

        #Play video frame
        cv.imshow('Feed', image)

        #Break gracefully
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()