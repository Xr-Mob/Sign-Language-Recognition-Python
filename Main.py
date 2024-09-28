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
        render_landmarks(image,results)

        #Play video frame
        cv.imshow('Feed', image)

        #Break gracefully
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
        
    capture.release()
    cv.destroyAllWindows()