import cv2 as cv
import numpy as np
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

#Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

#Action that we try to detect
actions = np.array(['hello','thanks','iloveyou'])

#Thirty videos worth of data
no_sequence = 30

#Videos are going to be 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequence):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

capture = cv.VideoCapture(0)
#Set Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequence):
            for frame_num in range(sequence_length):

                #Getting feed from camera
                retBool, frame = capture.read()

                #Make detection
                image, results = mediapipe_dectection(frame,holistic)
                print(results)

                #Draw Landmarks
                render_formatted_landmarks(image,results)

                #Apply wait logic
                if frame_num == 0:
                    cv.putText(image, 'STARTING COLLECTION', (120, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv.LINE_AA)
                    cv.putText(image, 'Collection frames for {} Video Number {}'.format(action,sequence), (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

                    #show to screen
                    cv.imshow('Feed', image)
                    cv.waitKey(2000)

                else:
                    cv.putText(image, 'Collection frames for {} Video Number {}'.format(action,sequence), (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                    cv.imshow('Feed', image)

                #Save keypoints as numpy array for future use
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                #Break gracefully
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

    capture.release()
    cv.destroyAllWindows()