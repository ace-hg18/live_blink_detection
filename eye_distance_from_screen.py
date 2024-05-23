# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:50:02 2023

@author: hguro
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:57:54 2023

@author: hguro
"""
import cv2
import numpy as np
import mediapipe as mp
from Eye_Dector_Module import EyeDetector as EyeDet
import glob
import os 
from pathlib import Path
import time
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
import numpy as np
# import datetime as dt
# from threading import Thread,Event
# import pandas as pd
from numpy import linalg as LA
# from scipy.spatial import distance as dist


EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
POSE_LMS_NUMS = [70,107,336,300,64,294,61,278,17,152,33,133,362,263]
LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
def _get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) \
                        for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks
    
    return biggest_face

# def animate(i, xs, ys,ear):

#     # Read temperature (Celsius) from TMP102
#     # Add x and y to lists
#     xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
#     ys.append(ear)

#     # Limit x and y lists to 20 items
#     xs = xs[-20:]
#     ys = ys[-20:]

#     # Draw x and y lists
#     ax.clear()
#     ax.plot(xs, ys)

#     # Format plot
#     plt.xticks(rotation=45, ha='right')
#     plt.subplots_adjust(bottom=0.30)
#     plt.title('TMP102 Temperature over Time')
#     plt.ylabel('Temperature (deg C)')

# Set up plot to call animate() function periodically



def main():
    ear_left = 0
    ear_right = 0    
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # set OpenCV optimization to True
        except:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected")
    
    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               refine_landmarks=True)
    
    Eye_det = EyeDet(show_processing=False)
    # cap = cv2.VideoCapture(0)
    
    t0 = int(round(time.time() * 1000))

    # print('before while loop')
    images = glob.glob('C:/autoyos/blink_profile_headpose_project/distance_from_screen/images/*')
    for i in images:
    # while True:

        # print('in while loop')
        # success, frame = cap.read()
        
        frame = cv2.imread(i)
        # if not success:  # if a frame can't be read, exit the program
        #     print("Can't receive frame from camera/stream end")
        #     break
        # img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        # find the faces using the face mesh model
        lms = detector.process(gray).multi_face_landmarks

        if lms:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = _get_landmarks(lms)
            
            i1 = (landmarks[LEFT_IRIS_NUM, :2] * frame_size)
            i2 = (landmarks[RIGHT_IRIS_NUM, :2] * frame_size)
            
            
            print(f'landmarks[LEFT_IRIS_NUM, :2]  = {landmarks[LEFT_IRIS_NUM, :2] }')
            
            pi1 = (i1[0],i1[1])
            pi2 = (i2[0],i2[1])
            
            # dist12 = dist.euclidean(pi1, pi2)
            dist12 = LA.norm(i1 - i2)
            
            d2s_calc = (np.power(dist12,-1.16))*5359
            
            print(f'len(landmarks) = {len(landmarks)}')
            
            # print(f'pi1 = {pi1} \n i2 = {pi2} \n dist = {dist12} \n ')
            
            print(f"eye_dist =  {dist12} \n manual distance from screen = {Path(i).stem.split('_')[0]} \n calculated dist from screen = {d2s_calc}\n ")
            
            
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size)
            
            # ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)
            ear_left,ear_right = Eye_det.get_EAR(frame=gray, landmarks=landmarks)
            # frame = cv2.flip(frame, 2)
            if ear_left is not None:
                cv2.putText(frame, "EAR LE:" + str(round(ear_left, 3)), (10, 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
            
            if ear_right is not None:
                cv2.putText(frame, "EAR RE:" + str(round(ear_right, 3)), (10, 80),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)

        
        # for n in range(len(landmarks)):
        for n in POSE_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
            cv2.putText(frame,f'{n}', (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.imwrite(f'C:/autoyos/blink_profile_headpose_project/distance_from_screen/eye_distance/{Path(i).stem}_pose1.png',frame)
        # cv2.imshow("Press 'q' to terminate", frame)
        time_millis = int(round(time.time() * 1000))
        # print(f' shape = {img1.shape}')
        # if cv2.waitKey(1) == ord('q'):
        # #     kill_p = True
        #     break
        
    # cap.release()
    # cv2.destroyAllWindows()
    
    # return df


main()
