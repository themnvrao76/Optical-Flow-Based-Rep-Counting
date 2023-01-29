import cv2 as cv
import cv2
import numpy as np
import time
import argparse
import shutil
import os
from tensorflow import keras


if __name__ == '__main__':

    model=keras.models.load_model("OpticalFlow_Conv2D.h5")
    CATEGORIES=['NO POSE', 'PUSHDOWN', 'PUSHUP']
    file="Squat.mp4"
    cap = cv.VideoCapture(file)
    IMG_SIZE = (256, 256)
    maxresult=[]
    ret, first_frame = cap.read()
    first_frame = cv.resize(first_frame, IMG_SIZE)

    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    i = 0
    number=0
    c=0
    count=[]

    def contains(a, b):
        '''
        Checks if array b is a subsequence of array a
        :param a:
        :param b:
        :return: returns True if array b is a subsequence of a, else False
        '''
        for i in range(a.shape[0] - b.shape[0] + 1):
            if (a[i:i + b.shape[0]] == b).all():
                return True
        return False
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv.resize(frame, IMG_SIZE)
        f2=frame
        f2=cv2.resize(f2,(640,640))
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 50, 3, 5, 1.1, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

       # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        rgbarray=np.array(rgb)
        rgbarray = np.expand_dims(rgbarray, axis=0)
        # print(rgbarray.shape)
        prednum=np.argmax(model.predict(rgbarray))
        Prediction=CATEGORIES[prednum]

        # print(Prediction)
        maxresult.append(prednum)
        if len(maxresult)>8:
            print(maxresult)
            Score=contains(np.array(maxresult),np.array([1,1,1,1,2,2,2,2]))
            if Score == True:
                c=c+1
                # print(c)
                maxresult.clear()
            
        cv2.putText(f2, f"{c}",(100,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0, 0, 255, 255),3)


        cv.imshow("dense optical flow", rgb)
        cv.imshow("input", f2)
        prev_gray = gray
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()