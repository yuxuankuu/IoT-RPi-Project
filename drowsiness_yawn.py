from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from gpiozero import LED, Buzzer
from signal import pause
import numpy as np
import argparse
import imutils
import time
import dlib    
import cv2
import os
import datetime
import logging

# 設定日誌記錄
logging.basicConfig(
    filename="alert_log.log",
    level=logging.INFO,
    format="%(asctime)s - ALERT: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class control:
    def __init__(self): 
        self.led_green = LED(17) # 綠色 LED 接 GPIO 17
        self.led_red = LED(27) # 紅色 LED 接 GPIO 27
        self.buzzer = Buzzer(23) # 蜂鳴器接 GPIO 23
    def green_on(self):
        self.led_green.on()
    def red_on(self):
        self.led_red.on()
    def green_off(self):
        self.led_green.off()
    def red_off(self):
        self.led_red.off()
    def green_blink(self):
        self.led_green.blink(on_time=0.1, off_time=0.1)
    def red_blink(self):
        self.led_red.blink(on_time=0.75, off_time=0.75)
    def turn_off_all(self):
        self.led_green.off()
        self.led_red.off()
    def beep_on(self):
        self.buzzer.beep(on_time=0.3, off_time=0.3, n=3)
    def beep_off(self):
        self.buzzer.off()
    

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) #眼睛垂直距離
    B = dist.euclidean(eye[2], eye[4]) #眼睛垂直距離
    C = dist.euclidean(eye[0], eye[3]) #眼睛水平距離

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1]) #上下嘴唇的垂直距離
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# 初始化變數與模型
EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 25
YAWN_THRESH = 25
COUNTER = 0
alert_count = 0
control = control()

# 載入臉部檢測器與特徵點模型
control.green_blink()
print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")   
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 開始視訊串流
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
control.green_on()

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=650, height=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                control.red_on()
                control.beep_on()
                alert_count += 1
                cv2.putText(frame, "DROWSINESS ALERT", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                logging.info(f"Drowsiness detected - Alert Count: {alert_count}")

        else:
            COUNTER = 0
            control.red_off()

        if (distance > YAWN_THRESH):
            control.red_on()
            control.beep_on()
            alert_count += 1
            cv2.putText(frame, "YAWN ALERT", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            logging.info(f"Yawn detected - Alert Count: {alert_count}")

        else:
            control.red_off()

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (500, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.waitKey(1)   

    #if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) <1:
        #break
control.turn_off_all()
cv2.destroyAllWindows()
# cv2.destroy(frame)
vs.stop()