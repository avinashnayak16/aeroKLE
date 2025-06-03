import os
from algorithm.object_detector import YOLOv7
import rospy
from std_msgs.msg import Float32MultiArray, String
from utils.detections import draw
import json
import cv2
from cv_bridge import CvBridge
import imutils
import numpy as np
from mss import mss
from PIL import Image



mtx = np.array([[845.549571023190, 0, 978.4245716999010],[0,848.221453033451, 547.499622688556], [0,0,1]]) 
dist = np.array([[-0.340671222,0.110426603,-.000867987573,0.000189669273,-0.0160049526]]) #0.309794503988986,0.100538129392808

sct = mss()
image = None
store = None
frame_counter = 0
image_folder = "images"


def callback(data):
    global store
    rospy.loginfo("Received: %s", data.data)
    store = data.data



yolov7 = YOLOv7()
yolov7.load('vic2.pt', classes='test.yaml', device='cpu') # use 'gpu' for CUDA GPU inference


rospy.init_node('object_detector')

pub = rospy.Publisher('/detections/centroid', Float32MultiArray, queue_size=10)
pub2 = rospy.Publisher('/detections/class1', String, queue_size=10)

rospy.Subscriber('my_topic', String, callback)



try:
    while True:
        w, h = 640, 480
        monitor = {'top': 180, 'left': 500, 'width': w, 'height': h}
        img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
        img_np = np.array(img)
        frame = img_np
        
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        #frame = cv2.undistort(frame, mtx, dist, None, mtx)    
        detections = yolov7.detect(frame)
        detected_frame = draw(frame, detections)
        pub2.publish(String(f"{detections}"))
        

        frame_filename = os.path.join(image_folder, f"frame_{frame_counter}.jpg")
        if store is not None:
            print(store)
            cv2.imwrite(frame_filename, detected_frame)
            frame_counter += 1
            store = None
        
        cv2.imshow('webcam', detected_frame)
        cv2.waitKey(1)
        
except KeyboardInterrupt:
    pass
webcam.release()
print('[+] webcam closed')
yolov7.unload()
