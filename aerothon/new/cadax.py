#!/usr/bin/env python

import device_patches      
import cv2
import os
import numpy as np
import argparse
import sys, getopt
import signal
import imutils
import time
import urllib.request

runner = None
stream_url = 'rtsp://192.168.179.46:8554/fpv_stream' #"http://192.168.43.50:8000/stream.mjpg"          #'rtsp://192.168.179.46:8554/fpv_stream'

# Open video stream as file-like object

show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False
def now():
    return round(time.time() * 1000)    



def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def hello():
    print()

def eye():
    while True:
        try:
            videoCaptureDeviceId = stream_url
            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret, img = camera.read()  # Get the frame
            
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                
                if show_camera:
                    # Show the video stream in a window
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
            else:
                raise Exception("Couldn't initialize selected camera.")
                
        finally:
            camera.release()
            if runner:
                runner.stop()
                
eye()





'''
#!/usr/bin/env python

import device_patches      
import cv2
import os
import numpy as np
import argparse
import sys, getopt
import signal
import imutils
import time
import urllib.request

runner = None
stream_url = 'rtsp://192.168.179.46:8554/fpv_stream'

# Open video stream as file-like object

show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

def now():
    return round(time.time() * 1000)    

def sigint_handler(sig, frame):
    print('Interrupted')
    if runner:
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def eye():
    while True:
        try:
            videoCaptureDeviceId = stream_url
            camera = cv2.VideoCapture(videoCaptureDeviceId)
            ret, frame = camera.read()  # Capture the frame
            if ret:
                backendName = camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) in port %s selected." % (backendName, h, w, videoCaptureDeviceId))
                
            else:
                raise Exception("Couldn't initialize selected camera.")

            if show_camera and ret:
                # Display the frame
                cv2.imshow('edgeimpulse', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            camera.release()  # Release the camera resource
            if runner:
                runner.stop()

eye()
'''