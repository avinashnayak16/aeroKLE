import cv2
import numpy as np
import threading
from datetime import datetime
import time 
from picamera2 import Picamera2

# Initialize video stream
picam2 = Picamera2()
picam2.start()
imW = 640
imH = 480
# Function to display the original camera feed
def show_original_frame():
    try:
        while True:
            frame=picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_output = cv2.resize(frame, (int(imW), int(imH)))
            cv2.imshow('Original Frame', frame_output)
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Function to display the processed frame with red color detection
def show_processed_frame():
    try:
        while True:
            # Grab frame from video stream
            frame=picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_output = cv2.resize(frame, (int(imW), int(imH)))
            cv2.imshow('Original Frame', frame_output)
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()



# Start threads for original and processed frames
t1 = threading.Thread(target=show_original_frame)
t2 = threading.Thread(target=show_processed_frame)


try:
    t1.start()
    time.sleep(2)
    t2.start()
    time.sleep(25)

    
    t1.join()
    t2.join()
    

except KeyboardInterrupt:
    print("Interrupted! Exiting...")
finally:
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
