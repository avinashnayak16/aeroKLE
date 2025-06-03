import cv2
import numpy as np
import threading
from datetime import datetime
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import argparse
from pymavlink import mavutil  # Required for sending mavlink commands
import os
import sys
import importlib.util

# parser = argparse.ArgumentParser()
# parser.add_argument('--connect', default='127.0.0.1:14550')
# args = parser.parse_args()

# Connect to the Vehicle
# print('Connecting to vehicle on: %s' % args.connect)
# vehicle = connect(args.connect, baud=921600, wait_ready=True, timeout=60)
vehicle = connect('/dev/serial0', baud=921600, wait_ready=True, timeout=60)


# Parameters
alt = 10  # Desired altitude in meters
groundspeed = 3  # Ground speed in m/s
# gridSize = 10  # Size of each grid cell in meters (10m x 10m)
x_divisions = 9
y_divisions = 13
# Define Geofence Corners (latitude, longitude)
geofence = [
    (15.3677944678254, 75.1253820955753),  # Bottom-Left
    (15.3675248, 75.1253325),  # Bottom-Right
    (15.3675067, 75.1255846),  # Top-Right
    (15.3677886, 75.1256061)   # Top-Left
]

# Initialize global variables
x, y = 0, 0
h_x = 400
h_y = 240
object_count = 0
image_detect = False

# Initialize webcam
cap = cv2.VideoCapture(-1)

# Create a lock object
lock = threading.Lock()
interruption_flag = threading.Event()
stop_event = threading.Event()

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to display the original camera feed
def show_original_frame(cap, lock):    
    while True:
        with lock:
            ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Original Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()


def detectObject(cap, lock, stop_event):
    while not stop_event.is_set():
        # Define and parse input arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            required=False,default='custom_model')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                            default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                            default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                            default=0.8)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                            default='640x320')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                            action='store_true')

        args = parser.parse_args()

        MODEL_NAME = args.modeldir
        GRAPH_NAME = args.graph
        LABELMAP_NAME = args.labels
        min_conf_threshold = float(args.threshold)
        resW, resH = args.resolution.split('x')
        imW, imH = int(resW), int(resH)
        use_TPU = args.edgetpu

        # Import TensorFlow libraries
        # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        # If using Coral Edge TPU, import the load_delegate library
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
            if use_TPU:
                from tflite_runtime.interpreter import load_delegate
        else:
            from tensorflow.lite.python.interpreter import Interpreter
            if use_TPU:
                from tensorflow.lite.python.interpreter import load_delegate

        # If using Edge TPU, assign filename for Edge TPU model
        if use_TPU:
            # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
            if (GRAPH_NAME == 'detect.tflite'):
                GRAPH_NAME = 'edgetpu.tflite'       

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

        # Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

        # Load the Tensorflow Lite model.
        # If using Edge TPU, use special load_delegate argument
        if use_TPU:
            interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            print(PATH_TO_CKPT)
        else:
            interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else: # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2

        # Initialize frame rate calculation
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()

        # Initialize video stream
        # videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        time.sleep(1)

        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            # frame1 = videostream.read()
            with lock:
                ret, frame1 = cap.read()
            if not ret:
                break

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    cx = int((xmax+xmin)//2)
                    cy = int((ymax+ymin)//2)
                    with open("coordinates.txt", "a") as f:
                        f.write(f"{cx}, {cy}\n")
                    print("The points are :Xcoordinate=",cx," Ycoordinate:",cy)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    interruption_flag.set()
                    time.sleep(1)

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"image_{timestamp}.jpg"
            # cv2.imwrite(filename, frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()
        # videostream.stop()

# Function to display the processed frame with red color detection
def show_processed_frame(cap, lock, stop_event):
    while not stop_event.is_set():
        with lock:
            ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key=cv2.contourArea, default=None)

        if largest_contour is not None:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print("The X and Y", cx, " &", cy)
                with open("coordinates.txt", "a") as f:
                    f.write(f"{cx}, {cy}\n")
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                a, b, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (a, b), (a + w, b + h), (255, 0, 0), 2)
            
            interruption_flag.set()
            time.sleep(1)
            # interruption_flag.clear()

        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()

def take_picture(cap, lock):
    global object_count ,image_detect  # Add this line to modify the global variable
    object_count += 1
    with lock:
        ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #filename = f"image_{timestamp}_lat{location.lat}_lon{location.lon}_alt{location.alt}.jpg"
        filename = f"image_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved image: {filename}")
        time.sleep(1)
        image_detect = True
    # clearFile()
    # processed_thread.join()

def restartcam():
    global image_detect

    # Start the initial thread
    processed_thread = threading.Thread(target=detectObject, args=(cap, lock, stop_event))
    time.sleep(1)
    processed_thread.start()

    while True:
        if not processed_thread.is_alive():
            print("Thread is not alive, restarting...")
            # Clear the stop event before restarting the thread
            stop_event.clear()
            time.sleep(5)

            # Reinitialize and restart the thread
            processed_thread = threading.Thread(target=detectObject, args=(cap, lock, stop_event))
            processed_thread.start()
            print("Thread restarted.")

        if image_detect:
            print("Image detected, waiting for the thread to finish...")
            image_detect = False
            stop_event.set()  # Signal the thread to stop
            processed_thread.join()  # Wait for the thread to finish
            time.sleep(1)
            print("Thread killed after image detection.")        

def clearFile(num1, num2):
    with open("coordinates.txt", "a") as f:
        f.write(f"{num1}, {num2}\n")  # Add f-string to format the string properly

def readingCoordinates():
    with open("coordinates.txt", "r") as file:
        lines = file.readlines()
        if not lines:
            return None, None  # Return None if the file is empty
        last_line = lines[-1].strip()
        x, y = map(int, last_line.split(", "))
        return x, y


def generate_grid(geofence, geofence, x_divisions, y_divisions):
    print("Grid is divided")
    bottom_left, bottom_right, top_right, top_left = geofence
    lat_start, lon_start = bottom_left
    lat_end, lon_end = top_right

    # Calculate the number of grid cells needed
    lat_steps = int(get_distance_metres(LocationGlobalRelative(lat_start, lon_start), LocationGlobalRelative(lat_end, lon_start)) / y_divisions)
    lon_steps = int(get_distance_metres(LocationGlobalRelative(lat_start, lon_start), LocationGlobalRelative(lat_start, lon_end)) / x_divisions)

    grid = []
    for i in range(lat_steps + 1):
        row = []
        for j in range(lon_steps + 1):
            lat = lat_start + (lat_end - lat_start) * (i / lat_steps)
            lon = lon_start + (lon_end - lon_start) * (j / lon_steps)
            row.append((lat, lon))
        grid.append(row)
    return grid

def serpentine_path_grid(grid):
    for i, row in enumerate(grid):
        if i % 2 == 0:
            # Move from left to right
            for point in row:
                print("Move left to right");
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(1)
        else:
            # Move from right to left
            for point in reversed(row):
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(1)


def arm_and_takeoff(a_target_altitude):
    """
    Arms vehicle and fly to a_target_altitude.
    """
    print("Basic Pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(a_target_altitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the commands
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        # Trigger just below target alt.
        if vehicle.location.global_relative_frame.alt >= a_target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def go_to_location(latitude, longitude, altitude):
    print(f"Going to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}")
    target_location = LocationGlobalRelative(latitude, longitude, altitude)
    vehicle.simple_goto(target_location, groundspeed=3)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        distance_to_target = get_distance_metres(current_location, target_location)
        print(f"Distance to target: {distance_to_target:.2f} meters")

        if interruption_flag.is_set():
            print("Interruption detected. Switching to moveToavi.")
            moveToavi()  # Call moveToavi on interruption
            interruption_flag.clear()
            time.sleep(1)
            vehicle.simple_goto(target_location, groundspeed=3)
            print("resuming back to mission")
            time.sleep(1)
            # break
    
        if distance_to_target <= 1.0:
            print("Reached target location.")
            break
        time.sleep(0.5)
    # return 

def get_distance_metres(location1, location2):
    dlat = location2.lat - location1.lat
    dlong = location2.lon - location1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

def send_ned_velocity(velocity_x, velocity_y,velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b10111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_y, velocity_x, 0,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    vehicle.send_mavlink(msg)
    vehicle.flush()
    
def moveToavi():

    print("Move Towards Hotspot")

    def send_velocity_based_on_position(x, y,g_speed):
        if x == h_x and y == h_y:
            send_ned_velocity(0, 0, 0)
        elif x > h_x and y > h_y:
            send_ned_velocity(g_speed, -g_speed, 0)
        elif x < h_x and y < h_y:
            send_ned_velocity(-g_speed, g_speed, 0)
        elif x < h_x and y > h_y:
            send_ned_velocity(-g_speed, -g_speed, 0)
        elif x > h_x and y < h_y:
            send_ned_velocity(g_speed, g_speed, 0)
        elif x == h_x and y != h_y:
            if y > h_y:
                send_ned_velocity(0, -g_speed, 0)
            elif y < h_y:
                send_ned_velocity(0, g_speed, 0)
        elif y == h_y and x != h_x:
            if x > h_x:
                send_ned_velocity(g_speed, 0, 0)
            elif x < h_x:
                send_ned_velocity(-g_speed, 0, 0)

    prev_x, prev_y = None, None

    # for i in range(10):
    while True:
        x,y = readingCoordinates()
        time.sleep(0.1)
        print("points=",x, y)
        send_velocity_based_on_position(x, y, 0.25)
        time.sleep(1)
        # if prev_x is not None and prev_y is not None and (x != prev_x and y != prev_y):
        #     send_velocity_based_on_position(x, y, 0.25)
        #     time.sleep(1)

        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        prev_x, prev_y = x, y

        if((abs(x-h_x)<=20) and (abs(y-h_y)<=20)):
            break

    pic_thread = threading.Thread(target=take_picture,name="pic_image" ,args=(cap, lock))
    pic_thread.start()
    time.sleep(1)
    pic_thread.join()  # Wait for the thread to finish before starting the next one
    

def main():
    start_time = time.time()

    print("Mission Begins")
    print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")
    clearFile(400, 240)
    
    # Start the threads
    # original_thread = threading.Thread(target=show_original_frame, args=(cap, lock))
    # original_thread.start()
    # time.sleep(1)    
    resume_thread = threading.Thread(target=restartcam, name="Resume the detection")   
    resume_thread.start()
    
    grid = generate_grid(geofence, x_divisions,y_divisions)
    time.sleep(2)
    arm_and_takeoff(alt)
    time.sleep(2)

    serpentine_path_grid(grid)
    time.sleep(5)

    print("Mission completed, landing...")
    vehicle.mode = VehicleMode("RTL")

    # original_thread.join()
    # processed_thread.join()
    resume_thread.join()
    end_time = time.time()
    print("Total time taken =",(end_time-start_time))
    print("total object detected are = ", object_count)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    vehicle.close()


if __name__ == "__main__":
    main()

