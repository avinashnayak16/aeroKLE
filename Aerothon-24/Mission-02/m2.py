import cv2
import numpy as np
import threading
from datetime import datetime
import time
import math
import os
import sys
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
from tflite_runtime.interpreter import Interpreter
import argparse
from threading import Thread
import importlib.util
from picamera2 import Picamera2
from PIL import Image
import glob

# Connect to the Vehicle
print("Connecting to vehicle")
vehicle = connect('/dev/ttyACM0', baud=921600, wait_ready=True, timeout=60)

geofence = [
    (15.3678572, 75.1254029),  # Bottom-Left
    (15.3674679, 75.1253647),  # Bottom-Right
    (15.3674059, 75.1259306),  # Top-Right
    (15.3677977, 75.1259641)   # Top-Left
]
x_divisions = 11
y_divisions = 15
altitude = 15

# detection 
interruption_flag = threading.Event()
stop_thread = False
yellowThread = False

tarLocation = LocationGlobalRelative(vehicle.location.global_relative_frame.lat,vehicle.location.global_relative_frame.lon,altitude)
objectDetectFlag = True

# Initialize video stream
picam2 = Picamera2()
picam2.start()

def detectionClassify():
    global objectDetectFlag
    print("Detection thread started")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.mp4"
    frameCount = 0
    objectCount = 0
    start_time = time.time()


    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=False,default='objects')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.8)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='640x480')
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

    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
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

    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize video stream
    frame_width = int(640)
    frame_height = int(480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
    # fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv
    out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    try:
        while not stop_thread:
            # Grab frame from video stream
            frame1=picam2.capture_array()
            # ret ,frame1 = cap.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (width, height))
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
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                    cx = (xmax+xmin)//2
                    cy = (ymax+ymin)//2
                    #print("The coordinates:",cx,cy)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    if (object_name == "objects") and objectDetectFlag:
                        with open("pixel.txt", "a") as f:
                                f.write(f"{cx}, {cy} \n")
                        print("The coordinates:",cx,cy,object_name)     
                        objectCount +=1
                    elif object_name == "hotspot":
                        # if abs(cx-320)==35 and abs(cy-240)==35:
                        #     cv2.imwrite(f'{frameCount}.jpg',frame)
                        #     frameCount +=1
                        frameCount +=1
                        cv2.imwrite(f'{frameCount}.jpg',frame)

            if objectDetectFlag and objectCount >= 3:
                interruption_flag.set()
                time.sleep(1)
            current_time = time.time()            
            if current_time - start_time >= 2 and objectDetectFlag:
                objectCount = 0 
                start_time = current_time

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Detect', frame)
            frame_output = cv2.resize(frame, (int(imW), int(imH)))
            out.write(frame_output)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user")
                break

    except KeyboardInterrupt:
        print("interrupted by user")
    finally:
        # Release everything when done
        #picam2.stop() 
        #picam2.close()
        out.release()
        cv2.destroyAllWindows()

    # Clean up
    out.release()
    cv2.destroyAllWindows()
    #picam2.stop() 
    #picam2.close()

def detectShapes():
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.mp4"
    print(filename)

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=False ,default='shapes')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
    parser.add_argument('--image', help='Name of the single image to perform detection on.', default=None)
    parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder', action='store_true')
    parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)', action='store_false')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

    args = parser.parse_args()

    # Parse user inputs
    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    use_TPU = args.edgetpu
    save_results = args.save_results
    show_results = args.noshow_results
    IM_NAME = args.image

    # Import TensorFlow Lite Interpreter or regular TensorFlow based on the package
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign the filename for Edge TPU model
    if use_TPU:
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Get the current working directory
    CWD_PATH = os.getcwd()

    # Create results directory if user wants to save results
    if save_results:
        RESULTS_PATH = os.path.join(RESULTS_DIR) # type: ignore
        if not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

    # Path to .tflite file (the model used for object detection)
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # If the first label is '???', remove it
    if labels[0] == '???':
        del(labels[0])

    # Load the TensorFlow Lite model (with or without Edge TPU support)
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
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

    # Determine if the model is based on TF2 or TF1
    outname = output_details[0]['name']

    if 'StatefulPartitionedCall' in outname:  # TF2 Model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:  # TF1 Model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Function to detect shapes (squares and circles) within a specified region of interest (ROI)
    def detect_shapes_in_bbox(image, bbox):
        # Extract the region of interest (ROI) using the bounding box coordinates
        xmin, ymin, xmax, ymax = bbox
        roi = image[ymin:ymax, xmin:xmax]

        # Convert the ROI to grayscale and apply thresholding to detect shapes
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        circle_detected = False

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            if len(approx) > 8:  # More than 8 vertices indicates a circle
                circle_detected = True

        return circle_detected

    for k in range(3):

        image = picam2.capture_array() 
        # ret ,image = cap.read()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize if using a floating model
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Run the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        total_square_count = 0
        total_circle_count = 0
        total_triangle_count = 0

        # Loop over the results and draw boxes for objects above the confidence threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                # Detect shapes within the bounding box
                circle_detected = detect_shapes_in_bbox(image, (xmin, ymin, xmax, ymax))
                color = (0, 0, 255) if circle_detected else (0, 255, 0)  # Red for circles, green for others

                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label
                object_name = labels[int(classes[i])]  # Get object name from labels
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # e.g., 'person: 72%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                            (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                if object_name == "square":
                    total_square_count+=1
                elif object_name == "circle":
                    total_circle_count+=1
                elif object_name == "triangle":
                    total_triangle_count+=1
                # cv2.imwrite(f"image_{timestamp}.jpg", image)

        # Display the total count of squares and circles on the image
        text = f"Squares: {total_square_count}, Circles: {total_circle_count},Triangles:{total_triangle_count}"
        cv2.putText(image, text, (10, imH - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Show the image with detections and counts
        cv2.imwrite(f"image_{timestamp}{k}.jpg", image)
        print("dete")
        time.sleep(0.2)
        # print(k)


    # Clean up
    take_picture()
    time.sleep(1)
    cv2.destroyAllWindows()


def take_picture():
    global objectDetectFlag,tarLocation  # Add this line to modify the global variable
    print("taking picture")
    for i in range(2):
        frame=picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #filename = f"image_{timestamp}_lat{location.lat}_lon{location.lon}_alt{location.alt}.jpg"
        filename = f"image_{timestamp}{i}.jpg"
        # time.sleep()
        cv2.imwrite(filename, frame)
        time.sleep(0.3)
        print(f"Saved image: {filename}")
    # time.sleep(0.1)
    tarLocation = LocationGlobalRelative(vehicle.location.global_relative_frame.lat,vehicle.location.global_relative_frame.lon,altitude)
    time.sleep(0.3)
    objectDetectFlag = False

def clearFile(filename,num1, num2):
    with open(filename, "w") as f:
        f.write(f"{num1}, {num2}\n")  # Add f-string to format the string properly

def readingCoordinates(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        if not lines:
            return None, None  # Return None if the file is empty
        last_line = lines[-1].strip()
        x, y = map(int, last_line.split(", "))
        return x, y

def generate_grid(geofence, x_divisions, y_divisions):
    print("Grid is divided")
    # print(x_divisions, y_divisions)
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
    print("Serpentine path started")
    for i, row in enumerate(grid):
        if i % 2 == 0:
            # Move from left to right
            print("Move from left to right")
            for point in row:
                lat, lon = point
                go_to_location(lat, lon, altitude)
                time.sleep(0.1)
        else:
            # Move from right to left
            print("Move from right to left")
            for point in reversed(row):
                lat, lon = point
                go_to_location(lat, lon, altitude)
                time.sleep(0.1)

def arm_and_takeoff(a_object_altitude):
    """
    Arms vehicle and fly to a_object_altitude.
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
    time.sleep(1)
    vehicle.simple_takeoff(a_object_altitude)  # Take off to object altitude

    # Wait until the vehicle reaches a safe height before processing the commands
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        # Trigger just below object alt.
        if vehicle.location.global_relative_frame.alt >= a_object_altitude * 0.95:
            print("Reached object altitude")
            break
        time.sleep(1)

def go_to_location(latitude, longitude, altitude):
    print(f"Going to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}")
    object_location = LocationGlobalRelative(latitude, longitude, altitude)
    vehicle.simple_goto(object_location, groundspeed=3.2)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        distance_to_object = get_distance_metres(current_location, object_location)
        print(f"Distance to waypoint: {distance_to_object:.2f} meters")

        if interruption_flag.is_set():
            print("Interruption detected. Switching to moveToavi.")
            moveToavi()  # Call moveToavi on interruption
            interruption_flag.clear()
            time.sleep(1)
            vehicle.simple_goto(object_location, groundspeed=3)
            print("resuming back to mission")
            time.sleep(1)
            # break
    
        if distance_to_object <= 1.0:
            print("Reached waypoint.")
            break
        time.sleep(1)
        
def go_to_object(location, altitude):
    print(f"Going to Location: {location}, Altitude: {altitude}")
    vehicle.simple_goto(location, groundspeed=3)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        distance_to_object = get_distance_metres(current_location, location)
        print(f"Distance to object: {distance_to_object:.2f} meters")
    
        if distance_to_object <= 1.0:
            print("Reached object.")
            break
        time.sleep(1)

def get_distance_metres(location1, location2):
    dlat = location2.lat - location1.lat
    dlong = location2.lon - location1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

def send_ned_velocity(velocity_x, velocity_y,velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # object system, object component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b10111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    vehicle.send_mavlink(msg)
    vehicle.flush()


def moveToavi():

    print("Move Towards location")
    clearFile("pixel.txt",320, 240)
    x_coord = 320
    y_coord = 240
    h_x = 320
    h_y = 240

    def send_velocity_based_on_position(x_coord,y_coord,g_speed):
            if x_coord == h_x and y_coord == h_y:
                send_ned_velocity(0, 0, 0)
            elif x_coord > h_x and y_coord > h_y:
                send_ned_velocity(-g_speed, g_speed, 0)
            elif x_coord < h_x and y_coord < h_y:
                send_ned_velocity(g_speed, -g_speed, 0)
            elif x_coord < h_x and y_coord > h_y:
                send_ned_velocity(-g_speed, -g_speed, 0)
            elif x_coord > h_x and y_coord < h_y:
                send_ned_velocity(g_speed, g_speed, 0)
            elif x_coord == h_x and y_coord != h_y:
                if y_coord > h_y:
                    send_ned_velocity(0, -g_speed, 0)
                elif y_coord < h_y:
                    send_ned_velocity(0, g_speed, 0)
            elif y_coord == h_y and x_coord != h_x:
                if x_coord > h_x:
                    send_ned_velocity(g_speed, 0, 0)
                elif x_coord < h_x:
                    send_ned_velocity(-g_speed, 0, 0)

    prev_x, prev_y = None, None

    for i in range(10):
    # while True:
        x_coord,y_coord = readingCoordinates("pixel.txt")
        time.sleep(0.3)
        print("points=",x_coord,y_coord)
        send_velocity_based_on_position(x_coord, y_coord, 0.3)
        time.sleep(1)

        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        prev_x, prev_y = x_coord,y_coord 

        # if((abs(x_coord-h_x)<=50) and (abs(y_coord-h_y)<=50)):
        #     break

    time.sleep(1)
    detectShapes()
    time.sleep(1)

    # take_picture()
    # time.sleep(1)


def main():
    global stop_thread , yellowThread
    start_time = time.time()

    print("Mission Begins")
    print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")

    try:
               
        grid = generate_grid(geofence, x_divisions,y_divisions)
        time.sleep(2)
        arm_and_takeoff(altitude)
        time.sleep(2)
        detection_thread = threading.Thread(target=detectionClassify, name="Resume the detection")   
        detection_thread.start()

        serpentine_path_grid(grid)
        #go_to_object(LocationGlobalRelative(15.3673324, 75.1253645,altitude),altitude)
        time.sleep(2)
        #moveToavi()
        time.sleep(2)

        print("Mission completed, Returning to Launch...")
        vehicle.mode = VehicleMode("RTL")
        time.sleep(2)
        
        end_time = time.time()
        print("Total time taken =",(end_time-start_time))
        # tarLat,tarLon,alti = tarLocation
        print("object location",tarLocation)

        # Release the webcam and close all windows
        stop_thread = True    
        yellowThread = True   
        detection_thread.join()
        cv2.destroyAllWindows()
        picam2.stop()
        picam2.close()
        vehicle.close()
    
    except KeyboardInterrupt:
        print("Recording interrupted by user")
    finally:
        stop_thread = True 
        yellowThread = True   
        detection_thread.join()

        cv2.destroyAllWindows()
        picam2.stop()
        picam2.close()
        vehicle.close()


if __name__ == "__main__":
    main()
