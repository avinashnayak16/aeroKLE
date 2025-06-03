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
#from tflite_runtime.interpreter import Interpreter
import argparse
from threading import Thread
import importlib.util
from picamera2 import Picamera2
from PIL import Image

# Connect to the Vehicle
print("Connecting to vehicle")
vehicle = connect('/dev/ttyACM0', baud=921600, wait_ready=True, timeout=60)

geofence = [
    (15.3693223, 75.1261747),  # Bottom-Left
    (15.3691413, 75.1262002),  # Bottom-Right
    (15.3691387, 75.1263973),  # Top-Right
    (15.3693534, 75.1263571)   # Top-Left
]
x_divisions = 10
y_divisions = 10
altitude = 15

# detection 
interruption_flag = threading.Event()
stop_thread = False
yellowThread = False

tarLocation = LocationGlobalRelative(vehicle.location.global_relative_frame.lat,vehicle.location.global_relative_frame.lon,altitude)
targetDetectFlag = True

# Initialize video stream
picam2 = Picamera2()
picam2.start()

def detectionClassify():
    global targetDetectFlag
    print("Detection thread started")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.mp4"
    frameCount = 0
    targetCount = 0
    start_time = time.time()


    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=False,default='targets')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect_quant.tflite')
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
    out = cv2.VideoWriter(filename, fourcc, 18.0, (frame_width, frame_height))

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    try:
        while not stop_thread:
            # Grab frame from video stream
            frame1=picam2.capture_array()

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

                    if object_name == "target" :
                       with open("YelloWcoordinates.txt", "a") as f:
                               f.write(f"{cx}, {cy} \n")
                            #    print("target The coordinates:",cx,cy,object_name)

                    if (object_name == "target" or object_name == "objects") and targetDetectFlag:
                        with open("pixel.txt", "a") as f:
                                f.write(f"{cx}, {cy} \n")
                        print("The coordinates:",cx,cy,object_name)     
                        targetCount +=1
                    elif object_name == "hotspot":
                        # if abs(cx-320)==35 and abs(cy-240)==35:
                        #     cv2.imwrite(f'{frameCount}.jpg',frame)
                        #     frameCount +=1
                        frameCount +=1
                        cv2.imwrite(f'{frameCount}.jpg',frame)

            if targetDetectFlag and targetCount >= 4:
                interruption_flag.set()
                time.sleep(1)
            current_time = time.time()            
            if current_time - start_time >= 2 and targetDetectFlag:
                targetCount = 0 
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

def take_picture():
    global targetDetectFlag,tarLocation  # Add this line to modify the global variable
    print("taking picture")
    for i in range(3):
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
    targetDetectFlag = False

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
        print(f"Distance to waypoint: {distance_to_target:.2f} meters")

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
            print("Reached waypoint.")
            break
        time.sleep(1)
        
def go_to_target(location, altitude):
    print(f"Going to Location: {location}, Altitude: {altitude}")
    vehicle.simple_goto(location, groundspeed=3)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        distance_to_target = get_distance_metres(current_location, location)
        print(f"Distance to target: {distance_to_target:.2f} meters")
    
        if distance_to_target <= 1.0:
            print("Reached target.")
            break
        time.sleep(1)

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

    for i in range(12):
    # while True:
        x_coord,y_coord = readingCoordinates("pixel.txt")
        time.sleep(0.5)
        print("points=",x_coord,y_coord)
        send_velocity_based_on_position(x_coord, y_coord, 0.28)
        time.sleep(1)

        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        prev_x, prev_y = x_coord,y_coord 

        # if((abs(x_coord-h_x)<=50) and (abs(y_coord-h_y)<=50)):
        #     break

    time.sleep(1)
    take_picture()
    time.sleep(1)

def drop_payload(PWM):
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target_system, target_component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,  # command
        0,  # confirmation
        9,  # servo number
        PWM,  # servo position between 1000 and 2000
        0, 0, 0, 0, 0)  # param 3 ~ 7 not used
    print("dropping payload...")
    # send command to vehicle
    vehicle.send_mavlink(msg)
    print("payload dropped...")

def yellowDetection():
        try:
            while not yellowThread:              

                frame1=picam2.capture_array()
                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert frame to HSV color space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Define lower and upper bounds for yellow color in HSV
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])

                # Threshold the HSV frame to create a mask for yellow colors
                red_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Define lower and upper bounds for blue color in HSV
                # lower_blue = np.array([100, 150, 150])  # Adjust values as needed
                # upper_blue = np.array([140, 255, 255])

                # # Threshold the HSV frame to create a mask for blue colors
                # red_mask = cv2.inRange(hsv, lower_blue, upper_blue)

                # Define lower and upper bounds for red color in HSV
                # lower_red1 = np.array([0, 100, 100])
                # upper_red1 = np.array([10, 255, 255])
                
                # lower_red2 = np.array([160, 100, 100])
                # upper_red2 = np.array([180, 255, 255])

                # # Threshold the HSV frame to create a mask for red colors
                # red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                # red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                # red_mask = red_mask1 + red_mask2

                # Find contours in the red mask
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Initialize variables to store information about the largest contour
                largest_contour = None
                largest_contour_area = 0

                # Find the largest contour
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > largest_contour_area:
                        largest_contour = contour
                        largest_contour_area = area

                # Draw a circle at the centroid of the largest contour
                if largest_contour is not None and len(largest_contour) >= 5:
                    # Calculate centroid of the largest contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                       
                        print("color detection :",cx , " &", cy)
                        with open("YelloWcoordinates1.txt", "a") as f:
                            f.write(f"{cx}, {cy}\n")
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)        

        except KeyboardInterrupt:
            print("interrupted by user")
        finally:
            cv2.destroyAllWindows()

def moveTotarget():

    def moveTotargetPoint():
        print("Move Towards target")
        x_coord = 320
        y_coord = 240
        h_x = 320
        h_y = 240

        def send_velocity_based_on_position(x_coord,y_coord,g_speed,z_speed):
            if x_coord == h_x and y_coord == h_y:
                send_ned_velocity(0, 0, 0.3)
            elif x_coord > h_x and y_coord > h_y:
                send_ned_velocity(-g_speed, g_speed, z_speed)
            elif x_coord < h_x and y_coord < h_y:
                send_ned_velocity(g_speed, -g_speed, z_speed)
            elif x_coord < h_x and y_coord > h_y:
                send_ned_velocity(-g_speed, -g_speed, z_speed)
            elif x_coord > h_x and y_coord < h_y:
                send_ned_velocity(g_speed, g_speed, z_speed)
            elif x_coord == h_x and y_coord != h_y:
                if y_coord > h_y:
                    send_ned_velocity(0, -g_speed, z_speed)
                elif y_coord < h_y:
                    send_ned_velocity(0, g_speed, z_speed)
            elif y_coord == h_y and x_coord != h_x:
                if x_coord > h_x:
                    send_ned_velocity(g_speed, 0, z_speed)
                elif x_coord < h_x:
                    send_ned_velocity(-g_speed, 0, z_speed)


        prev_x, prev_y = 320, 240

        # for i in range(5):
        while True:
            x_coord,y_coord = readingCoordinates("YelloWcoordinates.txt")
            time.sleep(0.6)
            print("points target =",x_coord,y_coord)

            if prev_x is not None and prev_y is not None:
                if x_coord == prev_x and y_coord == prev_y:
                    send_ned_velocity(0, 0, 0.3)
                    time.sleep(1)
                else:
                    send_velocity_based_on_position(x_coord, y_coord, 0.3, 0.3)

            if vehicle.location.global_relative_frame.alt <= 7:
                break
 
            
            print("Altitude: ", vehicle.location.global_relative_frame.alt)
            prev_x, prev_y = x_coord,y_coord 

            # if((abs(x_coord-h_x)<=50) and (abs(y_coord-h_y)<=50)) and vehicle.location.global_relative_frame.alt <=5:
            #    break
            
        time.sleep(1)
        take_picture()
        while True:
            x_coord,y_coord = readingCoordinates("YelloWcoordinates1.txt")
            time.sleep(0.6)
            print("points color=",x_coord,y_coord)

            if prev_x is not None and prev_y is not None:
                if x_coord == prev_x and y_coord == prev_y:
                    send_ned_velocity(0, 0, 0.3)
                    time.sleep(1)
                else:
                    send_velocity_based_on_position(x_coord, y_coord, 0.1, 0.2)

            if vehicle.location.global_relative_frame.alt <= 5:
                break
 
            
            print("Altitude: ", vehicle.location.global_relative_frame.alt)
            prev_x, prev_y = x_coord,y_coord 
    
    
    # goto target
    global yellowThread

    yellowdetection_thread = threading.Thread(target=yellowDetection , name="yellow")   
    yellowdetection_thread.start()
    time.sleep(1)

    go_to_target(tarLocation, altitude)
    time.sleep(1)
    print("Clearing points")
    clearFile("YelloWcoordinates.txt",320,240)
    clearFile("YelloWcoordinates1.txt",320,240)
    time.sleep(1)    
    
    moveTotargetPoint()
    time.sleep(2)

    drop_payload(1700)
    time.sleep(3)

    take_picture()
    time.sleep(1)
    yellowThread = True
    yellowdetection_thread.join()


def main():
    global stop_thread , yellowThread
    start_time = time.time()

    print("Mission Begins")
    print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")

    try:
        detection_thread = threading.Thread(target=detectionClassify, name="Resume the detection")   
        detection_thread.start()
        
        grid = generate_grid(geofence, x_divisions,y_divisions)
        time.sleep(2)
        arm_and_takeoff(altitude)
        time.sleep(2)

        serpentine_path_grid(grid)
        # go_to_target(15.367641317816378,75.12555770181555,altitude)
        time.sleep(4)

        # mission for target
        print("target location",tarLocation)
        print("move towards Target Location")
        moveTotarget()
        time.sleep(2)

        print("Mission completed, Returning to Launch...")
        vehicle.mode = VehicleMode("RTL")
        time.sleep(2)
        
        end_time = time.time()
        print("Total time taken =",(end_time-start_time))
        # tarLat,tarLon,alti = tarLocation
        print("target location",tarLocation)

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
