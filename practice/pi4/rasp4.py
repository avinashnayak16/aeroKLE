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

# Connect to the Vehicle
print("Connecting to vehicle")
vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True, timeout=60)

geofence = [
    (15.3677886, 75.1254277),  # Bottom-Left
    (15.3675572, 75.1253365),  # Bottom-Right
    (15.3674861, 75.1256423),  # Top-Right
    (15.3677305, 75.1257388)   # Top-Left
]
x_divisions = 8
y_divisions = 10
altitude = 15

# constraint of get lat lon 
R = 6371000  # Earth's radius in meters
image_width = 640  # pixels
image_height = 480  # pixels
hfov = 53  # degrees
vfov = 41  # degrees
# count number of Hotspot
object_count = 0 
# Create a lock object
lock = threading.Lock()

# detection 
hotspotFlag = True
targetFlag = True
leftToRightFlag = False
countFile = 0
interruption_flag = threading.Event()
interruption_flag.clear()
visited_hotspots = []
is_new_hotspot = False
moveToaviFlag = False
WAYPOINT_RADIUS = 2  # meters
# tarLat = vehicle.location.global_frame.lat
# tarLon = vehicle.location.global_frame.lon

class VideoStream:
        """Camera object that controls video streaming from the Picamera"""
        def __init__(self,resolution=(640,480),framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(-1)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
                
            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
        # Start the thread that reads frames from the video stream
            Thread(target=self.update,args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
        # Return the most recent frame
            return self.frame

        def stop(self):
        # Indicate that the camera and thread should be stopped
            self.stopped = True
# Initialize video stream
videostream = VideoStream(resolution=(640,480),framerate=30).start()
time.sleep(1)

def detection_thread():
    global countFile,hotspotFlag,targetFlag
    print("Detection thread started")

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"image_{timestamp}.mp4"
    
    hotspotDetectFlag = False
    targetDetectFlag = False
    

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

    # # Initialize video stream
    # videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    # time.sleep(1)
    # frame_width = int(640)
    # frame_height = int(480)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
    # # fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv
    # out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    try:
        while True:

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()

            # Grab frame from video stream
            frame1 = videostream.read()
            #frame2 = videostream.read()

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
            #count = 0
            hotspotCount = 0  # inialise the count of hotspot in the frame
            targetCount = 0
            # if hotspotDetectFlag == False and targetDetectFlag == False:
            #     #clearCoordinatesFile()  # to clear the file data 
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)) and leftToRightFlag:

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                    cx = (xmax+xmin)//20
                    cy = (ymax+ymin)//2
                    print("The coordinates:",cx,cy)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    # cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    # cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                    if object_name == "hotspot":
                        if hotspotFlag:
                            with open("coordinates.txt", "a") as f:
                                f.write(f"{cx}, {cy}, {countFile}, {object_name}\n")
                            countFile+=1
                        with open("pixel.txt", "a") as f:
                                f.write(f"{cx}, {cy}, {hotspotCount}, {object_name}\n")
                        print("The Hotspot coordinates:",cx,cy,hotspotCount,object_name)
                        hotspotCount +=1
                        hotspotDetectFlag = True
                        label = label + str(hotspotCount)
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                        #print("The X and Y", cx, " &", cy) 

                    if object_name == "objects":
                        if targetFlag:
                            with open("coordinates.txt", "a") as f:
                                f.write(f"{cx}, {cy}, {countFile}, {object_name}\n")
                            countFile+=1
                        with open("pixel.txt", "a") as f:
                                f.write(f"{cx}, {cy}, {targetCount}, {object_name}\n")
                        print("The objects coordinates:",cx,cy,targetCount,object_name)     
                        targetDetectFlag = True 
                        label = label + str(targetCount)
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    
                    print(label)
                    interruption_flag.set()

            if hotspotDetectFlag : 
                hotspotFlag = False
            if targetDetectFlag :
                targetFlag = False
            if not interruption_flag.is_set():
                hotspotFlag = True
                targetFlag = True
                hotspotDetectFlag = False
                targetDetectFlag = False

            cv2.imshow('Detect', frame)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user")
                break

    except KeyboardInterrupt:
        print("interrupted by user")
    finally:
        # Release everything when done
        videostream.stop() 
        cv2.destroyAllWindows()

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()


def send_ned_velocity(velocity_x, velocity_y,velocity_z):
    print("send_ned velocity call")
    # print(velocity_x, velocity_y,velocity_z)
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        0b10111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_y, velocity_x, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    vehicle.send_mavlink(msg)
    vehicle.flush()

def get_distance_metres(location1, location2):
    dlat = location2.lat - location1.lat
    dlong = location2.lon - location1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

# Function to set the yaw angle
def set_yaw(heading, relative=False):
    print("Set_fix the Yaw")
    if relative:
        is_relative = 1  # Yaw relative to direction of travel
    else:
        is_relative = 0  # Yaw is an absolute angle
    
    # Create the CONDITION_YAW command
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        heading,  # param 1: Yaw angle in degrees
        0,  # param 2: Yaw speed (unused)
        1,  # param 3: Direction -1 (counterclockwise), 1 (clockwise)
        is_relative,  # param 4: Relative offset or absolute angle
        0, 0, 0)  # param 5-7 (unused)
    
    # Send the command to the vehicle
    vehicle.send_mavlink(msg)

# Function to arm the drone and take off to a specified altitude
def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to a_target_altitude.
    """
    print("Basic Pre-arm checks")
    # Don't try to arm until autopilot is ready
    '''while not vehicle.is_armable:
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
    time.sleep(0.1)
    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the commands
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        # Trigger just below target alt.
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)'''

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

def getLatLon(x_pixel,y_pixel):
    
    #change later
    # x_c = image_width // 2
    # y_c = image_height // 2
    x_c = 400
    y_c = 240 

    # Angle per pixel
    hpp = hfov / image_width  # Horizontal angle per pixel
    vpp = vfov / image_height  # Vertical angle per pixel

    # Pixel displacement from center
    delta_x = x_pixel - x_c
    delta_y = y_pixel - y_c

    # Angle displacement
    theta_x = delta_x * hpp  # in degrees
    theta_y = delta_y * vpp  # in degrees

    # Convert angles from degrees to radians
    theta_x_rad = np.radians(theta_x)
    theta_y_rad = np.radians(theta_y)

    # Distance calculation
    Dx = abs(np.tan(theta_x_rad) * vehicle.location.global_relative_frame.alt)
    Dy = abs(np.tan(theta_y_rad) * vehicle.location.global_relative_frame.alt)
    # Dx = abs(np.tan(theta_x_rad) * altitude)
    # Dy = abs(np.tan(theta_y_rad) * altitude)
    # print(Dx,Dy ,"in metres")

    mid_lat = vehicle.location.global_relative_frame.lat
    mid_lon = vehicle.location.global_relative_frame.lon
    # mid_lat = 15.3678249
    # mid_lon = 75.1254934

    # Latitude change
    delta_lat = Dy / R
    target_lat = mid_lat + np.degrees(delta_lat)

    # Longitude change (adjusting for latitude)
    delta_lon = Dx / (R * np.cos(np.radians(mid_lat)))
    target_lon = mid_lon + np.degrees(delta_lon)

    if target_lat ==0 and  target_lon != 0:
        return mid_lat, target_lon

    elif target_lat !=0 and  target_lon == 0:
        return target_lat, mid_lon

    return target_lat, target_lon

def clearCoordinatesFile():
    print("Clear the file")
    with open("coordinates.txt", "w") as f:
        pass  # Opening in write mode with 'w' will clear the file content


def readingLastCoordinates():
    print("Reading lastest value")
    with open("pixel.txt", "r") as file:
            lines = file.readlines()
           
            line = lines[-1].strip()  # Remove any leading/trailing whitespace
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"Error: The line does not contain exactly 4 elements after splitting. Found {len(parts)} element(s).")

            x_pix, y_pix, hotspotNum, typeDetect = parts
            x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
            return x_pix, y_pix
        

def readingStepCoordinates(countFile):
    with open("coordinates.txt", "r") as file:
            lines = file.readlines()
            if countFile >= len(lines) or countFile < 0:
                raise IndexError("Error: Line number out of range.")

            line = lines[countFile].strip()  # Remove any leading/trailing whitespace
            #print(f"Raw line content: '{line}'")  # Debug: show the raw line content
            # Split using just a comma as the delimiter
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"Error: The line does not contain exactly 4 elements after splitting. Found {len(parts)} element(s).")

            x_pix, y_pix, hotspotNum, typeDetect = parts
            x_pix, y_pix, hotspotNum = map(int, [x_pix, y_pix, hotspotNum])
            return x_pix, y_pix, hotspotNum, typeDetect

def take_picture():
    global object_count  # Add this line to modify the global variable
    #print(object_count)
    object_count += 1
    #print(object_count)
    frame = videostream.read()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #filename = f"image_{timestamp}_lat{location.lat}_lon{location.lon}_alt{location.alt}.jpg"
    filename = f"image_{timestamp}.jpg"
    time.sleep(1)
    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")
    # time.sleep(1)
    #image_detect = True
    # clearFile()
    # processed_thread.join()

def serpentine_path_grid(grid):
    global leftToRightFlag
    print("The serpentine path started")
    model_thread = threading.Thread(target=detection_thread)   
    model_thread.start()
    time.sleep(1)
    leftToRightFlag = True
    #go_to_location(15.36754735679824,75.12549126633488,altitude)
    for i, row in enumerate(grid):
        if i % 2 == 0:
            leftToRightFlag = True
            # Move from left to right
            for point in row:
                cam = True
                print("Move left to right");
                lat, lon = point
                go_to_location(lat, lon, altitude)
                time.sleep(0.1)
        else:
            leftToRightFlag = False
            # Move from right to left
            for point in reversed(row):
                print("move right to left")
                lat, lon = point
                go_to_location(lat, lon, altitude)
                time.sleep(0.4) 
    
    # stop the thread
    model_thread.join()


def loopCheckingThread():
    global moveToaviFlag,is_new_hotspot,visited_hotspots,tarLat,tarLon
    print("loop checking thread is started")
    while True:
        prev_x, prev_y = None, None
        if interruption_flag.is_set() or hotspotFlag == False or targetFlag == False:
            print("loop")
            x_pixel,y_pixel,hotspotNum,typeDetect = readingStepCoordinates(countFile)
            prev_x, prev_y = readingLastCoordinates()
            hotspot_coordinates = getLatLon(x_pixel,y_pixel)
            tarLat,tarLon = hotspot_coordinates

            prevCount=0
            for i in range(2):
                x_pixel,y_pixel = readingLastCoordinates()
                if prev_x != x_pixel and prev_y !=y_pixel:
                    prevCount+=1
                    prev_x = x_pixel
                    prev_y = y_pixel

            if prevCount != 2:
                interruption_flag.clear()
                return
            
            # Check if the hotspot has already been visited
            is_new_hotspot = True
            for coord in visited_hotspots:
                if get_distance_metres(coord, hotspot_coordinates) <= WAYPOINT_RADIUS:
                    print("Hotspot already visited. Skipping...")
                    is_new_hotspot = False
                    break
            
            if is_new_hotspot:
                # Navigate to the hotspot
                moveToaviFlag = True  
                time.sleep(0.1)                
                # Store the hotspot coordinates
                visited_hotspots.append(hotspot_coordinates)                
                print("Hotspot coordinates stored:", hotspot_coordinates)
            
            #     set_yaw(vehicle.heading)
            #     go_toLocation(tarLat,tarLon)
            #     time.sleep(0.1)
            #     #visited_hotspots.append(tarLat,tarLon)
            #     pic_thread = threading.Thread(target=take_picture,name="pic_image" ,args=(cap, lock))
            #     pic_thread.start()
            #     time.sleep(1)
            #     pic_thread.join()  # Wait for the thread to finish before starting the next one

def go_to_location(latitude, longitude, altitude):
    global is_new_hotspot ,moveToaviFlag
    print(f"Going to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}")
    target_location = LocationGlobalRelative(latitude, longitude, altitude)
    #vehicle.simple_goto(target_location, groundspeed=3)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        distance_to_target = get_distance_metres(current_location, target_location)
        print(f"Distance to target: {distance_to_target:.2f} meters")

        if interruption_flag.is_set() and moveToaviFlag:
            print("Interruption detected. Switching to moveToavi.")
            moveToavi()  # Call moveToavi on interruption
            interruption_flag.clear()
            time.sleep(1)
            vehicle.simple_goto(target_location, groundspeed=3)
            print("resuming back to mission")
            time.sleep(0.1)
            # break
    
        if distance_to_target <= 1.0:
            print("Reached target location.")
            break
        time.sleep(1)

def moveToavi():
    print("Move Towards Hotspot")
    global moveToaviFlag
    
    # Store the hotspot coordinates
    # visited_hotspots.append(hotspot_coordinates)                
    # print("Hotspot coordinates stored:", hotspot_coordinates)
    set_yaw(vehicle.heading)
    go_toLocation(tarLat,tarLon,altitude)
    time.sleep(0.1)
    #visited_hotspots.append(tarLat,tarLon)
    # pic_thread = threading.Thread(target=take_picture,name="pic_image" ,args=(cap, lock))
    # pic_thread.start()
    take_picture()
    time.sleep(1)
    # pic_thread.join()  # Wait for the thread to finish before starting the next one

    moveToaviFlag = False
    
        # if typeDetect == "target":
        #     moveToavipart()
        
def main():
    print("Mission Begins")
    start_time = time.time()
    #print(f"Home Location: {vehicle.location.global_relative_frame.lat}, {vehicle.location.global_relative_frame.lon}")
    
    loop_thread = threading.Thread(target=loopCheckingThread)   
    loop_thread.start()
    time.sleep(1)
    
    grid = generate_grid(geofence, x_divisions,y_divisions)
    time.sleep(1)
    arm_and_takeoff(altitude)
    time.sleep(2)

    serpentine_path_grid(grid)
    time.sleep(5)
    # go_to_location(,altitude)

    print("Mission completed, landing...")
    vehicle.mode = VehicleMode("RTL")

    end_time = time.time()
    print("Total time taken =",(end_time-start_time))
    print("total object detected are = ", object_count)
    for coord in visited_hotspots:
        print("Hotspot already visited. ",coord)

    time.sleep(2)
    loop_thread.join()
    #end_time = time.time()
    #print("Total time taken =",(end_time-start_time))
    #print("total object detected are = ", object_count)

    # Release the webcam and close all windows
    videostream.stop() 
    cv2.destroyAllWindows()
    vehicle.close()

if __name__ == "__main__":
    main()

