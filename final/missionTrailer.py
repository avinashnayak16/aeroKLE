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


# Connect to the Vehicle
vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True, timeout=60)

tarLat = vehicle.global_relative_frame.lat
tarLon = vehicle.global_relative_frame.lon
moveToaviFlag = False
countFile = 0

# Parameters
altitude = 15 # Desired altitude in meters
groundSpeed = 3  # Ground speed in m/s
x_divisions = 12
y_divisions = 6

# Constants
R = 6371000  # Earth's radius in meters
image_width = 640  # pixels
image_height = 480  # pixels
  # meters
hfov = 53  # degrees
vfov = 41  # degrees
# mid_lat = 15.3677175  # Midpoint latitude
# mid_lon = 75.1254693  # Midpoint longitude

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


# List to store visited hotspots , Define waypoint radius
visited_hotspots = []
WAYPOINT_RADIUS = 1.5  # meters

def loopCheckingThread():
    while True:
        
        if detected:
            # Check if the hotspot has already been visited
            is_new_hotspot = True
            for coord in visited_hotspots:
                if get_distance_metres(coord, hotspot_coordinates) <= WAYPOINT_RADIUS:
                    print("Hotspot already visited. Skipping...")
                    is_new_hotspot = False
                    break
            
            if is_new_hotspot:
                # Naviga+++te to the hotspot
                moveToaviFlag = True
                
                # Store the hotspot coordinates
                visited_hotspots.append(hotspot_coordinates)                
                print("Hotspot coordinates stored:", hotspot_coordinates)
        
        # Add some delay before next detection
        time.sleep(1)

def loopCheckingThread_1():
    global countFile,tarLat,tarLon,moveToaviFlag,is_new_hotspot,visited_hotspots
    while True:
        prev_x, prev_y = None, None
        
        for i in countFile:
            x_pixel,y_pixel,hotspotNum,typeDetect = readingStepCoordinates(i)
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
                return
            if interruption_flag.is_set():
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
                    

def detection_thread(cap,lock,stop_event):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.mp4"

    MODEL_NAME = 'targets'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = float(0.8)
    count = 0
    hotspotFlag = True
    targetFlag = True
    hotspotDetectFlag = False
    targetDetectFlag = False

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

    # Open video file
    videopath="/home/avinash/Music/model/manual/image_20240817_161049.mp4"
    #video = cv2.VideoCapture(videopath)
    imW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    while():

        # Acquire frame and resize to expected shape [1xHxWx3]
        with lock:
            ret, frame = cap.read()
        if not ret:
            print('Reached the end of the video!')
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("he.jpg",frame_rgb)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        # cv2.imwrite("hell.jpg",frame_rgb)
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
        #     clearCoordinatesFile()  # to clear the file data 
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)) and leftToRightFlag:

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                cx = (xmax+xmin)//2
                cy = (ymax+ymin)//2
                print("The coordinates:",cx,cy)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                if object_name == "hotspot":
                    if hotspotFlag:
                        with open("coordinates.txt", "a") as f:
                            f.write(f"{cx}, {cy}, {hotspotCount}, {object_name}\n")
                        countFile+=1
                    with open("pixel.txt", "a") as f:
                            f.write(f"{cx}, {cy}\n")
                    print("The Hotspot coordinates:",cx,cy,hotspotCount,object_name)
                    hotspotCount +=1
                    hotspotDetectFlag = True
                    #print("The X and Y", cx, " &", cy) 

                if object_name == "target":
                    if targetFlag:
                        with open("coordinates.txt", "a") as f:
                            f.write(f"{cx}, {cy}, {targetCount}, {object_name}\n")
                        countFile+=1
                    with open("pixel.txt", "a") as f:
                            f.write(f"{cx}, {cy}\n")
                    print("The Targets coordinates:",cx,cy,count,object_name)     
                    targetDetectFlag = True 
                print(label)
                interruption_flag.set()

        if hotspotDetectFlag == True: 
            hotspotFlag = False
        if targetDetectFlag == True:
            targetFlag = False
        if interruption_flag.clear():
            hotspotFlag = True
            targetFlag = True
            hotspotDetectFlag = False
            targetDetectFlag = False

        cv2.imshow('Detect', frame)
        #count +=1
        #cv2.imwrite(f'{count}.jpg',frame)
        # Write the frame to the video file

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user")
            break
    # cap.release()
    cv2.destroyAllWindows()

'''def hotspotFlagcheck():
    global hotspotFlag,targetFlag
    while True:
        if areaFlag:
            hotspotFlag = True
            targetFlag = True'''


def take_picture():
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

'''def restartcam():
    global image_detect

    # Start the initial thread
    time.sleep(5)
    processed_thread = threading.Thread(target=show_processed_frame, args=(cap, lock, stop_event))
    time.sleep(1)
    processed_thread.start()

    while True:
        if not processed_thread.is_alive():
            print("Thread is not alive, restarting...")
            # Clear the stop event before restarting the thread
            stop_event.clear()
            time.sleep(7)

            # Reinitialize and restart the thread
            processed_thread = threading.Thread(target=show_processed_frame, args=(cap, lock, stop_event))
            processed_thread.start()
            print("Thread restarted.")

        if image_detect:
            print("Image detected, waiting for the thread to finish...")
            image_detect = False
            stop_event.set()  # Signal the thread to stop
            processed_thread.join()  # Wait for the thread to finish
            time.sleep(1)
            print("Thread killed after image detection.")  '''   

def clearCoordinatesFile():
    with open("coordinates.txt", "w") as f:
        pass  # Opening in write mode with 'w' will clear the file content


def readingLastCoordinates():
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

def getLatLon(x_pixel,y_pixel):
    #change later
    x_c = image_width // 2
    y_c = image_height // 2

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
    Dx = abs(np.tan(theta_x_rad) * vehicle.globalrelativeframe.alt)
    Dy = abs(np.tan(theta_y_rad) * vehicle.globalrelativeframe.alt)

    mid_lat = vehicle.globalrelativeframe.lat
    mid_lon = vehicle.globalrelativeframe.lon

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

def generate_grid(geofence, x_divisions, y_divisions):
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
    global leftToRightFlag
    print("The serpentine path started")
    # resume_thread = threading.Thread(target=restartcam, name="Resume the detection")   
    # resume_thread.start()
    model_thread = threading.Thread(target=detection_thread, args=(cap, lock, stop_event))   
    model_thread.start()
    time.sleep(0.1)
    for i, row in enumerate(grid):
        if i % 2 == 0:
            leftToRightFlag = True
            # Move from left to right
            for point in row:
                cam = True
                print("Move left to right");
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(0.1)
        else:
            leftToRightFlag = False
            # Move from right to left
            for point in reversed(row):
                print("move right to left")
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(0.4)    
    
    # stop the thread
    model_thread.join()

# Function to arm the drone and take off to a specified altitude
def arm_and_takeoff(aTargetAltitude):
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

# Function to set the yaw angle
def set_yaw(heading, relative=False):
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

def get_distance_metres(location1, location2):
    dlat = location2.lat - location1.lat
    dlong = location2.lon - location1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

def go_to_location(latitude, longitude, altitude):
    global is_new_hotspot ,moveToaviFlag
    print(f"Going to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}")
    target_location = LocationGlobalRelative(latitude, longitude, altitude)
    vehicle.simple_goto(target_location, groundspeed=3)
    
    while True:
        current_location = vehicle.location.global_relative_frame
        distance_to_target = get_distance_metres(current_location, target_location)
        print(f"Distance to target: {distance_to_target:.2f} meters")

        if is_new_hotspot and moveToaviFlag:
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
        time.sleep(0.5)
    # return 

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
    global tarLat,tarLon,moveToaviFlag
    
    # Store the hotspot coordinates
    # visited_hotspots.append(hotspot_coordinates)                
    # print("Hotspot coordinates stored:", hotspot_coordinates)
    set_yaw(vehicle.heading)
    go_toLocation(tarLat,tarLon)
    time.sleep(0.1)
    #visited_hotspots.append(tarLat,tarLon)
    pic_thread = threading.Thread(target=take_picture,name="pic_image" ,args=(cap, lock))
    pic_thread.start()
    time.sleep(1)
    pic_thread.join()  # Wait for the thread to finish before starting the next one

    moveToaviFlag = False
    
        # if typeDetect == "target":
        #     moveToavipart()
        
        


def main():
    start_time = time.time()

    print("Mission Begins")
    print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")
    loopCheckThread = threading.Thread(target=loopCheckingThread_1)   
    loopCheckThread.start()
    time.sleep(0.1)

    grid = generate_grid(geofence, x_divisions,y_divisions)
    time.sleep(2)
    arm_and_takeoff(alt)
    time.sleep(5)

    serpentine_path_grid(grid)
    time.sleep(5)

    print("Mission completed, landing...")
    vehicle.mode = VehicleMode("RTL")

    end_time = time.time()
    print("Total time taken =",(end_time-start_time))
    print("total object detected are = ", object_count)
    for coord in visited_hotspots:
        print("Hotspot already visited. ",coord)

    time.sleep(2)
    loopCheckThread.join()
    #end_time = time.time()
    #print("Total time taken =",(end_time-start_time))
    #print("total object detected are = ", object_count)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    vehicle.close()


if __name__ == "__main__":
    main()

