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

'''parser = argparse.ArgumentParser()
parser.add_argument('--connect', default='127.0.0.1:14550')
args = parser.parse_args()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % args.connect)
vehicle = connect(args.connect, baud=921600, wait_ready=True, timeout=60)
'''
vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True, timeout=60)

# Parameters
alt = 6  # Desired altitude in meters
groundspeed = 3  # Ground speed in m/s
# gridSize = 10  # Size of each grid cell in meters (10m x 10m)
x_divisions = 12
y_divisions = 6
# Define Geofence Corners (latitude, longitude)
geofence = [
    (15.3675914, 75.1254478),  # Bottom-Left
    (15.3673490, 75.1254800),  # Bottom-Right
    (15.3673516, 75.1257482),  # Top-Right
    (15.3675998, 75.1256624)   # Top-Left
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
    for i, row in enumerate(grid):
        if i % 2 == 0:
            # Move from left to right
            for point in row:
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(2)
        else:
            # Move from right to left
            for point in reversed(row):
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(2)


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

        '''if interruption_flag.is_set():
            print("Interruption detected. Switching to moveToavi.")
            moveToavi()  # Call moveToavi on interruption
            interruption_flag.clear()
            time.sleep(2)
            vehicle.simple_goto(target_location, groundspeed=3)
            print("resuming back to mission")
            time.sleep(1)
            # break'''
    
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

    for i in range(8):
    # while True:
        x,y = readingCoordinates()
        time.sleep(0.1)
        print("points=",x, y)
        send_velocity_based_on_position(x, y, 0.3)
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
    time.sleep(1)
    resume_thread.start()
    
    grid = generate_grid(geofence, x_divisions,y_divisions)
    time.sleep(2)
    arm_and_takeoff(alt)
    time.sleep(5)

    serpentine_path_grid(grid)
    time.sleep(5)

    print("Mission completed, landing...")
    vehicle.mode = VehicleMode("RTL")

    # original_thread.join()
    # processed_thread.join()
    end_time = time.time()
    print("Total time taken =",(end_time-start_time))
    print("total object detected are = ", object_count)

    time.sleep(2)
    resume_thread.join()
    #end_time = time.time()
    #print("Total time taken =",(end_time-start_time))
    #print("total object detected are = ", object_count)

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    vehicle.close()


if __name__ == "__main__":
    main()

