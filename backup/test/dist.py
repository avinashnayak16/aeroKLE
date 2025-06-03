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
altitude=15

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
    
        if distance_to_target <= 1.0:
            print("Reached waypoint.")
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
        velocity_y, velocity_x, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    vehicle.send_mavlink(msg)
    vehicle.flush()

def moveToavi():

    
    print("Move Towards target")
    x_coord = 540
    y_coord = 240
    h_x = 540
    h_y = 240

    def send_velocity_based_on_position(x_coord,y_coord,g_speed):
        if x_coord == h_x and y_coord == h_y:
            send_ned_velocity(0, 0, 1)
        elif x_coord > h_x and y_coord > h_y:
            send_ned_velocity(g_speed, -g_speed, g_speed)
        elif x_coord < h_x and y_coord < h_y:
            send_ned_velocity(-g_speed, g_speed, g_speed)
        elif x_coord < h_x and y_coord > h_y:
            send_ned_velocity(-g_speed, -g_speed, g_speed)
        elif x_coord > h_x and y_coord < h_y:
            send_ned_velocity(g_speed, g_speed, g_speed)
        elif x_coord == h_x and y_coord != h_y:
            if y_coord > h_y:
                send_ned_velocity(0, -g_speed, g_speed)
            elif y_coord < h_y:
                send_ned_velocity(0, g_speed, g_speed)
        elif y_coord == h_y and x_coord != h_x:
            if x_coord > h_x:
                send_ned_velocity(g_speed, 0, g_speed)
            elif x_coord < h_x:
                send_ned_velocity(-g_speed, 0, g_speed)

    prev_x, prev_y = None, None

    # for i in range(5):
    while True:
       
        send_velocity_based_on_position(540, 240, 0.3)
        time.sleep(1)

        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        prev_x, prev_y = x_coord,y_coord 

        #if((abs(x_coord-h_x)<=50) and (abs(y_coord-h_y)<=50)) and vehicle.location.global_relative_frame.alt <=5:
        #    break
        if vehicle.location.global_relative_frame.alt <= 2.5:
                break
    time.sleep(1)


def main():
    start_time = time.time()

    print("Mission Begins")
    print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")

    try:
        
        arm_and_takeoff(altitude)
        time.sleep(2)

        moveToavi()


        # mission for objects

        print("Mission completed, Returning to Launch...")
        vehicle.mode = VehicleMode("LAND")
        time.sleep(2)
        
        end_time = time.time()
        print("Total time taken =",(end_time-start_time))
        
        vehicle.close()
    
    except KeyboardInterrupt:
        print("Recording interrupted by user")
    finally:
        
        vehicle.close()


if __name__ == "__main__":
    main()
