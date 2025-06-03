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
print('Connecting to vehicle on:')
# vehicle = connect(args.connect, baud=921600, wait_ready=True, timeout=60)
vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True, timeout=60)


# Parameters
alt = 10  # Desired altitude in meters
groundspeed = 3  # Ground speed in m/s
x_divisions = 9
y_divisions = 13
# Define Geofence Corners (latitude, longitude)
geofence = [
    (15.3677944678254, 75.1253820955753),  # Bottom-Left
    (15.3675248, 75.1253325),  # Bottom-Right
    (15.3675067, 75.1255846),  # Top-Right
    (15.3677886, 75.1256061)   # Top-Left
]

def generate_grid(geofence,x_divisions, y_divisions):
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
            print(lat,lon)
        grid.append(row)
    return grid

def serpentine_path_grid(grid):
    for i, row in enumerate(grid):
        if i % 2 == 0:
            # Move from left to right
            for point in row:
                print("Move left to right")
                lat, lon = point
                go_to_location(lat, lon, alt)
                time.sleep(1)
        else:
            # Move from right to left
            for point in reversed(row):
                print("Move Right to left")
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
    
        if distance_to_target <= 1.0:
            print("Reached target location.")
            break
        time.sleep(0.5)
    # return 

def get_distance_metres(location1, location2):
    dlat = location2.lat - location1.lat
    dlong = location2.lon - location1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5


def main():
    start_time = time.time()

    print("Mission Begins")
    print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")
    
    
    grid = generate_grid(geofence, x_divisions,y_divisions)
    time.sleep(2)
    arm_and_takeoff(alt)
    time.sleep(2)

    serpentine_path_grid(grid)
    time.sleep(5)

    print("Mission completed, landing...")
    vehicle.mode = VehicleMode("RTL")

    end_time = time.time()
    print("Total time taken =",(end_time-start_time))
   
    vehicle.close()


if __name__ == "__main__":
    main()

