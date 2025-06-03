from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import numpy as np
import cv2
from math import sqrt, pow
from tflite_runtime.interpreter import Interpreter

# List to store visited hotspots , Define waypoint radius
visited_hotspots = []
WAYPOINT_RADIUS = 1.5  # meters

def create_exclusion_zone(latitude, longitude, radius):
    exclusion_zones.append((latitude, longitude, radius))
    print(f"Exclusion zone created at ({latitude}, {longitude}) with radius {radius} meters")

# Function to check if a point is within any exclusion zones
def is_within_exclusion_zone(latitude, longitude):
    for zone in exclusion_zones:
        zone_lat, zone_lon, zone_radius = zone
        distance = calculate_distance(latitude, longitude, zone_lat, zone_lon)
        if distance < zone_radius:
            return True
    return False

# Function to calculate distance between two GPS coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) * 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) * 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance
    
def detect_hotspot():
    # Placeholder for hotspot detection logic using TFLite model
    # You should integrate the camera feed with the MobileNet model
    # Example: ret, frame = cap.read()
    # Hotspot detected - for now, let's return some dummy GPS coordinates
    detected = False
    hotspot_coordinates = None
    
    # Integrate the model with your camera feed here
    # e.g., preprocess frame, run inference, and detect hotspot
    
    # For the example, let's assume a detection occurs
    detected = True
    hotspot_coordinates = vehicle.location.global_relative_frame

    return detected, hotspot_coordinates
        
def loopChecking():
    while True:
        # Detect hotspot
        detected, hotspot_coordinates = detect_hotspot()
        
        if detected:
            # Check if the hotspot has already been visited
            is_new_hotspot = True
            for coord in visited_hotspots:
                if get_distance_meters(coord, hotspot_coordinates) <= WAYPOINT_RADIUS:
                    print("Hotspot already visited. Skipping...")
                    is_new_hotspot = False
                    break
            
            if is_new_hotspot:
                # Navigate to the hotspot
                moveToaviFlag = True
                
                # Store the hotspot coordinates
                visited_hotspots.append(hotspot_coordinates)                
                print("Hotspot coordinates stored:", hotspot_coordinates)
        
        # Add some delay before next detection
        time.sleep(1)
        
try:
    # Set the vehicle to GUIDED mode
    print("Switching to GUIDED mode.")
    vehicle.mode = VehicleMode("GUIDED")
    
    # Main mission loop
    main()

except KeyboardInterrupt:
    print("Mission aborted by user.")
    
finally:
    print("Returning to Launch")
    vehicle.mode = VehicleMode("RTL")
    vehicle.close()