from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import argparse
from pymavlink import mavutil



# Connect to the Vehicle
vehicle = connect("/dev/ttyACM0", baud=921600, wait_ready=True, timeout=60)

def main():
    
    start_time = time.time()
    print("Mission Begins")
    while True:
        print(" Altitude:", vehicle.location.global_relative_frame.alt)

if __name__ == "__main__":
    main()
