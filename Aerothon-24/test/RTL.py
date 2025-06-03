from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--connect', default='127.0.0.1:14550')
args = parser.parse_args()

# Connect to the Vehicle
print ('Connecting to vehicle on: %s' % args.connect)
vehicle = connect("/dev/ttyAMA0", baud=921600, wait_ready=True, timeout=60)

# Parameters
alt = 3  # Desired altitude in meters
groundpeed = 3  # Ground speed in m/s
gridSize = 10  # Size of each grid cell in meters (10m x 10m)


def arm_and_takeoff(a_target_altitude):
    print("Arming motors")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialize...")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(a_target_altitude)

    while True:
        print(" Altitude:", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= a_target_altitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)


def go_to_location(lat, lon, alt):
    target_location = LocationGlobalRelative(lat, lon, alt)
    vehicle.simple_goto(target_location, groundspeed= 3)

    while True:
        current_location = vehicle.location.global_relative_frame
        dist_to_target = get_distance_metres(current_location, target_location)
        print(f"Distance to target: {dist_to_target:.2f} meters")
        if dist_to_target < 1:  # within 1 meter
            print("Reached target location")
            break
        time.sleep(1)

def goto_location_norm(to_lat, to_lon, alt):
    currentLocation = vehicle.location.global_frame
    target_Location = LocationGlobalRelative(to_lat, to_lon, alt)
    targetDistance = get_distance_metres(currentLocation, target_Location)
    Distance = get_distance_metres(vehicle.location.global_frame, target_Location)
    vehicle.simple_goto(target_Location, groundspeed=2.8)

    while vehicle.mode.name == "GUIDED":
        # print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance = get_distance_metres(vehicle.location.global_frame, target_Location)
        print("Distance to wavepoint: ", remainingDistance)
        if remainingDistance <= 1:  # Just below target, in case of undershoot.
            print("Reached target")
            break
        time.sleep(2)




def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = math.radians(aLocation2.lat - aLocation1.lat)
    dlong = math.radians(aLocation2.lon - aLocation1.lon)
    lat1 = math.radians(aLocation1.lat)
    lat2 = math.radians(aLocation2.lat)

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlong/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    earth_radius = 6371000  # meters
    return earth_radius * c

def main():
    
    start_time = time.time()
    print("Mission Begins")
    print("Home Location",vehicle.location.global_frame.lat)
    print(vehicle.location.global_frame.lon)

    arm_and_takeoff(alt)
    #go_to_location(15.3676587,75.1255162,alt)

    time.sleep(60)

    print("Mission completed, landing...")
    vehicle.mode = VehicleMode("RTL")
    end_time = time.time()
    print("Total time taken =",(end_time-start_time))
    print(time.time())
    vehicle.close()

if __name__ == "__main__":
    main()
