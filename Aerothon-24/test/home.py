from dronekit import connect, VehicleMode, LocationGlobalRelative
import time 
print('Connecting to vehicle on:')
# vehicle = connect(args.connect, baud=921600, wait_ready=True, timeout=60)
vehicle = connect('/dev/ttyUSB0', baud=57600, wait_ready=True, timeout=60)

while(True):
	print("Mission Begins")
	print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")
	time.sleep(5)

vehicle.close()
