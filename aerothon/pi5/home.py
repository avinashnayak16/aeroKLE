from dronekit import connect, VehicleMode, LocationGlobalRelative

print('Connecting to vehicle on:')
# vehicle = connect(args.connect, baud=921600, wait_ready=True, timeout=60)
vehicle = connect('/dev/ttyACM0', baud=921600, wait_ready=True, timeout=60)

print("Mission Begins")
print(f"Home Location: {vehicle.location.global_frame.lat}, {vehicle.location.global_frame.lon}")

vehicle.close()
