from picamera2 import Picamera2, Preview
import time

# Initialize the camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration()

# Set the resolution and frame rate, with additional settings to handle buffer
video_config["buffer_count"] = 3  # Set a higher buffer count for smoother playback
video_config["size"] = (1280, 720)  # Adjust as needed

picam2.configure(video_config)
output_file = "video_output.h264"

picam2.start_recording(output_file)
time.sleep(10)  # Record for 10 seconds

picam2.stop_recording()
picam2.close()
