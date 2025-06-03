import cv2
from picamera2 import Picamera2
import warnings

import time
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"image_{timestamp}.mp4"
# Suppress libpng sRGB profile warnings
warnings.filterwarnings("ignore", category=UserWarning, module="libpng")

picam2 = Picamera2()
picam2.start()


# Define the codec and create VideoWriter object
# Use 'mp4v' codec for .mp4 files or 'X264' for .mkv files
# FourCC code is platform dependent, you may need to try different ones
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
# fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv

out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

try:
    while True:
        frame = picam2.capture_array()

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        # Write the frame to the video file
        out.write(frame)
        cv2.imshow('Object Detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user")
            break

except KeyboardInterrupt:
    print("Recording interrupted by user")

finally:
    # Clean up resources properly
    picam2.stop()
    picam2.close()
    out.release()
    cv2.destroyAllWindows()

