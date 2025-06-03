import cv2
from picamera2 import Picamera2
import warnings

picam2 = Picamera2()
picam2.start()


try:
    while True:
        frame = picam2.capture_array()

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
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
    cv2.destroyAllWindows()
