import os
import cv2
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.10/dist-packages')
from mss import mss

from PIL import Image

# Camera matrix and distortion coefficients (optional: can be used to undistort the screen capture)
mtx = np.array([[845.549571023190, 0, 978.4245716999010], [0, 848.221453033451, 547.499622688556], [0, 0, 1]])
dist = np.array([[-0.340671222, 0.110426603, -0.000867987573, 0.000189669273, -0.0160049526]])

# Screen capture setup
sct = mss()
frame_counter = 0
image_folder = "images"
os.makedirs(image_folder, exist_ok=True)  # Create directory if it doesn't exist

try:
    while True:
        # Define the screen region to capture
        w, h = 640, 480
        monitor = {'top': 180, 'left': 500, 'width': w, 'height': h}
        
        # Capture the screen and convert to NumPy array
        img = Image.frombytes('RGB', (w, h), sct.grab(monitor).rgb)
        img_np = np.array(img)
        
        # Convert to OpenCV format and BGR color space
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # (Optional) Undistort the frame using the camera matrix and distortion coefficients
        # frame = cv2.undistort(frame, mtx, dist, None, mtx)
        
        # Convert the frame to grayscale for contour detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection and contour detection
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original frame
        detected_frame = frame.copy()
        cv2.drawContours(detected_frame, contours, -1, (0, 255, 0), 2)
        
        # Display the frame with detected contours
        cv2.imshow('Screen Detection', detected_frame)
        
        # Save frame if any contours are detected
        if contours:
            frame_filename = os.path.join(image_folder, f"frame_{frame_counter}.jpg")
            cv2.imwrite(frame_filename, detected_frame)
            frame_counter += 1
            print(f"[INFO] Saved {frame_filename}")
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("[INFO] Process interrupted")

finally:
    cv2.destroyAllWindows()
    print("[INFO] All windows closed")
