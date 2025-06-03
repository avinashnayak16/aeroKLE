import cv2
import numpy as np

# Load your calibration data (these are example values; replace them with your calibration data)
mtx = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])  # Example camera matrix
dist = np.array([0.1, -0.25, 0, 0, 0])  # Example distortion coefficients

# Open the camera

sour = "rtsp://192.168.43.1:8554/fpv_stream"
cap = cv2.VideoCapture(0)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get optimal new camera matrix
ret, frame = cap.read()
h, w = frame.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Loop to continuously get frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break
    
    # Undistort the frame
    frame = cv2.resize(frame,(640,480))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)
    
    # Optionally, crop the image to the valid ROI
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    
    # Display the undistorted frame
    cv2.imshow('Undistorted Camera Feed', undistorted_frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

'''
import cv2

sour = "rtsp://192.168.43.1:8554/fpv_stream"
cap = cv2.VideoCapture(sour)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the frame
       
        # Display or process the undistorted frame
        cv2.imshow("Undistorted Frame", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Recording interrupted by user")

finally:
    # Clean up resources properly
    cap.release()
    cv2.destroyAllWindows()'''
