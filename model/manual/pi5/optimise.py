import cv2
import numpy as np
import time
from picamera2 import Picamera2
# Initialize webcam
picam2 = Picamera2(-1)
picam2.start()

while True:
    # Capture frame
    ret, frame = picam2()
    
    if not ret:
        break
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red color in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV frame to create a mask for red colors
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store information about the largest contour
    largest_contour = None
    largest_contour_area = 0

    # Find the largest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_contour_area:
            largest_contour = contour
            largest_contour_area = area

    # Draw a circle at the centroid of the largest contour
    if largest_contour is not None and len(largest_contour) >= 5:
        # Calculate centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            print("X coordinate:", cx)
            print("Y coordinate:", cy)
            with open("xa.txt", "a") as f:
                f.write(str(cx) + "\n")
            with open("yb.txt", "a") as f:
                f.write(str(cy) + "\n")
            # Draw a circle at the centroid
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            # Calculate and draw the bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Hotspot", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1) # Draw label text

    # Display the resulting frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    cv2.imshow('Webcam', frame)
    time.sleep(0.04)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

