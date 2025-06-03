import cv2

# Initialize the video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Set up a named window for displaying the cropped frame
cv2.namedWindow("Cropped Video Feed", cv2.WINDOW_NORMAL)

# Define the cropping ratio
crop_ratio = 0.2  # Crop 20% from each side

while True:
    # Read the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Get the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Calculate the number of pixels to crop from each side
    crop_x = int(original_width * crop_ratio / 2)
    crop_y = int(original_height * crop_ratio / 2)

    # Crop equally from all sides
    cropped_frame = frame[crop_y:original_height-crop_y, crop_x:original_width-crop_x]

    # Show the cropped frame
    cv2.imshow("Cropped Video Feed", cropped_frame)
    cv2.imshow("Cropped ", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
