import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best.pt')  # Replace with your YOLO model path if custom

# Set up video capture
video_path = 'cadxx.mp4'  # Replace with your video path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Set frame skip value
frame_skip = 5  # Skip every 5 frames; adjust as needed
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only run inference on every (frame_skip)th frame
    if frame_count % frame_skip == 0:
        results = model(frame)  # Run YOLO model on the current frame

        # results is a list; get the first result
        result = results[0]

        # Draw bounding boxes and labels on the frame
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = box.conf[0]                      # Confidence score
            cls = int(box.cls[0])                   # Class label index

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {cls} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("YOLO Output", frame)

    # Increment frame count
    frame_count += 1

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

