import cv2

sour = "rtsp://192.168.43.1:8554/fpv_stream"
cap = cv2.VideoCapture(sour)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Undistorted Frame", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Recording interrupted by user")

finally:
    # Clean up resources properly
    cap.release()