import cv2
import time
from datetime import datetime
from picamera2 import Picamera2

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"image_{timestamp}.mp4"
# Initialize video stream
picam2 = Picamera2()
picam2.start()

def main():
    # Open the default camera (0) for video capture
    
    # Define the codec and create VideoWriter object
    # Use 'mp4v' codec for .mp4 files or 'X264' for .mkv files
    # FourCC code is platform dependent, you may need to try different ones
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
    # fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv

    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    try:
        while True:
            frame=picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Write the frame to the video file
            out.write(frame)

            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Check for 'q' key press to break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user")
                break
    except KeyboardInterrupt:
        print("Recording interrupted by user")
    finally:
        # Release everything when done
        picam2.stop()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
