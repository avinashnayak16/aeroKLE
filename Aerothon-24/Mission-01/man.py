# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from datetime import datetime
from threading import Thread
import importlib.util

# from picamera2 import Picamera2
# picam2 = Picamera2()
# picam2.start()
cap = cv2.VideoCapture('photo.mp4')

start_time = time.time()
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# location = vehicle.location.global_frame
# filename = f"image_{timestamp}_lat{location.lat}_lon{location.lon}_alt{location.alt}.jpg"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"image_{timestamp}.mp4"
frameCount = 0
shapesCount = 0
objectCount = 0

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=False, default='objects')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.8)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_width = int(640)
frame_height = int(480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
# fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv

out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

def shapesDetection(frame):
    global shapesCount
    print("Shapes Detection started")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                            required=False,default='shapes')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.8)
    parser.add_argument('--image', help='Name of the single image to perform detection on.', default=None)
    parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder', action='store_true')
    parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)', action='store_false')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

    args = parser.parse_args()

    # Parse user inputs
    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    use_TPU = args.edgetpu
    save_results = args.save_results
    show_results = args.noshow_results
    IM_NAME = args.image

    # Import TensorFlow Lite Interpreter or regular TensorFlow based on the package
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign the filename for Edge TPU model
    if use_TPU:
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Get the current working directory
    CWD_PATH = os.getcwd()

    images = frame.copy()

    # Path to .tflite file (the model used for object detection)
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # If the first label is '???', remove it
    if labels[0] == '???':
        del(labels[0])

    # Load the TensorFlow Lite model (with or without Edge TPU support)
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Determine if the model is based on TF2 or TF1
    outname = output_details[0]['name']

    if 'StatefulPartitionedCall' in outname:  # TF2 Model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:  # TF1 Model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Function to detect shapes (squares and circles) within a specified region of interest (ROI)
    def detect_shapes_in_bbox(image, bbox):
        # Extract the region of interest (ROI) using the bounding box coordinates
        xmin, ymin, xmax, ymax = bbox
        roi = image[ymin:ymax, xmin:xmax]

        # Convert the ROI to grayscale and apply thresholding to detect shapes
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        circle_detected = False

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            if len(approx) > 8:  # More than 8 vertices indicates a circle
                circle_detected = True

        return circle_detected

    # Load and preprocess the image
    image = frame.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize if using a floating model
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Run the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    total_square_count = 0
    total_circle_count = 0
    total_triangle_count = 0

    # Loop over the results and draw boxes for objects above the confidence threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))


            # Detect shapes within the bounding box
            circle_detected = detect_shapes_in_bbox(image, (xmin, ymin, xmax, ymax))
            color = (0, 0, 255) if circle_detected else (0, 255, 0)  # Red for circles, green for others

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label
            object_name = labels[int(classes[i])]  # Get object name from labels
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # e.g., 'person: 72%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                            (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if object_name == "square":
                total_square_count+=1
            elif object_name == "circle":
                total_circle_count+=1
            elif object_name == "triangle":
                total_triangle_count+=1
            # cv2.imwrite(f"image_{timestamp}.jpg", image)


    # Display the total count of squares and circles on the image
    # Display the total count of squares and circles on the image
    text = f"Squares: {total_square_count}, Circles: {total_circle_count},Triangles:{total_triangle_count}"
    cv2.putText(image, text, (10, imH - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(f"image_{timestamp}.jpg", image)
    shapesCount+=1

    # return image

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
try:
    while True:
        # Grab frame from video stream
        ret ,frame = cap.read()

    
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Object detector', frame)
        frame_output = cv2.resize(frame, (int(imW), int(imH)))
        out.write(frame_output)
        end_time = time.time()
        if (end_time-start_time) >=300:
            break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user")
            break

except KeyboardInterrupt:
    print("Recording interrupted by user")
finally:
    # Release everything when done
    cap.release()
    # picam2.stop()
    # picam2.close()
    out.release()    
    cv2.destroyAllWindows()

# Clean up
cap.release()
# picam2.stop()
# picam2.close()
cv2.destroyAllWindows()