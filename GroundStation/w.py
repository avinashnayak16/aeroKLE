import os
import argparse
import cv2
import numpy as np
import importlib.util
import time

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='Shapes')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.8)
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')
args = parser.parse_args()

# Model and label file paths
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Load TensorFlow Lite model and label map
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')] if use_TPU else None)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1:3]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean, input_std = 127.5, 127.5
outname = output_details[0]['name']
boxes_idx, classes_idx, scores_idx = (1, 3, 0) if 'StatefulPartitionedCall' in outname else (0, 1, 2)

# RTSP Stream
sour = "rtsp://192.168.43.1:8554/fpv_stream"
reconnect_attempts = 0

while True:
    # Attempt to connect or reconnect
    cam = cv2.VideoCapture(sour)
    if not cam.isOpened():
        print("[INFO] Unable to connect to the RTSP stream. Retrying...")
        time.sleep(2)  # Wait before retrying
        reconnect_attempts += 1
        if reconnect_attempts > 5:  # Limit the reconnection attempts
            print("[ERROR] Could not reconnect to RTSP stream after several attempts.")
            break
        continue

    try:
        reconnect_attempts = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                print("[WARNING] Frame read failed. Attempting to reconnect...")
                break  # Exit inner loop and attempt reconnect

            # Resize and preprocess frame
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(cv2.resize(frame_rgb, (width, height)), axis=0)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Model inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            boxes, classes, scores = (interpreter.get_tensor(output_details[idx]['index'])[0] for idx in [boxes_idx, classes_idx, scores_idx])

            # Display detected objects
            total_square_count, total_circle_count, total_triangle_count = 0, 0, 0
            for i in range(len(scores)):
                if min_conf_threshold < scores[i] <= 1.0:
                    ymin, xmin = int(max(1, (boxes[i][0] * 480))), int(max(1, (boxes[i][1] * 640)))
                    ymax, xmax = int(min(480, (boxes[i][2] * 480))), int(min(640, (boxes[i][3] * 640)))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
                    object_name = labels[int(classes[i])]
                    label = f'{object_name}: {int(scores[i] * 100)}%'
                    labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (xmin, ymin - labelSize[1] - 10), (xmin + labelSize[0], ymin + 5), (255, 255, 255), -1)
                    cv2.putText(frame, label, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    if object_name == "square": total_square_count += 1
                    elif object_name == "circle": total_circle_count += 1
                    elif object_name == "triangle": total_triangle_count += 1

            # Display counts and frame
            text = f"Squares: {total_square_count}, Circles: {total_circle_count}, Triangles: {total_triangle_count}"
            cv2.putText(frame, text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow('Object detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("[INFO] Process interrupted.")
        break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and all windows closed.")
