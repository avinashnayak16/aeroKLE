# Import packages
import os
import cv2
import numpy as np
import sys
# import importlib.util
from tflite_runtime.interpreter import Interpreter
from datetime import datetime
import time
from picamera2 import Picamera2

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.mp4"

    MODEL_NAME = 'targets'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = float(0.8)
    count = 0
    hotspotFlag = False
    targetFlag = False

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
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

    # Open video file
    # videopath="/home/avinash/Music/model/manual/image_20240817_161049.mp4"
    #video = cv2.VideoCapture(0)
    picam2 = Picamera2(-1)
    picam2.start()
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    # Define the codec and create VideoWriter object
    # Use 'mp4v' codec for .mp4 files or 'X264' for .mkv files
    # FourCC code is platform dependent, you may need to try different ones
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
    # fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv

    out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

    try:
        while(video.isOpened()):

            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame = video.read()
            if not ret:
                print('Reached the end of the video!')
                break
            # ret ,frame = picam2.capture_array()
            # if not ret:
            #     print('Reached the end of the video!')
            #     break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imwrite("he.jpg",frame_rgb)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            # cv2.imwrite("hell.jpg",frame_rgb)
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            count = 0
            hotspotCount = 0  # inialise the count of hotspot in the frame
            targetCount = 0
            #clearCoordintes()  # to clear the file data 
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                    cx = (xmax+xmin)//2
                    cy = (ymax+ymin)//2
                    print("The coordinates:",cx,cy)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    

            # All the results have been drawn on the frame, so it's time to display it.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
            frame_output = cv2.resize(frame, (int(imW), int(imH)))
            out.write(frame_output)
            
            cv2.imshow('Detect', frame)
            count +=1

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user")
                break
    except KeyboardInterrupt:
        print("Recording interrupted by user")
    finally:
        # Release everything when done
        video.release()
        out.release()    
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
