# import cv2
# import numpy as np

# def detect_shapes_and_colored_objects_in_video(video_path):
#     # Initialize video capture
#     # cap = cv2.VideoCapture(video_path)

#     img= "object/unnamed.jpg"
    
#     # Initialize object count
#     total_object_count = 0

#     # Process each frame
#     while True:
#         # ret, frame = cap.read()
        
#         # if not ret:
#         #     break
#         frame = cv2.imread(img)
        
#         # Resize the frame to a larger size
#         #resized_frame = cv2.resize(frame, (800, 600))
        
#         # Convert resized frame to HSV color space
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # Define lower and upper bounds for white color in HSV
#         lower_white = np.array([0, 0, 200])
#         upper_white = np.array([180, 30, 255])
        
#         # Threshold the HSV frame to get white regions
#         mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
#         # Invert the white mask
#         mask_non_white = cv2.bitwise_not(mask_white)
        
#         # Find contours of non-white regions
#         contours, _ = cv2.findContours(mask_non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Dictionary to store counts of different shape categories
#         shape_categories = {
#             "Triangle": 0,
#             "Square": 0,
#             "Circle": 0,
#             "Unknown": 0
#         }
        
#         # Get dimensions of the frame
#         height, width, _ = resized_frame.shape

#         # Iterate through contours
#         for contour in contours:
#             # Calculate contour area
#             area = cv2.contourArea(contour)
            
#             # Ignore contours with small area
#             if area < 100:  # Adjust the threshold according to your needs
#                 continue
            
#             # Ignore contours that are too close to the frame edges
#             x, y, w, h = cv2.boundingRect(contour)
#             if x <= 5 or y <= 5 or (x + w) >= (width - 5) or (y + h) >= (height - 5):
#                 continue
            
#             # Get the perimeter of the contour
#             perimeter = cv2.arcLength(contour, True)
            
#             # Approximate the contour to identify the shape
#             epsilon = 0.04 * perimeter
#             approx = cv2.approxPolyDP(contour, epsilon, True)
            
#             # Classify shape based on the number of vertices
#             num_vertices = len(approx)
            
#             if num_vertices == 3:
#                 shape_category = "Triangle"
#             elif num_vertices == 4:
#                 # Further classify between square and rectangle
#                 (x, y, w, h) = cv2.boundingRect(approx)
#                 aspect_ratio = float(w) / h
#                 if 0.95 <= aspect_ratio <= 1.05:  # Check if the aspect ratio is close to 1
#                     shape_category = "Square"
#                 else:
#                     shape_category = "Unknown"  # Ignore rectangles
#             elif num_vertices > 4:
#                 # Check for circles
#                 area = cv2.contourArea(contour)
#                 if area > 0:
#                     circularity = 4 * np.pi * (area / (perimeter * perimeter))
#                     if circularity > 0.7:  # Adjust threshold for circularity
#                         shape_category = "Circle"
#                     else:
#                         shape_category = "Unknown"
#                 else:
#                     shape_category = "Unknown"
#             else:
#                 shape_category = "Unknown"
            
#             # Increment the count for the detected shape category
#             if shape_category != "Unknown":
#                 shape_categories[shape_category] += 1
#                 total_object_count += 1
            
#                 # Draw bounding box
#                 cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
#                 # Add shape category label to the rectangle
#                 cv2.putText(resized_frame, shape_category, (x + int(w/2) - 30, y + int(h/2) + 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Display the processed frame
#         cv2.imshow("Output", resized_frame)
        
#         # Write the number of shapes on the frame
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(resized_frame, f'Number of objects: {total_object_count}', (10, 30), font, .5, (0, 255, 0), 2, cv2.LINE_AA)
        
#         # Print counts of different shape categories
#         print("Shape Categories in Current Frame:")
#         for category, count in shape_categories.items():
#             print(f"{category}: {count}")
      
#         # Wait for a short period to slow down the video playback
#         key = cv2.waitKey(10)  # 100 ms delay between frames
        
#         # Break the loop when 'q' is pressed
#         if key & 0xFF == ord('q'):
#             break
    
#     # Release the video capture and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Test the function
# video_path = "/home/avinash/Documents/Aerothon/video.mp4"  # Provide the path to your video
# detect_shapes_and_colored_objects_in_video(video_path)


import cv2
import numpy as np

def detect_shapes_and_colored_objects_in_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Image not found.")
        return
    
    # Resize the frame to a larger size
    resized_frame = cv2.resize(frame, (800, 600))
    
    # Convert resized frame to HSV color space
    hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Threshold the HSV frame to get white regions
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert the white mask
    mask_non_white = cv2.bitwise_not(mask_white)
    
    # Find contours of non-white regions
    contours, _ = cv2.findContours(mask_non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dictionary to store counts of different shape categories
    shape_categories = {
        "Triangle": 0,
        "Square": 0,
        "Circle": 0,
        "Unknown": 0
    }
    
    # Get dimensions of the frame
    height, width, _ = resized_frame.shape

    # Iterate through contours
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Ignore contours with small area
        if area < 10:  # Adjust the threshold according to your needs
            continue
        
        # Ignore contours that are too close to the frame edges
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 5 or y <= 5 or (x + w) >= (width - 5) or (y + h) >= (height - 5):
            continue
        
        # Get the perimeter of the contour
        perimeter = cv2.arcLength(contour, True)
        
        # Approximate the contour to identify the shape
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify shape based on the number of vertices
        num_vertices = len(approx)
        
        if num_vertices == 3:
            shape_category = "Triangle"
        elif num_vertices == 4:
            # Further classify between square and rectangle
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:  # Check if the aspect ratio is close to 1
                # Calculate contour area and compare it with the area of bounding rectangle
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                if abs(contour_area - rect_area) < 0.1 * rect_area:  # Tolerance for contour area vs bounding box area
                    shape_category = "Square"
                else:
                    shape_category = "Unknown"  # Ignore rectangles that are not squares
            else:
                shape_category = "Unknown"  # Ignore rectangles
        elif num_vertices > 4:
            # Check for circles
            area = cv2.contourArea(contour)
            if area > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.7:  # Adjust threshold for circularity
                    shape_category = "Circle"
                else:
                    shape_category = "Unknown"
            else:
                shape_category = "Unknown"
        else:
            shape_category = "Unknown"
        
        # Increment the count for the detected shape category
        if shape_category != "Unknown":
            shape_categories[shape_category] += 1

            # Draw bounding box
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add shape category label to the rectangle
            cv2.putText(resized_frame, shape_category, (x + int(w/2) - 30, y + int(h/2) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the number of shapes on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_frame, f'Total objects: {sum(shape_categories.values())}', (10, 30), font, .5, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Print counts of different shape categories
    print("Shape Categories in the Image:")
    for category, count in shape_categories.items():
        print(f"{category}: {count}")
    
    # Display the processed image
    cv2.imshow("Output", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function
image_path = "im.png"  # Provide the path to your image
detect_shapes_and_colored_objects_in_image(image_path)
