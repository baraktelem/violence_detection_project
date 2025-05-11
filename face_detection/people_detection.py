from roboflow import Roboflow
import supervision as sv
import cv2
from tracker import Tracker  # Import the Tracker class
import os
import datetime
import requests  # Add requests import
import time  # Add time import

import skeleton  # Import the skeleton detection module

# model parameters
# These parameters are used to set the initial confidence levels for detection and tracking in the model.
MODEL_DETECTION_CONFIDENCE_THRESHOLD = 50  # his line sets the initial confidence threshold for the model's predictions (for all predictions)
MODEL_ALLOWED_OVERLAP = 20     # A value of x means that if two bounding boxes overlap by more than x%, the one with the lower confidence score will be removed.
MODEL_PERSON_DETECTION_THRESHOLD = 0.1  # Confidence threshold for detecting a person

# tracker parameters
# These parameters are used to set the maximum number of frames to track an object before re-detection and the overlap threshold for bounding boxes.
CREATE_NEW_TRACKER_THRESHOLD = 0.2 # If IoU with any existing tracker is above this, no new tracker is created

# skeleton pose detection parameters
# These parameters are used to set the minimum confidence levels for detection and tracking in the skeleton detection model.
SKELETON_MIN_DETECTION_CONFIDENCE = 0.5 # minimum confidence for detection
SKELETON_MIN_TRACKING_CONFIDENCE = 0.5 # minimum confidence for tracking


def track_from_prediction(frame, prediction: dict):
    # Iterate through existing trackers
    for tracker in tracker_list:
        # Calculate IoU between the prediction and the tracker's bounding box
        iou = tracker.check_bbox_overlap(prediction)
        
        # Update the tracker with the new prediction if its tracking the predicted person
        if iou > CREATE_NEW_TRACKER_THRESHOLD:
            # Instead of destroying and reinitializing the same tracker,
            # create a new one to replace it
            new_tracker = Tracker(tracker.bbox_color)
            new_tracker.initialize_tracker(frame, prediction)
            tracker_list[tracker_list.index(tracker)] = new_tracker
            return
    
    # Create a new tracker instance
    new_tracker = Tracker()
    
    # Initialize the new tracker with the current frame and prediction
    new_tracker.initialize_tracker(frame, prediction)
    
    # Add the new tracker to our list of active trackers
    tracker_list.append(new_tracker)


def update_trackers(frame) -> list[tuple]:
    # If there are no active trackers, return immediately
    if len(tracker_list) == 0:
        return None

    bbox_list = []  # List to store bounding boxes of active trackers

    # Iterate over a copy of the tracker list to safely modify it during iteration
    for tracker in tracker_list[:]:
        # Update each tracker with the current frame
        is_active, bbox = tracker.update_tracker(frame)

        if is_active is True:
            if bbox is not None:
                bbox_list.append(bbox)
        # If the tracker is no longer active remove it from the list
        elif is_active is False:
            tracker_list.remove(tracker)  # Remove the tracker from the list

    return bbox_list  # Return the list of active bounding boxes

def draw_results(frame, model_detections, skeleton_results, tracker_bboxes: list):
    
    # add model detections and annotations to show in the video
    annotated_frame = box_annotator.annotate(scene=frame, detections=model_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=model_detections, labels=labels)

    # add skeleton results to the annotade frame
    skeleton.draw_skeleton_landmarks(annotated_frame, skeleton_results)

    for tracker in tracker_list:
        p1 = (int(tracker.bbox[0]), int(tracker.bbox[1]))
        p2 = (int(tracker.bbox[0] + tracker.bbox[2]), int(tracker.bbox[1] + tracker.bbox[3]))
        cv2.rectangle(img=annotated_frame, pt1=p1, pt2= p2, color=tracker.bbox_color, thickness=2, lineType=1)

    return annotated_frame

def make_prediction_with_retry(model, frame, max_retries=5, initial_delay=1):
    """
    Make a prediction with exponential backoff retry mechanism
    """
    for attempt in range(max_retries):
        try:
            return model.predict(frame, 
                               confidence=MODEL_DETECTION_CONFIDENCE_THRESHOLD, 
                               overlap=MODEL_ALLOWED_OVERLAP).json()
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Failed after {max_retries} attempts: {e}")
                # Return empty predictions instead of raising
                return {"predictions": []}
            
            delay = initial_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return {"predictions": []}

# Main code starts here
# Initialize Roboflow with API key
rf = Roboflow(api_key="NeHIyTiEmZ0e7Y5WVLb9")

# Access specific project and model version
project = rf.workspace().project("people-detection-o4rdr")
model = project.version(8).model

# initialize the tracker list
tracker_list = [] # List to store active trackers

# Get input and output directories
input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(output_dir, f'log_{datetime.datetime.now().strftime("%d_%m_%H_%M_%S")}.txt')

# Get all video files from input directory
video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

with open(log_file, 'w') as f:
    f.write(f'Detected {len(video_files)} video files in input directory\n')

# Process each video in input directory
for i, video_file in enumerate(video_files, start=1):  # Start enumeration (i) from 1
    # print(f"\n[{datetime.datetime.now().strftime('%H:%M')}] Processing video: {video_file}")
        
    # Reset tracker list for each video
    tracker_list = []
    
    # Get full path for input video
    video_path = os.path.join(input_dir, video_file)
    video_name = os.path.splitext(video_file)[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize annotators for labeling and bounding boxes
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print(f'Video Parameters:\n\tFrame Width: {frame_width},\n\tFrame Height: {frame_height},\n\tFPS: {fps}')
    
    # Log video parameters
    with open(log_file, 'a') as f:
        f.write(f"\n[{datetime.datetime.now().strftime('%H:%M')}] Processing video {i}: {video_file}\n")
        f.write(f'\t{i}. Video Parameters:\n\t\tFrame Width: {frame_width}\n\t\tFrame Height: {frame_height}\n\t\tFPS: {fps}\n\t\tNumber of Frames: {frame_num}\n')

    # Set up video writer
    output_path = os.path.join(output_dir, f'{video_name}_out.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update trackers first to match new predictions
        # This way we can avoid creating new trackers for the same person
        tracker_bbox_list = update_trackers(frame)

        # get predictions from the model and handle errors
        model_results = make_prediction_with_retry(model, frame)

        # Extract class labels and confidence levels from predictions
        labels = [
            f"{item['class']} ({item['confidence']:.2f})"
            for item in model_results["predictions"]
        ]

        # add predictions to show in the video
        detections = sv.Detections.from_inference(model_results)

        # go through every person prediction and update trackers list
        for prediction in model_results["predictions"]:
            if prediction['class'].lower() == 'person' and prediction['confidence'] >= MODEL_PERSON_DETECTION_THRESHOLD:
                track_from_prediction(frame, prediction)

        #  get skeleton results
        skeleton_results = skeleton.detect_skeleton_poses(frame, 
                                                          min_detection_confidence=SKELETON_MIN_DETECTION_CONFIDENCE, 
                                                          min_tracking_confidence=SKELETON_MIN_TRACKING_CONFIDENCE)
        
        # display results
        annotated_frame = draw_results(frame, detections, skeleton_results, tracker_bbox_list)
        # cv2.imshow(winname="LETS GET ALL 'THEM FACES", mat=annotated_frame)
        
        # Write the frame to output video
        output_video.write(annotated_frame)

        # if cv2.waitKey(10) & 0xFF in [ord('q'), 27]:  # 27 is the ASCII code for the ESC key
        #     break

    # Release resources for this video
    cap.release()
    output_video.release()  # Release the video writer
    
    with open(log_file, 'a') as f:
        f.write(f"[{datetime.datetime.now().strftime('%H:%M')}] Finished processing video {i}\n")

# Close all windows at the end
cv2.destroyAllWindows()

with open(log_file, 'a') as f:
    f.write(f"\n[{datetime.datetime.now().strftime('%H:%M')}] Finished processing all videos!\n")
