from roboflow import Roboflow
import supervision as sv
import cv2
from tracker import Tracker  # Import the Tracker class

DETECTION_CONFIDENCE_THRESHOLD = 50  # his line sets the initial confidence threshold for the model's predictions (for all predictions)
ALLOWED_OVERLAP = 20     # A value of x means that if two bounding boxes overlap by more than x%, the one with the lower confidence score will be removed.
PERSON_DETECTION_THRESHOLD = 0.1  # Confidence threshold for detecting a person
CREATE_NEW_TRACKER_THRESHOLD = 0.2 # If IoU with any existing tracker is above this, no new tracker is created

# Initialize Roboflow with API key
rf = Roboflow(api_key="NeHIyTiEmZ0e7Y5WVLb9")


# Access specific project and model version
project = rf.workspace().project("people-detection-o4rdr")
model = project.version(8).model

# Open the video file
video_path = "test_video.mp4"
# video_path = "/home/p24w01/data/sample_video_from_hospitel_crop.mp4"    # a cropped videos from hospital sample
# video_path = "/home/p24w01/data/dataset/NTU_fight0169_run_1.mp4"  # videos from dataset
cap = cv2.VideoCapture(video_path)

# Initialize annotators for labeling and bounding boxes
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

# Get video properties for output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


def open_new_trackers(frame, prediction):
    # Iterate through existing trackers
    for tracker in tracker_list:
        # Calculate IoU between the prediction and the tracker's bounding box
        iou = tracker.check_bbox_overlap(prediction)
        
        # If IoU exceeds the threshold, don't create a new tracker
        if iou > CREATE_NEW_TRACKER_THRESHOLD:
            # Significant overlap with an existing tracker, so we don't need a new one
            return

    # If we've reached this point, no significant overlap was found
    # print("Opening new tracker")
    
    # Create a new tracker instance
    new_tracker = Tracker()
    
    # Initialize the new tracker with the current frame and prediction
    new_tracker.initialize_tracker(frame, prediction)
    
    # Add the new tracker to our list of active trackers
    tracker_list.append(new_tracker)


def update_trackers(frame):
    # If there are no active trackers, return immediately
    if len(tracker_list) == 0:
        return

    # Iterate over a copy of the tracker list to safely modify it during iteration
    for i,tracker in enumerate(tracker_list[:]):
        # Update each tracker with the current frame
        is_active = tracker.update_tracker(frame)

        # If the tracker is no longer active (returns None), remove it from the list
        if is_active is None:
            tracker_list.remove(tracker)  # Remove the tracker from the list
            print(f"Tracker {i} removed due to inactivity")

    # Note: After this function, tracker_list will only contain active trackers


tracker_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    update_trackers(frame)

    result = model.predict(frame, confidence=DETECTION_CONFIDENCE_THRESHOLD, overlap=ALLOWED_OVERLAP).json()
    
    # Extract class labels and confidence levels from predictions
    labels = [
        f"{item['class']} ({item['confidence']:.2f})"
        for item in result["predictions"]
    ]

    # add predictions to show in the video
    detections=sv.Detections.from_inference(result)
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # go through every person prediction and update trackers list
    for prediction in result["predictions"]:
        if prediction['class'].lower() == 'person' and prediction['confidence'] >= PERSON_DETECTION_THRESHOLD:
            open_new_trackers(frame, prediction)

    # display result
    cv2.imshow("Tracking", frame)

    # exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

# Release resources
cap.release()
# out.release()
