import cv2
import numpy as np

class Tracker:
    NUM_OF_FRAMES_TO_TRACK = 30  # Maximum number of frames to track an object before re-detection
    OVERLAP_THRESHOLD = 0.2

    def __init__(self):
        # Initialize the GOTURN tracker
        self.tracker = cv2.TrackerCSRT_create()  # Replace CSRT with GOTURN
        self.frame_counter = 0  # Frame counter to track how many frames the tracker has been active
        self.bbox = None  # Bounding box (to be set during initialization)

        # Load the Haar cascade classifier once during initialization
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def initialize_tracker(self, frame, prediction):
        """
        Initializes the tracker with the first frame and bounding box.
        This method will also reset the frame counter.
        """
        self.bbox = self.convert_prediction_bbox(prediction)
        self.tracker.init(frame, self.bbox)  # Initialize the tracker with the frame and bounding box
        self.frame_counter = 0  # Reset frame counter
        return self  # Returning self allows method chaining or direct access to the object

    def update_tracker(self, frame):
        """
        Updates the tracker with a new frame and returns the updated bounding box.
        If the tracker exceeds the specified number of frames to track, it stops tracking.
        """
        self.frame_counter += 1  # Increment frame counter

        # If the tracker has been active for more than the specified number of frames, destroy it
        if self.frame_counter >= self.NUM_OF_FRAMES_TO_TRACK:
            self.destroy_tracker()
            return None

        ok, new_bbox = self.tracker.update(frame)  # Update the tracker with the new frame
        if not ok:  # Tracking failed
            self.destroy_tracker()
            return None

        # Draw the tracked bounding box on the frame
        p1 = (int(new_bbox[0]), int(new_bbox[1]))
        p2 = (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        self.bbox = new_bbox

        # Crop the tracked region from the frame
        x, y, w, h = map(int, new_bbox)
        tracked_region = frame[y:y + h, x:x + w]

        # Convert the cropped region to grayscale (required for Haar cascades)
        gray_tracked_region = cv2.cvtColor(tracked_region, cv2.COLOR_BGR2GRAY)

        # Detect objects (e.g., faces) in the cropped region
        faces = self.face_cascade.detectMultiScale(gray_tracked_region, # Detect faces in the tracked region
                                                    scaleFactor=1.1,    # Scale factor for the detection
                                                    minNeighbors=1)     # Minimum neighbors for a rectangle to be retained
        
        if len(faces) == 0:  # No faces detected in the tracked region
            print(f"No faces detected in bbox {self.bbox}")

        # Draw rectangles around detected objects on the original frame
        else:
            for (fx, fy, fw, fh) in faces:
                # Adjust face coordinates relative to the original frame
                face_x = x + fx
                face_y = y + fy
                face_w = fw
                face_h = fh

            # Draw the rectangle on the original frame
            cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)

        return True  # Return updated bounding box if tracking is successful

    def destroy_tracker(self):
        """
        Destroys the tracker and frees up the resources.
        """
        del self.tracker  # Explicitly delete the tracker object to free resources
        self.tracker = None  # Set the tracker reference to None

    def check_bbox_overlap(self, prediction):
        pred_bbox = self.convert_prediction_bbox(prediction)
        x_track, y_track, w_track, h_track = self.bbox
        x_pred, y_pred, w_pred, h_pred = pred_bbox

        # Calculate the intersection
        inter_left = max(x_track, x_pred)
        inter_top = max(y_track, y_pred)
        inter_right = min(x_track + w_track, x_pred + w_pred)
        inter_bottom = min(y_track + h_track, y_pred + h_pred)

        # Check if there is an intersection
        if inter_right < inter_left or inter_bottom < inter_top:
            return 0.0

        # Calculate intersection area
        intersection_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Calculate union area
        track_area = w_track * h_track
        pred_area = w_pred * h_pred
        union_area = track_area + pred_area - intersection_area

        # Calculate IoU (Intersection over Union)
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    def convert_prediction_bbox(self, prediction):
        # Extract the bounding box from the prediction (center x, center y, width, height)
        x_center, y_center, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        
        # Convert to top-left corner (x, y) and width and height for OpenCV tracker
        x = int(x_center - width / 2)
        y = int(y_center - height / 2)
        w = int(width)
        h = int(height)
        
        converted_bbox = (x, y, w, h)  # Bounding box in (x, y, width, height)
        return converted_bbox