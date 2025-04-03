import cv2

class Tracker:
    NUM_OF_FRAMES_TO_TRACK = 30  # Maximum number of frames to track an object before re-detection
    OVERLAP_THRESHOLD = 0.2


    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()  # Tracker initialization (You can replace with other trackers)
        self.frame_counter = 0  # Frame counter to track how many frames the tracker has been active
        self.bbox = None  # Bounding box (to be set during initialization)


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
        if ok:
            p1 = (int(new_bbox[0]), int(new_bbox[1]))
            p2 = (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            self.bbox = new_bbox
            # prediction_match = self.check_bbox_overlap(prediction)
            return True # Return updated bounding box if tracking is successful
        else:
            self.destroy_tracker()
            return None  # If tracking failed, return None


    def destroy_tracker(self):
        """
        Destroys the tracker and frees up the resources.
        """
        print("closing tracker")
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