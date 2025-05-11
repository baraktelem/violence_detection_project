# youtube video link: https://www.youtube.com/watch?v=06TE_U21FK4
# use mediapipe to detect skeletons in videos for detecting faces
# this modles run after the people detection model has detected any people in the video

import cv2 
import mediapipe as mp  # different compomemts in media pipe are often referred to as solutions
mp_drawing = mp.solutions.drawing_utils # drawing options for skeleton
mp_pose = mp.solutions.pose # pose estimation module in media pipe

def detect_skeleton_poses(frame, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """
    Detects and processes the skeleton on the given frame using MediaPipe Pose.
    
    Parameters:
        frame: The input video frame.
        min_detection_confidence (float): Minimum confidence for detection.
        min_tracking_confidence (float): Minimum confidence for tracking.
        
    Returns:
        The results of the pose detection for further processing.
    """
    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
        # Convert the frame to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Make the image writeable for MediaPipe

        # Make detections
        results = pose.process(image)

        # Recolor image back to BGR for OpenCV
        image.flags.writeable = True  # Make the image writeable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections on the image
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # return image  # Return the frame with the skeleton drawn on it
    return results  # Return the results for further processing if needed

def draw_skeleton_landmarks(frame, results):
    """
    Draws the skeleton landmarks on the given frame.
    
    Parameters:
        frame: The input video frame.
        results: The results of the pose detection.
        
    Returns:
        The frame with the skeleton drawn on it.
    """
    if results.pose_landmarks:
        # Define custom specs
        face_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green for face
        body_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)  # Red for body

        # Create a custom landmark drawing function
        def custom_landmark_drawing(landmark_list, landmark_drawing_specs=None):
            if landmark_drawing_specs is None:
                landmark_drawing_specs = {}
            
            for idx, landmark in enumerate(landmark_list.landmark):
                current_spec = face_drawing_spec if idx <= 10 else body_drawing_spec
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 
                          current_spec.circle_radius, 
                          current_spec.color, 
                          current_spec.thickness)

        # Draw connections first
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=None,  # We'll draw landmarks manually
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

        # Draw landmarks with custom colors
        custom_landmark_drawing(results.pose_landmarks)

    return frame