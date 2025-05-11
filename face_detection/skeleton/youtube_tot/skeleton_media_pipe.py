# youtube video link: https://www.youtube.com/watch?v=06TE_U21FK4
# use mediapipe to detect skeletons in videos for detecting faces
# this modles run after the people detection model has detected any people in the video

import cv2 
import mediapipe as mp  # different compomemts in media pipe are often referred to as solutions
mp_drawing = mp.solutions.drawing_utils # drawing options for skeleton
mp_pose = mp.solutions.pose # pose estimation module in media pipe
import os 

#####  Get Video #####
# video_path = os.path.join('C:', 'Users', 'barak', 'OneDrive - Technion', 'Documents', 'GitHub', 'violence_detection_project', 'test_video.mp4')
video_path = os.path.join(os.path.dirname(__file__), 'hospital_exp.mp4')  # Navigate to the video file

#  TEMPORARY: using a given video, in the future this will be replaced with a video from the people detection model
cap = cv2.VideoCapture(video_path) # open video file

##### Setup Mediapipe #####
# setup mediapipe instance
MIN_DETECTION_CONFIDENCE = 0.5 # minimum confidence for detection
MIN_TRACKING_CONFIDENCE = 0.5 # minimum confidence for tracking
with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose: # initialize pose estimation model
    while cap.isOpened():
        ret, frame = cap.read() # read a frame from the video
        if not ret:
            print("End of video or cannot read the video file.")
            break # if no frame is read, break the loop
        
        # recolor the frame to RGB for mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # make the image writeable for mediapipe

        # make detections
        results = pose.process(image)
        # print(results.pose_landmarks) # print the pose landmarks for debugging

        # recolor image back to BGR for OpenCV
        image.flags.writeable = True # make the image writeable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # render detections on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image) # show the frame

        if cv2.waitKey(10) & 0xFF in [ord('q'), 27]:  # 27 is the ASCII code for the ESC key
            break

    cap.release() # release the video capture object
    cv2.destroyAllWindows() # close all windows