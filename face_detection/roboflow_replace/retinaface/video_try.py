from retinaface import RetinaFace
import cv2
import numpy as np
import os
from datetime import datetime

def create_log_file(output_folder):
    # Create a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(output_folder, f'face_detection_log_{timestamp}.txt')
    return open(log_filename, 'w')

def log_message(log_file, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(message)
    log_file.write(log_entry)
    log_file.flush()  # Ensure the message is written immediately

def process_video(video_path, output_path=None, log_file=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer if output path is specified
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 30 == 0:  # Show progress every 30 frames
            print(f"Processing frame {frame_count}")
        frame_count += 1
        
        # Detect faces in the frame
        resp = RetinaFace.detect_faces(frame)
        
        # Draw detections on the frame
        if isinstance(resp, dict):
            for face_key in resp:
                face_data = resp[face_key]
                facial_area = face_data['facial_area']
                landmarks = face_data['landmarks']
                
                # Draw facial area rectangle
                cv2.rectangle(frame, 
                            (facial_area[0], facial_area[1]), 
                            (facial_area[2], facial_area[3]), 
                            (0, 255, 0), 2)
                
                # Draw landmarks
                for landmark_key in landmarks:
                    point = landmarks[landmark_key]
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        
        # Show the frame
        # cv2.imshow('Face Detection', frame)
        
        # Write the frame if output is specified
        if out:
            out.write(frame)
            
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Process both videos
input_folder = r"c:\Users\barak\OneDrive - Technion\Documents\GitHub\violence_detection_project\face_detection\input\videos"
output_folder = r"c:\Users\barak\OneDrive - Technion\Documents\GitHub\violence_detection_project\face_detection\output\retinaface_videos"

# Create output folder explicitly
os.makedirs(output_folder, exist_ok=True)
print(f"Created output folder: {output_folder}")

# Create log file
log_file = create_log_file(output_folder)
log_message(log_file, "Starting face detection processing")
log_message(log_file, f"Input folder: {input_folder}")
log_message(log_file, f"Output folder: {output_folder}")

# Get all video files from input directory
video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
log_message(log_file, f"Found {len(video_files)} video files in input directory")

# List of videos that have already been processed (check output folder)
finished_videos = [f.replace('detected_', '') for f in os.listdir(output_folder) 
                  if f.startswith('detected_') and f.endswith(('.mp4', '.avi', '.mov'))]

for i, video in enumerate(video_files, start=1):
    if video in finished_videos:
        log_message(log_file, f"Skipping already processed video: {video}")
        continue
        
    input_path = os.path.join(input_folder, video)
    output_path = os.path.join(output_folder, f'detected_{video}')
    log_message(log_file, f"\nProcessing video {i}/{len(video_files)}: {video}")
    process_video(input_path, output_path, log_file)

log_message(log_file, "Face detection processing completed")
log_file.close()

