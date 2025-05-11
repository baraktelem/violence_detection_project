from retinaface import RetinaFace
import matplotlib.pyplot as plt

## github link https://github.com/serengil/retinaface/blob/master/README.md

folder_path = r"c:\Users\barak\OneDrive - Technion\Documents\GitHub\violence_detection_project\face_detection\input\images"
hospital_1 = r"\hospital_1_jpg.jpg"
hospital_2 = r"\hospital_2_jpg.jpg"


## Face Detection
import cv2
import numpy as np

img_path = r"c:\Users\barak\OneDrive - Technion\Documents\GitHub\violence_detection_project\face_detection\input\images\hospital_2_jpg.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Detect faces and get their locations
resp = RetinaFace.detect_faces(img_path)
print("Detection response:", resp)

# Draw detections on the image
for face_key in resp:
    face_data = resp[face_key]
    facial_area = face_data['facial_area']
    landmarks = face_data['landmarks']
    
    # Draw facial area rectangle
    cv2.rectangle(img, 
                 (facial_area[0], facial_area[1]), 
                 (facial_area[2], facial_area[3]), 
                 (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark_key in landmarks:
        point = landmarks[landmark_key]
        cv2.circle(img, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

# Show the image with detections
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.title('Detected Faces and Landmarks')
plt.show()



## Alignment 

# faces = RetinaFace.extract_faces(img_path = "img.jpg", align = True)
# for face in faces:
#   plt.imshow(face)
#   plt.show()


## Face Recognition

# from deepface import DeepFace
# obj = DeepFace.verify(folder_path + hospital_1, folder_path + hospital_2,
#           model_name = 'ArcFace', detector_backend = 'retinaface', enforce_detection=False)
# print(obj["verified"])