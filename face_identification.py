"""
Face Identification

Detects the faces of a webcame and identifies it with a picture
"""

# Import Libraries
import cv2
import numpy as np
import face_recognition
import os

"""
Import images
"""
image_folder_path = "training_images"
all_images = []
all_names = []
image_files = os.listdir(image_folder_path)

"""
Get rid of '.jpg' extension name
"""
for name in image_files:
    current_image = cv2.imread(f'{image_folder_path}/{name}')
    all_images.append(current_image)
    all_names.append(os.path.splitext(name)[0])
print(all_names)


def find_encodings(images):
    """
    Encode all the images
    """
    encoded_image_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoded_image = face_recognition.face_encodings(image)[0]
        encoded_image_list.append(encoded_image)
    return encoded_image_list


encoded_images_list = find_encodings(all_images)
print("Encoding Complete")

"""
Webcam Functionality
"""
# Get reference to default webcam
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Starting Webcam...")

while True:
    # Grab frame of video
    ret, frame = video_capture.read()

    # Show the webcam frame onto the screen
    cv2.imshow("Webcam", frame)

    # Hit "q" on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
