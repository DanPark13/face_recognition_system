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

encoded_images = find_encodings(all_images)
print("Encoding Complete")
