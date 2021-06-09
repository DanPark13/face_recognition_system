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
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    small_frame = small_frame[:, :, ::-1]

    # Get the face location within the frame of the webcam
    face_frame = face_recognition.face_locations(small_frame)
    # Get the encoders of the face within the frame of the webcam
    encoder_frame = face_recognition.face_encodings(small_frame, face_frame)

    # Grab the encoded face and face location from the frame
    for encoded_face, face_location in zip(encoder_frame, face_frame):
        best_matches = face_recognition.compare_faces(encoded_images_list, encoded_face)
        face_accuracy = face_recognition.face_distance(encoded_images_list, encoded_face)
        print(face_accuracy)
        face_match_index = np.argmin(face_accuracy)

        if best_matches[face_match_index]:
            name = all_names[face_match_index].upper()
            print(name)

    # Show the webcam frame onto the screen
    cv2.imshow("Webcam", frame)

    # Hit "q" on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
