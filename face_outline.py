from PIL import Image, ImageDraw
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("testing_images/joe_biden_test.jpg")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Put the image into an array
pil_image = Image.fromarray(image)

# For each landmark
for face_landmarks in face_landmarks_list:
    # Initialize the image to draw
    image = ImageDraw.Draw(pil_image, 'RGBA')
    print(face_landmarks.keys())
    # For every feature, draw a white line on the feature
    for feature in face_landmarks.keys():
        image.line(face_landmarks[feature], fill=(255, 255, 255))
    # Show the image
    pil_image.show()
