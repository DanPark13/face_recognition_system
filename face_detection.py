"""
Facial Detection

Detects the faces on a picture
"""

# Import Libraries
import cv2
import face_recognition

'''
Load images for testing
'''
# Load training image
training_image_trump = face_recognition.load_image_file("training_images/DonaldTrump_DT5893.jpg")
# Convert training image to RGB (red-green-blue)
training_image_trump = cv2.cvtColor(training_image_trump, cv2.COLOR_BGR2RGB)
# Load testing image
testing_image_trump = face_recognition.load_image_file("testing_images/donald_trump_test.jpg")
# Convert testing image to RGB (red-green-blue)
testing_image_trump = cv2.cvtColor(testing_image_trump, cv2.COLOR_BGR2RGB)

'''
Detect Training Face
'''
# Identify the location of the Face
trump_train_face_location = face_recognition.face_locations(training_image_trump)[0]
# Encodes the location of the Face (used for comparison)
encode_trump_train_face = face_recognition.face_encodings(training_image_trump)[0]
# Draw a rectangle around the Face using the Face Location
cv2.rectangle(training_image_trump, (trump_train_face_location[3], trump_train_face_location[0]),
              (trump_train_face_location[1], trump_train_face_location[2]), (255, 0, 255), 2)

print(f"Training Trump Face Location: ", trump_train_face_location)

'''
Detect Testing Face
'''
# Identify the location of the Face
trump_test_face_location = face_recognition.face_locations(testing_image_trump)[0]
# Encodes the location of the Face (used for comparison)
encode_trump_test_face = face_recognition.face_encodings(testing_image_trump)[0]
# Draw a rectangle around the Face using the Face Location
cv2.rectangle(testing_image_trump, (trump_test_face_location[3], trump_test_face_location[0]),
              (trump_test_face_location[1], trump_test_face_location[2]), (255, 0, 255), 2)

print(f"Testing Trump Face Location: {trump_test_face_location}")

'''
Compare the two faces
'''
comparison_results = face_recognition.compare_faces([encode_trump_train_face], encode_trump_test_face)
face_distance = face_recognition.face_distance([encode_trump_train_face],encode_trump_test_face) # Lower number = better
print(comparison_results, face_distance)

'''
Show Testing Images
'''
cv2.imshow("Donald Trump Official", training_image_trump)
cv2.imshow("Donald Trump Test", testing_image_trump)
cv2.waitKey(0)
