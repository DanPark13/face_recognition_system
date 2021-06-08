import cv2
import numpy as np
import face_recognition

'''
Load images for testing
'''
# Load training image
training_image_trump = face_recognition.load_image_file("training_images/donald_trump_official.jpg")
# Convert training image to RGB (red-green-blue)
training_image_trump = cv2.cvtColor(training_image_trump, cv2.COLOR_BGR2RGB)
# Load testing image
testing_image_trump = face_recognition.load_image_file("testing_images/donald_trump_test.jpg")
# Convert testing image to RGB (red-green-blue)
testing_image_trump = cv2.cvtColor(testing_image_trump, cv2.COLOR_BGR2RGB)

'''
Show Trump Images
'''
cv2.imshow("Donald Trump Official", training_image_trump)
cv2.imshow("Donald Trump Test", testing_image_trump)
cv2.waitKey(0)
