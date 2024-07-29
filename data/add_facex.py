import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

face_images = []
names = []

i = 0

name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(face_images) <= 100 and i % 10 == 0:
            face_images.append(resized_img)
            names.append(name)
        i += 1
        cv2.putText(frame, str(len(face_images)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(face_images) == 100:
        break

video.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_images)
names_data = np.asarray(names)

face_data = face_data.reshape(len(face_images), -1)

if 'names.pkl' not in os.listdir('data/'):
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names_data, f)
else:
    with open('data/names.pkl', 'rb') as f:
        existing_names = pickle.load(f)
    names_data = np.append(existing_names, names_data)
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names_data, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        existing_faces = pickle.load(f)
    face_data = np.append(existing_faces, face_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
