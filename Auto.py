import cv2
import numpy as np
from face_recognition import face_distance, face_encodings, compare_faces, face_locations
import os
from datetime import datetime

model = "cnn"
num_jitters = 10
distance_threshold = 0.6

path = 'Faces'
images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(imgs):
    encodeList = []
    for image in imgs:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_encodings(image, num_jitters=num_jitters, model=model)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(names):
    with open('Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if names not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{names}, {dtString}')

now = datetime.now()
dtString = now.strftime('%H:%M:%S')

unknown = "UNKNOWN"

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_locations(imgS)
    encodesCurFrame = face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matching = compare_faces(encodeListKnown, encodeFace)
        faceDis = face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matching[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 + 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, dtString, (x1 + 6, y2 + 65), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 + 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, unknown, (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, dtString, (x1 + 6, y2 + 65), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            markAttendence(unknown)

    cv2.imshow('Webcam - Press Q to Quit', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break