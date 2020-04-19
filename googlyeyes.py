import cv2
import numpy as np

# faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")
# cap = cv2.VideoCapture(0)
# while True:
#        ret, frame = cap.read()
#        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#        face_rects = faceCascade.detectMultiScale(gray, 1.3, 5)
#        for (x, y, w, h) in face_rects:
#            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
#        cv2.imshow("FD", frame)
#        v = cv2.waitKey(20)
#        c = chr(v & 0xFF)
#        if c == 'q':
#            break
# cap.release()

face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye1.xml")
if face_cascade.empty():
    raise IOError(' Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError(' Unable to load the eye cascade classifier xml file')
cap = cv2.VideoCapture(0)
ds_factor = 0.5
while True:
    ret, img = cap.read()
    img2 = img[:, ::-1, :]
    img2 = cv2.resize(img2, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2.4, 1)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img2[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        while eyes.__len__()>2: #Finds the two best eyes and only shows circles for those
            least = 0
            for l in range(1, eyes.__len__()):
                if (eyes[l][3] < eyes[least][3]):
                    least = l
            eyes = np.delete(eyes, (least), axis=0)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            if radius > 10:
                cv2.circle(roi_color, center, radius, color, thickness)
    cv2.imshow('Eye Detector', img2)
    x = cv2.waitKey(10)
    userChar = chr(x & 0xFF)
    if userChar == 'q':
        break

eyeImg = cv2.imread("GooglyEye.png")
while True:
    ret, img = cap.read()
    img2 = img[:, ::-1, :]
    img2 = cv2.resize(img2, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img2[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        while eyes.__len__()>2: #Finds the two best eyes and only shows circles for those
            least = 0
            for l in range(1, eyes.__len__()):
                if (eyes[l][3] < eyes[least][3]):
                    least = l
            eyes = np.delete(eyes, (least), axis=0)
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            #center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
            radius = int(0.3 * (w_eye + h_eye))+2
            newImg = cv2.resize(eyeImg, (radius, radius))
            x_offset = int(x_eye + 0.2 * w_eye+x)
            y_offset = int(y_eye + 0.2 * h_eye+y)
            img2[y_offset:y_offset + newImg.shape[0], x_offset:x_offset + newImg.shape[1]] = newImg
    cv2.imshow('Eye Detector', img2)
    x = cv2.waitKey(10)
    userChar = chr(x & 0xFF)
    if userChar == 'q':
        break

cap.release()
cv2.destroyAllWindows()
