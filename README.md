# Face_Recognition

from matplotlib import pyplot as plt
import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#This code is using a live feed from webcam to detect the faces

video_capture = cv2.VideoCapture(0)
def detect_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:

#Will show the result of the video capture
    result, video_frame = video_capture.read()  
    if result is False:
        break
#Loop will terminate if video frame is not found

#This line will apply the frame capture
    faces = detect_box(
        video_frame
    )  

#This line will display the video capture status under "My Face Detection Project"
    cv2.imshow(
        "My Face Detection Project", video_frame
    )  

    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

plt.figure(figsize=(20,20))
plt.imshow(faces)
plt.axis("off")
plt.show()

video_capture.release()
cv2.destroyAllWindows()
