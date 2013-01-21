import cv2
import cv2.cv as cv
#3333333333333333333333333333333333333333333333333333333
class FaceDetector(object):
    def __init__(self, cascade_file="/home/w/tmp/openCV/OpenCV-2.4.2/data/haarcascades/haarcascade_frontalface_alt.xml"):
        self.cascade = cv2.CascadeClassifier(cascade_file)

    def detect(self, img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return self.detect(gray, self.cascade)
#3333333333333333333333333333333333333333333333333333333
