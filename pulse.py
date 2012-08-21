#! /usr/bin/env python
import sys
import cv2.cv as cv
import cv2
import numpy as np


def getsize(img):
    h, w = img.shape[:2]
    return w, h

#initialize camera
def init_camera():
    if len(sys.argv) == 1:
        capture = cv2.VideoCapture(0)
        #capture = cv.CreateCameraCapture(0)
    #elif len(sys.argv) == 2 and sys.argv[1].isdigit():
        #capture = cv.CreateCameraCapture(int(sys.argv[1]))
    #elif len(sys.argv) == 2:
        #capture = cv.CreateFileCapture(sys.argv[1])

    if not capture:
        print "Could not initialize capturing..."
        sys.exit(-1)

    return capture


#build laplacian pyramid


#build pixel functions
def build_funs():
    pass

#fft
#(optionaly show functions after fft)
def fft(discrete):
    cv2.DFT(discrete, discrete, cv.CV_DXT_FORWARD, nonzeros=1)


#filter out
#(optionaly show functions after filtering)
def apply_filter(fun, filter_fun):
    #read how much are samples separated
    return [value * filter_value for (value, filter_value) in zip(fun, filter_fun)]


#find rate
#(optionaly mark peak which was selected as most probable)
def find_rate():
    pass

import random
class Detector(object):
    def __init__(self):
        self.queque = []
        self.ready = False
        self.rate = None

    def push(self, frame):
        pyramid = Pyramid(frame)
        self.queque.append(pyramid)
        if len(self.queque) > 20:
            self.queque.pop(0)

        if len(self.queque) == 20:
            self.ready = True
            self.process_queue()

    def process_queue(self):
        self.rate = random.randint(30,220)




class Pyramid(object):
    def __init__(self, frame):
        self.levels =  self.build_pyramid(frame)

    #taken from samples/python/build_lappyr
    def build_pyramid(self, img, leveln=6, dtype=np.int16):
        #img = dtype(img)
        levels = []
        for i in xrange(leveln-1):
            next_img = cv2.pyrDown(img)
            img1 = cv2.pyrUp(next_img, dstsize=getsize(img))
            levels.append(img-img1)
            img = next_img
        levels.append(img)
        return levels


if __name__ == '__main__':
    app_name = 'test app'
    video_in = init_camera()

    detector = Detector()

    cv.NamedWindow(app_name)

    while True:
        flag, frame = video_in.read()
        if not flag:
            continue

        cv.ShowImage(app_name, cv.fromarray(frame))
        #cv2.imshow('frame', cv.fromarray(frame))

        detector.push(frame)
        if detector.ready:
            rate = detector.rate
            print 'Rate', rate

        if cv.WaitKey(10) is not -1:
            break

    cv.DestroyAllWindows()


