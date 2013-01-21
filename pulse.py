#! /usr/bin/env python
import sys
import cv2.cv as cv
import cv2
import numpy as np
import numpy.fft as fourier
import matplotlib.pyplot as plt
from facedetect import FaceDetector
from clint.textui import puts, progress, colored
from helpers import  inprint

__metaclass__ = type


def getsize(img):
    h, w = img.shape[:2]
    return w, h

def fft(discrete, filtr=None):
    result = fourier.fft(discrete) 
    result = map(abs, result)
    if filtr is None:
        return result

    if callable(filtr):
        return filtr(result)

    return apply_filter(result, filtr)


def apply_filter(fun, filter_fun):
    return [value * filter_value for (value, filter_value) in zip(fun, filter_fun)]


class Pyramid:
    def __init__(self, frame, channels=None, laplacian=True):
        if channels is None:
            channels = 0

        X, Y, channels = frame.shape
        for x in range(X):
            for y in range(Y):
                frame[x,y,0] = 0
                frame[x,y,1] = 0

        #frame = frame[:, :, channels]

        if laplacian:
            self.levels = self.build_laplacian_pyramid(frame)
        else:
            self.levels = self.build_gaussian_pyramid(frame)

    #taken from samples/python/build_lappyr
    def build_laplacian_pyramid(self, img, leveln=4, dtype=np.int16):
        #img = dtype(img)
        levels = []
        for i in xrange(leveln-1):
            next_img = cv2.pyrDown(img)
            img1 = cv2.pyrUp(next_img, dstsize=getsize(img))
            levels.append(img-img1)
            img = next_img
        levels.append(img)
        return levels

    def build_gaussian_pyramid(self, img, leveln=4, dtype=np.int16):
        levels = []
        for i in range(leveln):
            next_img = cv2.pyrDown(img)
            levels.append(next_img)
            img = next_img
        return levels


def expected_pulse_spectrum(freq):
    values = (0.5,0), (0.6,0.5), (0.9, 0.9), (1,1), (1.2,0.9), (1.6, 0.7),(1.9,0.5), (2,0)

    if freq < 0.5:
        return 0
    if freq > 2:
        return 0

    for (x1, y1), (x2, y2) in zip(values, values[1:]):
        if x1 <= freq <= x2:
            return y1 + ((y2 - y1)/(x2-x1)) * (freq - x1)



#--------------------------------------------------------------------------------
def init_camera():
    if len(sys.argv) == 1:
        capture = cv2.VideoCapture(0)
    elif len(sys.argv) == 2:
        print 'using file', sys.argv[-1]
        capture = cv2.VideoCapture(sys.argv[1])

    if not capture:
        print "Could not initialize capturing..."
        sys.exit(-1)

    return capture


class Frame:
    def __init__(self, data, ticks):
        self.data = data
        self.ticks = ticks
        self.red = 0
        self.green = 0
        self.blue = 0

    def to_img(self):
        return cv.fromarray(self.data)


class Application:

    APPLICATION_NAME = 'Pulse detector'

    def __init__(self):
        self.video_in = init_camera()
        self.frames_number = 0
        self.reds = []
        self.greens = []
        self.blues = []
        self.frames = []
        self.processor = Magnification()

    def process(self, frame):
        self.processor.real_time_process(frame)

    def main_loop(self):
        #skip few first frames
        for i in range(5):
            flag, frame = self.video_in.read()

        self.t0 = cv2.getTickCount()
        while True:
            flag, frame = self.video_in.read()
            if not flag:
                continue

            self.frames_number += 1
            frame = Frame(frame, cv2.getTickCount())

            self.process(frame)
            #cv.ShowImage(self.APPLICATION_NAME, frame.to_img())
            self.frames.append(frame)
            t = cv2.getTickCount()
            inprint('fps %.1f, frames %d \r'% ((1.0/((t - self.t0) / cv2.getTickFrequency() )), self.frames_number))
            self.t0 = t

            if cv.WaitKey(10) is not -1:
                self.tps = cv2.getTickFrequency()
                break

    def run(self):
        cv.NamedWindow(self.APPLICATION_NAME)

        t0 = cv2.getTickCount()
        self.main_loop()
        print 
        puts(colored.green('running time %.1f'% ((cv2.getTickCount() - t0) / cv2.getTickFrequency())))

        cv.DestroyAllWindows()

        puts(colored.green('collected %d frames' % len(self.frames)))
        print
        frames = self.frames[-64:]
        print 'processing %d frames' % len(frames)
        self.processor.frames = frames 
        self.processor.process_frames()
        self.processor.get_result(self.tps)
        self.processor.show(self.tps)


class Magnification:
    def __init__(self, frames=None):
        if frames is None:
            frames = []
        self.frames = frames
        self.facedetector = FaceDetector()

        self.color = (0, 255, 0)

    def real_time_process(self, frame):
        rects = self.facedetector.detect_face(frame.data)
        cpy = frame.data.copy()
        for x1, y1, x2, y2 in rects:
            width = abs(x2 - x1)
            height = abs(y1 - y2)
            cv2.rectangle(cpy, (x1 + int(width * 0.2), y1 ) , 
                               (x2 - int(width * 0.2), y2- int(height * 0.2)), 
                               self.color, 2)

        img = cv.fromarray(cpy)
        cv.ShowImage('face', img)

        return
        frame = frame.data
        X,Y, channels = frame.shape
        for x in range(X):
            for y in range(Y):
                frame[x,y,1] = 0
                frame[x,y,2] = 0
        img = cv.fromarray(frame)
        cv.ShowImage('red channel', img)

    def get_face(self, frame, rects):
        for x1, y1, x2, y2 in rects:
            width = abs(x1 - x2)
            height = abs(y1 - y2)
            rect = ( x1 + int(width * 0.2), y1 , x2 - int(width * 0.2), y2- int(height * 0.2))

            return rect

    def process_frame(self, frame):
        rects = self.facedetector.detect_face(frame)
        if rects is None or len(rects) == 0:
            raise AttributeError()
        x1, y1, x2, y2= self.get_face(frame, rects)

        subframe = frame[x1:x2, y1:y2, :]

        pyramid = Pyramid(subframe, laplacian=False)
        pyramid.reds = []
        for level in pyramid.levels:
            red = 0
            X, Y, channels = level.shape

            red = 0

            for x in range(x1, x2):
                for y in range(y1, y2):
                    red += frame[x, y, 2]

            red /= float(X*Y)
            pyramid.reds.append(red)

        return pyramid

    def process_frames(self):
        frames = self.frames
        processed = []
        for frame in progress.bar(frames):
            try:
                p = self.process_frame(frame.data)
                processed.append(p)
            except AttributeError:
                puts(colored.red('face not detected'))

        self.processed = processed

    def sample_expected(self, freq_step, length):
        return [expected_pulse_spectrum(freq_step * i) for i in range(length)] 

    def show(self, tps):
        sampling_rate = len(self.frames) / (float(self.frames[-1].ticks - self.frames[0].ticks) /tps)
        print 'sampling_rate', sampling_rate
        max_fq = 0.5 * sampling_rate

        def filtr(data):
            hz_step = max_fq / len(data)
            self.step = hz_step
            fi = self.sample_expected(hz_step, len(data))
            return apply_filter(data, fi)

        plt.axhline()
        plt.plot(fft([item.reds[0] for item in self.processed], filtr), 'r')
        plt.plot(fft([item.reds[1] for item in self.processed], filtr), 'y')
        plt.plot(fft([item.reds[2] for item in self.processed], filtr), 'b')
        plt.plot(fft([item.reds[3] for item in self.processed], filtr), 'g')
        plt.axvline(1/self.step)

        print 'freq step', self.step
        plt.show()
     
        #plt.axhline()
        #plt.plot(self.sample_expected(self.step, 10))
        #plt.show()

    def sampling_rate(self, tps):
        return len(self.frames) / (float(self.frames[-1].ticks - self.frames[0].ticks) /tps)

    def get_result(self, tps):
        sampling_rate = self.sampling_rate(tps)
        max_fq = 0.5 * sampling_rate

        def filtr(data):
            hz_step = max_fq / len(data)
            self.step = hz_step
            fi = self.sample_expected(hz_step, len(data))
            return apply_filter(data, fi)

        results = []
        for i in range(4):
            results.append(fft([item.reds[i] for item in self.processed], filtr))
        indexes = map(max_on_index, results)
        print 'indexes', indexes
        avg = sum(indexes) / len(indexes)
        value = int(avg * self.step * 60)

        print
        puts(colored.red(u' \u2764 '), newline=False) 
        puts(colored.yellow('%d BPM' % value ))
        print

def max_on_index(li):
    m = li[0]
    mindex = 0
    for i, item in enumerate(li):
        if item > m:
            m = item
            mindex = i
    return mindex


if __name__ == '__main__':
    Application().run()

