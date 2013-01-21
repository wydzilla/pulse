Attempt to detect human pulse from video stream.
=========

Summary of algo:
for every frame in stream:
- Create laplacian pyramid. And for every level:
-- detect interior of face in video frame
-- calculate sum of red channel over detected range

In this way, funtion over time is created.
Apply fft to obtain frequency image of function.
Filter for most probable human pulse.
Detect highest peak in result data - it is interpreted as measured pulse.
