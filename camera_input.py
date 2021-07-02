import numpy as np
import cv2 as cv
import time
import torch
import matplotlib.pyplot as plt

def get_camera_device(device):
    return cv.VideoCapture(device)

def read_frame(camera, img_size=None, grey=None, dtype='np'):

    ret, frame = camera.read()

    if grey:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if img_size is not None:
        frame = cv.resize(frame, dsize=(28, 28))

    if dtype == 'pt':
        return torch.tensor(frame)
    elif dtype=='np':
        return frame

def kill_camera(camera):
    camera.release()

camera = get_camera_device(0)

frame1 = read_frame(camera, grey=True, dtype='pt')
print('taken')

time.sleep(2)

frame2 = read_frame(camera, grey=True, dtype='pt')
print('taken')

output = torch.cat([frame1, frame2], dim=1)

plt.imshow(output, cmap='gray')
plt.show()

kill_camera(camera)


