import cv2
import os


def frames_img(filename):
    cam = cv2.VideoCapture('example.mp4')


    currentFrame = 0

    while ():
        ret, frame = cam.read()

        name = '...' + str(currentFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)
        currentFrame += 1

        cam.release()
        cv2.destroyAllWindows()