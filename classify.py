import cv2
import argparse
import time


# make a pyramid of images
# these functions are inspired by http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def pyramid(image, scale=0.75, minSize=(30, 30)):
    yield image
    while True:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        if image.shape[0] >= minSize[0] and image.shape[1] >= minSize[1]:
            yield image
        else:
            break


def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
windowSize = (128, 128)

for (i, resized) in enumerate(pyramid(image, minSize=windowSize)):
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=windowSize):
        if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
            continue

        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + 128, y + 128), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
