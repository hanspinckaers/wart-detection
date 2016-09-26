import cv2
import argparse
import time
import numpy as np
from sklearn.externals import joblib
from features import get_features_img


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
clf = joblib.load('model.pkl')

params = {'nfeatures': 60, 'bow_size': 1013, 'svm_gamma': 1.7507582, 'edgeThreshold': 50., 'svm_C': 0.47862167, 'sigma': 2.25901868, 'contrastThreshold': 0.001}
dect_params = {
    "nfeatures": params['nfeatures'],
    "contrastThreshold": params['contrastThreshold'],
    "edgeThreshold": params['edgeThreshold'],
    "sigma": params['sigma']
}
voca = np.load("vocabulary_model.npy")

for (i, resized) in enumerate(pyramid(image, minSize=windowSize)):
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=windowSize):
        if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
            continue

        descs = get_features_img(window, 'SIFT', 'SIFT', dect_params)

        hist = np.zeros(len(voca))

        for desc in descs:
            match = np.sum(np.square(np.abs(voca - desc)),1).argmin()
            hist[match] += 1

        hist /= len(descs)
        hist = hist.astype(np.float32)

        prediction = clf.predict(hist)

        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + windowSize[0], y + windowSize[1]), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
