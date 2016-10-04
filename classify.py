import cv2
import argparse
import numpy as np
from sklearn.externals import joblib
from features import get_features_img
from nms import nms


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


def classifyImg(image_filename):
    image = cv2.imread(image_filename)
    windowSize = (128, 128)
    clf = joblib.load('model_original_data/model.pkl')

    params = {'nfeatures': 84, 'bow_size': 262, 'svm_gamma': -0.10575844, 'edgeThreshold': 88.41417693, 'svm_C': 2.83142991, 'sigma': 0.46683204, 'contrastThreshold': 0.0001}

    dect_params = {
        "nfeatures": params['nfeatures'],
        "contrastThreshold": params['contrastThreshold'],
        "edgeThreshold": params['edgeThreshold'],
        "sigma": params['sigma']
    }

    voca = np.load("model_original_data/vocabulary_model.npy")

    detections = []

    for (i, resized) in enumerate(pyramid(image, minSize=(256, 256))):
        # clone = resized.copy()
        newdetections = []
        n = 0
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

            prediction = clf.predict([hist])
            score = clf.decision_function([hist])

            # check = resized.copy()
            # cv2.rectangle(check, (x, y), (x + 128, y + 128), (0, 255, 0), 2)
            # cv2.imshow("Window", check)
            # cv2.waitKey(1)

            if prediction[0] == 0.:
                if i > 0:
                    detections.append([x / (0.75 ** i), y / (0.75 ** i), score[0], windowSize[0] / (0.75 ** i), windowSize[1] / (0.75 ** i)])
                else:
                    detections.append([x, y, score[0], windowSize[0], windowSize[1]])

                newdetections.append([x, y, score[0], windowSize[0], windowSize[1]])
                n = n + 1

        # for pred in newdetections:
        #    cv2.rectangle(clone, (pred[0], pred[1]), (pred[0] + pred[3], pred[1] + pred[4]), (0, 255, 0), 2)

        # cv2.imshow("Window res " + str(i), clone)
        print n

    clone = image.copy()
    preds = nms(detections)

    for pred in preds:
        cv2.rectangle(clone, (int(pred[0]), int(pred[1])), (int(pred[0]) + int(pred[3]), int(pred[1]) + int(pred[4])), (0, 255, 0), 2)

    cv2.imshow("Result", clone)

    cv2.waitKey(0)

    return preds

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    args = vars(ap.parse_args())
