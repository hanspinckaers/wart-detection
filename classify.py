import cv2
import argparse
import numpy as np
from sklearn.externals import joblib
from features import get_features_img
from nms import nms_heatmap
import time
import matplotlib.pyplot as plt
# import pudb


# make a pyramid of images
# these functions are inspired by http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def pyramid(image, scale=0.75, min_size=(30, 30)):
    yield image
    while True:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        if image.shape[0] >= min_size[0] and image.shape[1] >= min_size[1]:
            yield image
        else:
            break


def sliding_window(image, step_size, window_size):
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def classify_img(image_filename, threshold=10, window_size=(128, 128), min_size=(256,256), step_size=32, pyr_scale=0.75):
    image = cv2.imread(image_filename)
    clf = joblib.load('final_model/model.pkl')
    voca = np.load("final_model/vocabulary_model.npy")

    overall_start_time = time.time()

    params = {'nfeatures': 84, 'bow_size': 262, 'svm_gamma': -0.105758443512, 'edgeThreshold': 88.4141769336, 'svm_C': 2.83142990871, 'sigma': 0.466832044073, 'contrastThreshold': 0.0001}

    dect_params = {
        "nfeatures": params['nfeatures'],
        "contrastThreshold": params['contrastThreshold'],
        "edgeThreshold": params['edgeThreshold'],
        "sigma": params['sigma']
    }

    detections = []

    for (i, resized) in enumerate(pyramid(image, min_size=min_size, scale=pyr_scale)):
        newdetections = []
        n = 0
        for (x, y, window) in sliding_window(resized, step_size=32, window_size=window_size):
            if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
                continue

            descs = get_features_img(window, 'SIFT', 'SIFT', dect_params)

            hist = np.zeros(len(voca))

            for desc in descs:
                match = np.sum(np.square(np.abs(voca - desc)),1).argmin()
                hist[match] += 1

            if len(descs) == 0:
                return False, 0.

            hist /= len(descs)
            hist = hist.astype(np.float32)

            # prediction = clf.predict([hist])
            score = clf.decision_function([hist])
            if score < 0:
                prediction = 0.
            else:
                prediction = 1.

            # check = resized.copy()
            # cv2.rectangle(check, (x, y), (x + 128, y + 128), (0, 255, 0), 2)
            # cv2.imshow("Window", check)
            # cv2.waitKey(1)

            if prediction == 0.:
                if i > 0:
                    detections.append([x / (0.75 ** i), y / (0.75 ** i), score[0], window_size[0] / (0.75 ** i), window_size[1] / (0.75 ** i)])
                else:
                    detections.append([x, y, score[0], window_size[0], window_size[1]])

                newdetections.append([x, y, score[0], window_size[0], window_size[1]])
                n = n + 1

        # for pred in newdetections:
        #    cv2.rectangle(clone, (pred[0], pred[1]), (pred[0] + pred[3], pred[1] + pred[4]), (0, 255, 0), 2)

        # cv2.imshow("Window res " + str(i), clone)
        print n

    # clone = image.copy()
    preds, heatmap = nms_heatmap(detections, image.shape[:2])

    max_val = np.max(heatmap)
    print "Max_val: " + str(max_val)

    # for pred in preds:
    #     cv2.rectangle(clone, (int(pred[0]), int(pred[1])), (int(pred[0]) + int(pred[3]), int(pred[1]) + int(pred[4])), (0, 255, 0), 2)

    # cv2.imshow("Result", shifted)
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.show()
    # cv2.waitKey(0)
    print("--- Overall classification took %.2f seconds -" % (time.time() - overall_start_time))

    if max_val > threshold:
        return True, max_val
    else:
        return False, max_val

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    ap.add_argument("-d", "--id", required=True, help="image id")
    
    args = vars(ap.parse_args())
    compliant, max_val = classify_img(args['image'])
    if compliant:
        print "Image classified as compliant"
    else:
        print "Image classified as not compliant"

    target = open("img_" + args['id'] + ".txt", 'w')
    target.write(str(max_val) + "\n")
    if compliant:
        target.write("compliant")
    else:
        target.write("non-compliant")
    target.close()
