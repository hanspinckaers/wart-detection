import cv2
import os
import fnmatch
import numpy as np
from bhtsne import run_bh_tsne
from image_scatter import image_scatter
from image_scatter import min_resize
from matplotlib import pyplot as plt

# symbols used for printing output
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

# sift-tsne-10-features-per-img.png
# params: sift = cv2.xfeatures2d.SIFT_create(nfeatures=10, contrastThreshold=0.02, edgeThreshold=2, sigma=0.4)
# same params for matching

warts = []

for root, dirnames, filenames in os.walk("classified/warts"):
    for filename in fnmatch.filter(filenames, '*.png'):
        warts.append("classified/warts" + "/" + filename)

negatives = []

for root, dirnames, filenames in os.walk("classified/negatives"):
    for filename in fnmatch.filter(filenames, '*.png'):
        negatives.append("classified/negatives" + "/" + filename)


warts_cream = []

for root, dirnames, filenames in os.walk("classified/warts_cream"):
    for filename in fnmatch.filter(filenames, '*.png'):
        warts_cream.append("classified/warts_cream" + "/" + filename)


# http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html

#     if chunks[0] == 'sift':
#         detector = cv2.xfeatures2d.SIFT_create()
#         norm = cv2.NORM_L2
#     elif chunks[0] == 'surf':
#         detector = cv2.xfeatures2d.SURF_create(800)
#         norm = cv2.NORM_L2
#     elif chunks[0] == 'orb':
#         detector = cv2.ORB_create(400)
#         norm = cv2.NORM_HAMMING
#     elif chunks[0] == 'akaze':
#         detector = cv2.AKAZE_create()
#         norm = cv2.NORM_HAMMING
#     elif chunks[0] == 'brisk':
#         detector = cv2.BRISK_create()
#         norm = cv2.NORM_HAMMING

# norm should be used in matches (currently brute force, use flann?!
# see opencv/samples/python/find_obj.py

# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#sift-sift
def SIFT_detector(sensitivity, n_features):
    if sensitivity == 2:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.01, edgeThreshold=20, sigma=0.4)
    elif sensitivity == 1:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.02, edgeThreshold=10, sigma=0.4)
    elif sensitivity == 0:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.02, edgeThreshold=7, sigma=0.4)
    return sift


# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#SURF%20:%20public%20Feature2D
def SURF_detector(sensitivity, n_features):
    if sensitivity == 2:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=1)
    elif sensitivity == 1:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=30, nOctaves=1)
    elif sensitivity == 0:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=50, nOctaves=2)
    return surf


# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#SURF%20:%20public%20Feature2D
def AKAZE_detector(sensitivity, n_features):
    if sensitivity == 2:
        akaze = cv2.AKAZE_create(threshold=0.000001)
    elif sensitivity == 1:
        akaze = cv2.AKAZE_create(threshold=0.00001)
    elif sensitivity == 0:
        akaze = cv2.AKAZE_create(threshold=0.0001)
    return akaze


# http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html#gsc.tab=0
def ORB_detector(sensitivity, n_features):
    if sensitivity == 2:
        orb = cv2.ORB_create(n_features, edgeThreshold=10, patchSize=10)
    elif sensitivity == 1:
        orb = cv2.ORB_create(n_features)
    elif sensitivity == 0:
        orb = cv2.ORB_create(n_features)
    return orb


# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#SURF%20:%20public%20Feature2D
# the BRISK implementation is broken in openCV (https://github.com/opencv/opencv/pull/3383)
# def BRISK_detector(sensitivity, n_features):


# cannot seem to get it working
# http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=surffeaturedetector#StarFeatureDetector%20:%20public%20FeatureDetector
def STAR_detector(sensitivity, n_features):
    if sensitivity == 2:
        star = cv2.xfeatures2d.StarDetector_create(responseThreshold=5, lineThresholdProjected=5)
    elif sensitivity == 1:
        star = cv2.xfeatures2d.StarDetector_create()
    elif sensitivity == 0:
        star = cv2.xfeatures2d.StarDetector_create()
    return star


# http://docs.opencv.org/3.1.0/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html#gsc.tab=0
# NB only descriptor!
# diffusivity
def FREAK_descriptor(sensitivity, keypoints):
    if sensitivity == 2:
        freak = cv2.xfeatures2d.FREAK_create(patternScale=10.)
    elif sensitivity == 1:
        freak = cv2.xfeatures2d.SURF_create(hessianThreshold=30, nOctaves=1)
    elif sensitivity == 0:
        freak = cv2.xfeatures2d.SURF_create(hessianThreshold=50, nOctaves=2)
    return freak

# latch = cv2.xfeatures2d.LATCH_create()
# _, desc = latch.compute(gray, kps)


def get_features(images, n_features):
    p_i = 0
    featureVectors = None
    for i, filename in enumerate(images):
        if i > 0:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
        print "--- Extracting features " + str(filename) + " ---"

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detector = SIFT_detector(2, n_features)
        kps, desc = detector.detectAndCompute(gray, None)

        # test = cv2.drawKeypoints(gray, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('test', test)
        # cv2.waitKey(0)

        if desc is None:
            continue

        len_features = desc.shape[0]

        if len_features == 0:
            continue

        if featureVectors is None:
            featureVectors = np.zeros((len(images) * n_features * 2, desc.shape[1]), dtype=np.float32)

        # if featureVectors[p_i:p_i + len_features].shape != desc.shape:
        #    pu.db

        featureVectors[p_i:p_i + len_features] = desc

        p_i = p_i + len_features

        continue
    print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

    return featureVectors[0: p_i + 1]

load_saved = False

if load_saved:
    wart_features = np.load("wart_features.npy")
    wart_features = np.delete(wart_features, np.where(~wart_features.any(axis=1))[0], 0)
    wart_features = wart_features.astype(np.float32)

    wart_features_cream = np.load("warts_cream.npy")
    wart_features_cream = np.delete(wart_features_cream, np.where(~wart_features_cream.any(axis=1))[0], 0)
    wart_features_cream = wart_features_cream.astype(np.float32)

else:
    wart_features = get_features(warts, 10)
    wart_features = np.delete(wart_features, np.where(~wart_features.any(axis=1))[0], 0)
    np.save("wart_features", wart_features)

    wart_features_cream = get_features(warts_cream, 10)
    wart_features_cream = np.delete(wart_features_cream, np.where(~wart_features_cream.any(axis=1))[0], 0)
    np.save("warts_cream", wart_features_cream)

features = np.concatenate((wart_features, wart_features_cream))

# train the bag of words
if False:
    vocabulary = np.load("vocabulary.npy")
else:
    print "--- Training BOW (can take a long time) ---"

    bow = cv2.BOWKMeansTrainer(1000)
    bow.add(features)
    vocabulary = bow.cluster()

    np.save("vocabulary", vocabulary)

# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# matcher = cv2.FlannBasedMatcher(index_params, search_params)

# make a matcher to bag of words
matcher = cv2.BFMatcher()

sift = cv2.xfeatures2d.SIFT_create(nfeatures=10, contrastThreshold=0.02, edgeThreshold=5, sigma=0.4)
extractor = cv2.BOWImgDescriptorExtractor(sift, matcher)
extractor.setVocabulary(vocabulary)

histograms = np.zeros((len(warts) + len(warts_cream), 1000), dtype=np.float32)
labels = np.zeros(len(warts) + len(warts_cream))

images = []

load_cache = False
if not load_cache:
    # double code: should be a function
    for i, filename in enumerate(warts):
        if i > 0:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

        print "--- Creating histograms " + str(filename) + " ---"

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
            print "--- No features found for " + str(filename) + "--- \n"
            continue

        hist = extractor.compute(image, kp)

        if np.sum(hist) == 0:  # this shouldn't happen... histogram always sums to 1
            print "[Error] hist == 0 no kps for " + str(filename) + "--- \n"
            continue

        histograms[i] = hist
        labels[i] = 1
        images.append(image)

    for i, filename in enumerate(warts_cream):
        if i > 0:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

        print "--- Creating histograms " + str(filename) + " ---"

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
            print "--- No features found for " + str(filename) + "--- \n"
            continue

        hist = extractor.compute(image, kp)

        if np.sum(hist) == 0:
            print "[Error] hist == 0 no kps for " + str(filename) + "--- \n"
            continue

        histograms[i + len(warts)] = hist
        labels[i + len(warts)] = 180
        images.append(image)

    histograms = np.delete(histograms, np.where(~histograms.any(axis=1))[0], 0)
    labels = np.delete(labels, np.where(labels == 0), 0)

    print "--- Run TSNE " + str(filename) + " ---"

    features_TSNE = run_bh_tsne(histograms, verbose=True)

    np.save("images", images)
    np.save("tsne", features_TSNE)
    np.save("labels", labels)

else:
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    features_TSNE = np.load("tsne.npy")

# subset_features = features_TSNE[(features_TSNE[:, 0] < -30) & (features_TSNE[:, 1] > 0)]
subset_features = features_TSNE

for i, image in enumerate(images):
    image = min_resize(image, 40)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255,255,255))

    if labels[i] == 180:
        images[i] = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(17,141,246))
    else:
        images[i] = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(245,219,10))


img = image_scatter(subset_features, images[0:len(subset_features)], scatter_size=8000)
cv2.imwrite("image_scatter.png", img)

plt.scatter(features_TSNE[:, 0], features_TSNE[:, 1], c=labels, cmap="viridis")
plt.clim(-0.5, 9.5)
figure = plt.gcf()  # get current figure
figure.set_size_inches(35, 35)
plt.savefig("tsne-scatter.png", dpi=100, bbox_inches='tight')
