import cv2
import os
import fnmatch
import numpy as np
from bhtsne import run_bh_tsne
from image_scatter import image_scatter
from image_scatter import min_resize
from matplotlib import pyplot as plt
import time

# kmedoids
from kmajority import kmajority

from detectors_descriptors import get_detector, get_descriptor, norm_for_descriptor, get_features
import pudb

# arguments:
detector_name = 'SIFT'
descriptor_name = 'FREAK'
n_features = 10
sensitivity = 2
norm = norm_for_descriptor(descriptor_name)
matcher = 'BF'

gray_detector = False
gray_descriptor = False

cached = True

# could we use Bayesian optimization?

arg_string = ' - detect ' + detector_name + ' - desc ' + descriptor_name + ' - n_feat ' + str(n_features) + ' - matcher ' + matcher
print "----- Run with: " + arg_string
print "----- Using cache: " + str(cached)

if not os.path.exists("cache/"):
    os.makedirs("cache/")

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


if not cached:
    detector = get_detector(detector_name, sensitivity, n_features=n_features)
    descriptor = get_descriptor(descriptor_name)

    wart_features = get_features(warts, detector=detector, descriptor=descriptor, gray_detector=gray_detector, gray_descriptor=gray_descriptor)
    # wart_features = np.delete(wart_features, np.where(~wart_features.any(axis=1))[0], 0)
    np.save("cache/wart_features" + arg_string, wart_features)
    
    wart_features_cream = get_features(warts_cream, detector=detector, descriptor=descriptor)
    # wart_features_cream = np.delete(wart_features_cream, np.where(~wart_features_cream.any(axis=1))[0], 0)
    np.save("cache/warts_cream_features" + arg_string, wart_features_cream)
else:
    print "Loading features from cache"
    wart_features = np.load("cache/wart_features" + arg_string + ".npy")
    wart_features = wart_features.astype(np.float32)

    wart_features_cream = np.load("cache/warts_cream_features" + arg_string + ".npy")
    wart_features_cream = wart_features_cream.astype(np.float32)

features = np.concatenate((wart_features, wart_features_cream))

# local bug numba
# train the bag of words
if cached:
    print "--- Training BOW (can take a long time) ---"
    if norm == cv2.NORM_L2:
        bow = cv2.BOWKMeansTrainer(1000)
        bow.add(features)
        vocabulary = bow.cluster()
    else:
        # implementation of https://www.researchgate.net/publication/236010493_A_Fast_Approach_for_Integrating_ORB_Descriptors_in_the_Bag_of_Words_Model
        binary_vectors = np.unpackbits(features.astype(np.ubyte), axis=1)
        vocabulary = kmajority(features.astype(int), 1000)

    np.save("cache/vocabulary" + arg_string, vocabulary)
else:
    print "Loading vocabulary from cache"
    vocabulary = np.load("cache/vocabulary" + arg_string + ".npy")
    pu.db

# FlannMatcher broken in 3.1.0, pull request that fixes it: https://github.com/opencv/opencv/issues/5667
# should built openCV from source!
# Could use DescriptorMatcher_create?
if matcher == 'Flann':
    print "--- Using Flann ---"
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH = 6

    if norm == cv2.NORM_L2:
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        flann_params = dict(algorithm=FLANN_INDEX_LSH,table_number=20, key_size=10, multi_probe_level=2)

    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
else:
    matcher = cv2.BFMatcher(norm, crossCheck=True)


if cached:
    detector = get_detector(detector_name, sensitivity, n_features=n_features)
    descriptor = get_descriptor(descriptor_name)

    extractor = cv2.BOWImgDescriptorExtractor(descriptor, matcher)
    extractor.setVocabulary(vocabulary)

    matcher.add([1, vocabulary])

    histograms = np.zeros((len(warts) + len(warts_cream), 1000), dtype=np.float32)
    labels = np.zeros(len(warts) + len(warts_cream))

    images = []

    for images in [warts, warts_cream]:
        for i, filename in enumerate(images):
            if i > 0:
                print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

            print "--- Creating histograms " + str(filename) + " ---"

            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            kps = detector.detect(gray, None)
            if kps is None or len(kps) == 0:
                print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
                print "--- No features found for " + str(filename) + "--- \n"
                continue

            if gray_descriptor:
                _, desc = descriptor.compute(gray, kps)
            else:
                _, desc = descriptor.compute(image, kps)

            # desc = desc.astype(np.float32)
            # pu.db
            matcher.match(desc)
            matches = matcher.match(desc, vocabulary)

            # pu.db

            hist = extractor.compute(gray, kps)

            if np.sum(hist) == 0:  # this shouldn't happen... histogram always sums to 1
                print "[Error] hist == 0 no kps for " + str(filename) + "--- \n"
                continue

            histograms[i] = hist
            labels[i] = 1
            images.append(image)

    histograms = np.delete(histograms, np.where(~histograms.any(axis=1))[0], 0)
    labels = np.delete(labels, np.where(labels == 0), 0)

    print "--- Run TSNE " + str(filename) + " ---"

    features_TSNE = run_bh_tsne(histograms, verbose=True)

    np.save("cache/images" + arg_string, images)
    np.save("cache/tsne" + arg_string, features_TSNE)
    np.save("cache/labels" + arg_string, labels)

else:
    images = np.load("cache/images" + arg_string + ".npy")
    labels = np.load("cache/labels" + arg_string + ".npy")
    features_TSNE = np.load("cache/tsne" + arg_string + ".npy")

# subset_features = features_TSNE[(features_TSNE[:, 0] < -30) & (features_TSNE[:, 1] > 0)]
subset_features = features_TSNE

for i, image in enumerate(images):
    image = min_resize(image, 40)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255,255,255))

    if labels[i] == 180:
        images[i] = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(17,141,246))
    else:
        images[i] = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(245,219,10))

# make plots

img = image_scatter(subset_features, images[0:len(subset_features)], scatter_size=8000)
cv2.imwrite("results/image_scatter" + arg_string + ".png", img)

plt.scatter(features_TSNE[:, 0], features_TSNE[:, 1], c=labels, cmap="viridis")
plt.clim(-0.5, 9.5)
figure = plt.gcf()  # get current figure
figure.set_size_inches(35, 35)
plt.savefig("results/tsne_scatter" + arg_string + ".png", dpi=100, bbox_inches='tight')
