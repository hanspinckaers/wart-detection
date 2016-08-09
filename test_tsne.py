import cv2
import os
import fnmatch
import numpy as np
from bhtsne import run_bh_tsne

from image_scatter import image_scatter
from matplotlib import pyplot as plt

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


def get_features(images, n_features):
    featureVectors = np.zeros((len(images) * n_features, 128), dtype=np.float32)
    p_i = 0
    
    for i, filename in enumerate(images):
        print filename

        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.02, edgeThreshold=5, sigma=0.4)
        kps, desc = sift.detectAndCompute(gray, None)
        # kps = sift.detect(gray)

        orb = cv2.ORB_create(nfeatures=n_features, edgeThreshold=2)
        kps = orb.detect(gray)

        # latch = cv2.xfeatures2d.LATCH_create()
        # _, desc = latch.compute(gray, kps)

        test = cv2.drawKeypoints(gray, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('test', test)
        cv2.waitKey(0)

        if desc is None:
            continue

        len_features = len(desc)

        if len_features == 0:
            continue

        featureVectors[p_i:p_i + len_features] = desc
        p_i = p_i + len_features
        
        continue

    print p_i
    return featureVectors[0: p_i + 1]

load_saved = True

if load_saved:
    wart_features = np.load("wart_features.npy")
    wart_features = np.delete(wart_features, np.where(~wart_features.any(axis=1))[0], 0)
    wart_features = wart_features.astype(np.float32)

    wart_features_cream = np.load("warts_cream.npy")
    # negative_features = negative_features[0:9000]
    wart_features_cream = np.delete(wart_features_cream, np.where(~wart_features_cream.any(axis=1))[0], 0)
    wart_features_cream = wart_features_cream.astype(np.float32)
    
else:
    wart_features = get_features(warts, 50)
    wart_features = np.delete(wart_features, np.where(~wart_features.any(axis=1))[0], 0)
    np.save("wart_features", wart_features)
    
    wart_features_cream = get_features(warts_cream, 50)
    wart_features_cream = np.delete(wart_features_cream, np.where(~wart_features_cream.any(axis=1))[0], 0)
    np.save("warts_cream", wart_features_cream)

    # negative_features = get_features(negatives, 50)
    # np.save("negative_features", negative_features)

features = np.concatenate((wart_features, wart_features_cream))

# train the bag of words
if True:
    vocabulary = np.load("vocabulary.npy")
else:
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

load_tsne = True
if not load_tsne:
    # double code: should be a function
    for i, filename in enumerate(warts):
        print filename
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            print "no kps for " + str(filename)
            continue

        hist = extractor.compute(image, kp)

        if np.sum(hist) == 0:  # this shouldn't happen... histogram always sums to 1
            print "hist == 0 no kps for " + str(filename)
            continue

        histograms[i] = hist
        labels[i] = 1
        images.append(image)

    for i, filename in enumerate(warts_cream):
        print filename
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kp, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            print "no kps for " + str(filename)
            continue

        hist = extractor.compute(image, kp)

        if np.sum(hist) == 0:
            print "hist == 0 no kps for " + str(filename)
            continue

        histograms[i + len(warts)] = hist
        labels[i + len(warts)] = 180
        images.append(image)

    histograms = np.delete(histograms, np.where(~histograms.any(axis=1))[0], 0)
    labels = np.delete(labels, np.where(labels == 0), 0)
    print len(labels)
    print len(histograms)

    np.save("images", images)
    features_TSNE = run_bh_tsne(histograms, verbose=True)
    np.save("tsne", features_TSNE)
    np.save("labels", labels)

else:
    # images = np.load("images.npy")
    images = []
    labels = np.load("labels.npy")
    print len(labels)
    features_TSNE = np.load("tsne.npy")
    print len(features_TSNE)

img = image_scatter(features_TSNE, images, 50)
img = img * 255
cv2.imwrite("image_scatter.png", img)

plt.scatter(features_TSNE[:, 0], features_TSNE[:, 1], c=labels, cmap="viridis")
plt.clim(-0.5, 9.5)
figure = plt.gcf()  # get current figure
figure.set_size_inches(35, 35)
plt.savefig("sift-tsne-scatter.png", dpi=100, bbox_inches='tight')

