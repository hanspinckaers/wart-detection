import cv2
import os
import fnmatch
import numpy as np
import time
import sys

from kmajority import kmajority, compute_hamming_hist
from detectors_descriptors import get_features_array, get_features

from sklearn import neighbors
from sklearn import cluster

def train_model(train_pos, train_neg, detector_name='SIFT', descriptor_name='SIFT', dect_params=None, n_features=10, bow_size=1000, k=15):
    overall_start_time = time.time()

    print("---Gather features---")
    pos_feat_p_img, neg_feat_p_img = extract_features([train_pos, train_neg], detector_name, descriptor_name, dect_params, n_features)
    pos_feat = [item for sublist in pos_feat_p_img for item in sublist]
    pos_feat = np.asarray(pos_feat)

    neg_feat = [item for sublist in neg_feat_p_img for item in sublist]
    neg_feat = np.asarray(neg_feat)

    features = np.concatenate((pos_feat, neg_feat))
    np.random.RandomState(1)
    np.random.shuffle(features)

    print("---Train BOW---")
    vocabulary = train_bagofwords(features, bow_size)
    print("---Make hists---")
    hists, labels, _ = hist_using_vocabulary([pos_feat_p_img, neg_feat_p_img], vocabulary)
    print("---Fit model---")
    model = fit_model_kneighbors(hists, labels, k)

    print("--- Overall training model took %s ---" % (time.time() - overall_start_time))

    return model, vocabulary


def validate_model(model, val_pos, val_neg, detector_name='SIFT', descriptor_name='SIFT', dect_params=None, n_features=10):
    features = np.concatenate((val_pos, val_neg))
    np.random.RandomState(1)
    np.random.shuffle(features)


def classify_img_using_model(img_filename, vocabulary, model, detector_name='SIFT', descriptor_name='SIFT', dect_params=None, n_features=10):
    descs = get_features(img_filename, detector_name, descriptor_name, dect_params, max_features=n_features)
    hist = hist_with_img(descs, vocabulary)
    pred = model.predict([hist])
    return pred[0]


def extract_features(classes, detector_name, descriptor_name, dect_params, n_features):
    features_per_class = []
    for c in classes:
        features = get_features_array(c, detector_name, descriptor_name, dect_params, max_features=n_features)
        features_per_class.append(features)
    return features_per_class


def train_bagofwords(features, bow_size, norm=cv2.NORM_L2):
    if norm == cv2.NORM_L2:
        # scipy:
        start_time = time.time()

        est = cluster.KMeans(bow_size, n_jobs=-1)
        est.fit(features)
        est.cluster_centers_

        print("--- scipy took %s ---" % (time.time() - start_time))

        start_time = time.time()

        bow = cv2.BOWKMeansTrainer(bow_size)
        bow.add(features)
        vocabulary = bow.cluster()

        print("--- cv2 took %s ---" % (time.time() - start_time))

    else:
        # implementation of https://www.researchgate.net/publication/236010493_A_Fast_Approach_for_Integrating_ORB_Descriptors_in_the_Bag_of_Words_Model
        vocabulary = kmajority(features.astype(int), bow_size)

    return vocabulary


def hist_with_img(descs, vocabulary, norm=cv2.NORM_L2):
    if norm == cv2.NORM_L2:
        hist = np.zeros(len(vocabulary))

        for desc in descs:
            match = np.sum(np.square(np.abs(vocabulary - desc)),1).argmin()
            hist[match] += 1

        hist /= len(descs)
        hist = hist.astype(np.float32)

    else:
        hist = compute_hamming_hist(descs, vocabulary)

    return hist


def hist_using_vocabulary(feat_per_img_per_class, vocabulary, norm=cv2.NORM_L2):
    max_count = 0
    for c in feat_per_img_per_class:
        max_count += len(c)

    histograms = np.zeros((max_count, len(vocabulary)), dtype=np.float32)
    labels = np.zeros(max_count)
    indices = np.zeros(max_count)

    i = 0
    no_feat_counter = 0
    for label, wart_imgs in enumerate(feat_per_img_per_class):
        for j, descs in enumerate(wart_imgs):
            if descs is None or len(descs) == 0:
                no_feat_counter += 1
                continue

            hist = hist_with_img(descs, vocabulary, norm)

            if np.sum(hist) == 0:  # this shouldn't happen... histogram always sums to 1 (except when no match)
                print "[Error] hist == 0 no kps for " + str(filename) + "--- \n"
                continue

            histograms[i] = hist
            labels[i] = label
            indices[i] = j

            i += 1

    if no_feat_counter > 0:
        print str(os.getpid()) + "--- No histograms for %s images ---" % str(no_feat_counter)

    return (histograms[0:i], labels[0:i], indices[0:i])


def fit_model_kneighbors(feat, classes, k, weights='uniform'):
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    clf.fit(feat, classes)
    return clf


if __name__ == '__main__':
    if len(sys.argv) == 3:
        n_features = int(sys.argv[1])
        bow_size = int(sys.argv[2])

        warts = []

        for root, dirnames, filenames in os.walk("classified/warts"):
            for filename in fnmatch.filter(filenames, '*.png'):
                warts.append("classified/warts" + "/" + filename)

        negatives = []

        for root, dirnames, filenames in os.walk("classified/negatives"):
            for filename in fnmatch.filter(filenames, '*.png'):
                negatives.append("classified/negatives" + "/" + filename)

        for root, dirnames, filenames in os.walk("classified/warts_cream"):
            for filename in fnmatch.filter(filenames, '*.png'):
                warts.append("classified/warts_cream" + "/" + filename)

        model_vocabulary = train_model(warts, negatives)
