import cv2
import os
import fnmatch
import numpy as np
import time
import sys
sys.path.append('../3_svm_model/')

from bhtsne import run_bh_tsne
from image_scatter import image_scatter
from image_scatter import min_resize
from matplotlib import pyplot as plt
from kmajority import kmajority, compute_hamming_hist
from features import get_descriptor, norm_for_descriptor, get_features_array


def analyze_images(class_one, class_two, detector_name, descriptor_name, n_features, sensitivity, bow_size, gray_detector=True, gray_descriptor=True, cache=False, testing=False, img_size=50, border=5, scatter_size=8000):
    overall_start_time = time.time()

    norm = norm_for_descriptor(descriptor_name)

    arg_string = 'new - det ' + detector_name + ' - desc ' + descriptor_name + ' - n ' + str(n_features) + ' - s ' + str(sensitivity) + ' - bow ' + str(bow_size)

    print str(os.getpid()) + "----- Run with: " + arg_string
    print str(os.getpid()) + "----- Using cache: " + str(cache)

    if not os.path.exists("cache/"):
        os.makedirs("cache/")

    if not os.path.exists("results/"):
        os.makedirs("results/")

    feature_cache = False
    if os.path.isfile("cache/wart_features" + arg_string + ".npy") and os.path.isfile("cache/warts_cream_features" + arg_string + ".npy"):
        feature_cache = True

    # cache is broken for features! (features_per_img not saved)
    if not feature_cache or cache or True:
        print str(os.getpid()) + "--- Create features ---"

        wart_features_per_img = get_features_array(class_one, sensitivity=sensitivity, detector=detector_name, max_features=n_features, descriptor=descriptor_name, gray_detector=gray_detector, gray_descriptor=gray_descriptor, testing=testing)
        wart_features = [item for sublist in wart_features_per_img for item in sublist]
        wart_features = np.asarray(wart_features)
        if cache:
            np.save("cache/wart_features" + arg_string, wart_features)

        wart_features_cream_per_img = get_features_array(class_two, sensitivity=sensitivity, detector=detector_name, max_features=n_features, descriptor=descriptor_name, testing=testing)
        wart_features_cream = [item for sublist in wart_features_cream_per_img for item in sublist]
        wart_features_cream = np.asarray(wart_features_cream)
        if cache:
            np.save("cache/warts_cream_features" + arg_string, wart_features_cream)
    else:
        print str(os.getpid()) + "Loading features from cache"
        wart_features = np.load("cache/wart_features" + arg_string + ".npy")
        wart_features = wart_features.astype(np.float32)

        wart_features_cream = np.load("cache/warts_cream_features" + arg_string + ".npy")
        wart_features_cream = wart_features_cream.astype(np.float32)

    features = np.concatenate((wart_features, wart_features_cream))
    np.random.RandomState(1)
    np.random.shuffle(features)

    # local bug numba
    # train the bag of words
    bow_cache = False
    if os.path.isfile("cache/vocabulary" + arg_string + ".npy"):
        bow_cache = True

    if not bow_cache or not cache:
        print str(os.getpid()) + "--- Training BOW (can take a while) ---"
        if norm == cv2.NORM_L2:
            bow = cv2.BOWKMeansTrainer(bow_size)
            bow.add(features)
            vocabulary = bow.cluster()
        else:
            # implementation of https://www.researchgate.net/publication/236010493_A_Fast_Approach_for_Integrating_ORB_Descriptors_in_the_Bag_of_Words_Model
            vocabulary = kmajority(features.astype(int), bow_size)

        # if cache:
        np.save("cache/vocabulary" + arg_string, vocabulary)
    else:
        print str(os.getpid()) + "Loading vocabulary from cache"
        vocabulary = np.load("cache/vocabulary" + arg_string + ".npy")
        if norm != cv2.NORM_L2:
            vocabulary = vocabulary.astype(int)

    hist_cache = False
    if os.path.isfile("cache/images" + arg_string + ".npy") and os.path.isfile("cache/labels" + arg_string + ".npy") and os.path.isfile("cache/tsne" + arg_string + ".npy"):
        hist_cache = True

    if not hist_cache or not cache:
        print str(os.getpid()) + "--- Create histograms ---"

        descriptor, _ = get_descriptor(descriptor_name, sensitivity, n_features)

        histograms = np.zeros((len(class_one) + len(class_two), bow_size), dtype=np.float32)
        labels = np.zeros(len(class_one) + len(class_two))

        images = []
        i = 0
        no_feat_counter = 0
        for label, wart_imgs in enumerate([wart_features_per_img, wart_features_cream_per_img]):
            for j, descs in enumerate(wart_imgs):
                if len(descs) == 0:
                    continue

                if label == 0:
                    img = cv2.imread(class_one[j])
                else:
                    img = cv2.imread(class_two[j])

                if norm == cv2.NORM_L2:
                    hist = np.zeros(len(vocabulary))

                    for desc in descs:
                        match = np.sum(np.square(np.abs(vocabulary - desc)),1).argmin()  # ask yuri if this is ok
                        hist[match] += 1

                    hist /= len(descs)
                    hist = hist.astype(np.float32)

                else:
                    if descs is None or len(descs) == 0:
                        no_feat_counter += 1
                        continue

                    hist = compute_hamming_hist(descs, vocabulary)

                if np.sum(hist) == 0:  # this shouldn't happen... histogram always sums to 1 (except when no match)
                    print "[Error] hist == 0 no kps for " + str(filename) + "--- \n"
                    continue

                histograms[i] = hist
                labels[i] = label * 179 + 1  # 1 = blue, 179 = yellow
                images.append(img)

                i += 1

        if no_feat_counter > 0:
            print str(os.getpid()) + "--- No histograms for %s images ---" % str(no_feat_counter)

        histograms = np.delete(histograms, np.where(~histograms.any(axis=1))[0], 0)
        labels = np.delete(labels, np.where(labels == 0), 0)

        print str(os.getpid()) + "--- Run TSNE ---"
        features_TSNE = run_bh_tsne(histograms, verbose=testing)

        if cache:
            np.save("cache/images" + arg_string, images)
            np.save("cache/tsne" + arg_string, features_TSNE)
            np.save("cache/labels" + arg_string, labels)

    else:
        images = np.load("cache/images" + arg_string + ".npy")
        labels = np.load("cache/labels" + arg_string + ".npy")
        features_TSNE = np.load("cache/tsne" + arg_string + ".npy")

    # create a border around images for labelling in scatter plot
    for i, image in enumerate(images):
        image = min_resize(image, img_size - border * 2)
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255,255,255))

        b = border - 1
        if labels[i] == 180:
            images[i] = cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_CONSTANT, value=(17,141,246))  # orange
        else:
            images[i] = cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_CONSTANT, value=(245,219,10))  # blue

    # make plots
    img = image_scatter(features_TSNE, images, scatter_size=8000, img_size=img_size)
    cv2.imwrite("results/image_scatter" + arg_string + "n_feat " + str(len(features)) + ".png", img)

    plt.clf()
    plt.scatter(features_TSNE[:, 0], features_TSNE[:, 1], c=labels, cmap="viridis")
    plt.clim(-0.5, 9.5)
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(35, 35)
    plt.savefig("results/tsne_scatter" + arg_string + "n_feat " + str(len(features)) + ".png", dpi=100, bbox_inches='tight')

    print(str(os.getpid()) + "--- Overall in %.3f seconds -" % (time.time() - overall_start_time))


if __name__ == '__main__':
    if len(sys.argv) == 6:
        detector_name = sys.argv[1]
        descriptor_name = sys.argv[2]
        n_features = int(sys.argv[3])
        sensitivity = int(sys.argv[4])
        bow_size = int(sys.argv[5])

        warts = []

        for root, dirnames, filenames in os.walk("../../images/classified/warts"):
            for filename in fnmatch.filter(filenames, '*.png'):
                warts.append("../../images/classified/warts" + "/" + filename)

        negatives = []

        for root, dirnames, filenames in os.walk("../../images/classified/negatives"):
            for filename in fnmatch.filter(filenames, '*.png'):
                negatives.append("../../images/classified/negatives" + "/" + filename)

        warts_cream = []

        for root, dirnames, filenames in os.walk("../../images/classified/warts_cream"):
            for filename in fnmatch.filter(filenames, '*.png'):
                warts_cream.append("../../images/classified/warts_cream" + "/" + filename)
                # warts.append("classified/warts_cream" + "/" + filename)

        analyze_images(warts, negatives, detector_name, descriptor_name, n_features, sensitivity, bow_size, testing=False, cache=False, scatter_size=10000)
