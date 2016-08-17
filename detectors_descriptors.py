import cv2
import numpy as np
import pudb

# Please Note: The detector parameters are tweaked by hand for the wart pictures, when using detectors for other purposes you may need to change the parameters

# symbols used for printing output
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


def get_features(images, detector, descriptor, max_features=500, gray_detector=True, gray_descriptor=True):
    """
    This function runs detector.detect() on all images and returns all features in a numpy array

    Parameters
    ---------
    images: array
        The array of images to run the feature detection on

    detector: object
        Should be able to respond to detect() and return keypoints.

    descriptor: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    max_features: integer
        Maximum amount of features per image, needed to initialize array of features

    gray_detector: boolean
        If True all images will be converted to grayscale before running detector

    gray_descriptor: boolean
        If True all images will be converted to grayscale before running descriptor


    Returns
    ------
    features: numpy array
        array of all the descriptors of the features in images
    """
    p_i = 0
    featureVectors = None
    for i, filename in enumerate(images):
        if i > 0:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

        print "--- Extracting features " + str(filename) + " ---"

        image = cv2.imread(filename)

        if gray_detector or gray_descriptor:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detector = SIFT_detector(2, n_features)
        if gray_detector:
            kps = detector.detect(gray)
        else:
            kps = detector.detect(image)

        if gray_descriptor:
            _, desc = descriptor.compute(gray, kps)
        else:
            _, desc = descriptor.compute(image, kps)

        pu.db

        # test = cv2.drawKeypoints(gray, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('test', test)
        # cv2.waitKey(0)

        if desc is None:
            continue

        len_features = desc.shape[0]

        if len_features == 0:
            continue

        if featureVectors is None:
            featureVectors = np.zeros((len(images) * 500, desc.shape[1]), dtype=np.float32)  # max 500 features per image

        featureVectors[p_i:p_i + len_features] = desc

        p_i = p_i + len_features

        continue

    print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

    return featureVectors[0: p_i + 1]


def get_detector(name, sensitivity, n_features):
    detector_func = None
    if name == 'SIFT':
        detector_func = SIFT_detector
    elif name == 'SURF':
        detector_func = SURF_detector
    elif name == 'AKAZE':
        detector_func = AKAZE_detector
    elif name == 'KAZE':
        detector_func = KAZE_detector
    elif name == 'ORB':
        detector_func = ORB_detector
    elif name == 'BRISK':
        detector_func = BRISK_detector  # doesn't work
    elif name == 'Agast':
        detector_func = Agast_detector
    elif name == 'GFTT':
        detector_func = GFTT_detector
    elif name == 'MSER':  # can work on color!
        detector_func = MSER_detector
    return detector_func(sensitivity, n_features)


def norm_for_descriptor(name):
    # binary descriptors should be compared with Hamming distance
    if name == 'SIFT':
        return cv2.NORM_L2  # just the Euclidean distance
    if name == 'SURF':
        return cv2.NORM_L2
    if name == 'AKAZE':
        return cv2.NORM_HAMMING  # the Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different.
    if name == 'ORB':
        return cv2.NORM_HAMMING
    if name == 'BRISK':
        return cv2.NORM_HAMMING
    if name == 'BRIEF':
        return cv2.NORM_HAMMING
    if name == 'FREAK':
        return cv2.NORM_HAMMING

    return cv2.NORM_L2


def get_descriptor(name):
    desc = None
    if name == 'SIFT':
        desc = cv2.xfeatures2d.SIFT_create()
    elif name == 'SURF':
        desc = cv2.xfeatures2d.SURF_create()
    elif name == 'AKAZE':  # AKAZE detector can only be used with the AKAZE descriptor
        desc = cv2.AKAZE_create()
    elif name == 'ORB':
        desc = cv2.ORB_create()
    elif name == 'BRIEF':
        desc = cv2.BriefDescriptorExtractor_create()
    elif name == 'FREAK':
        desc = cv2.xfeatures2d.FREAK_create(patternScale=10.)  # FREAK is the retina descriptor
    elif name == 'BRISK':
        desc = cv2.BRISK_create()

    return desc


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


# http://docs.opencv.org/trunk/d3/d61/classcv_1_1KAZE.html#gsc.tab=0
def KAZE_detector(sensitivity, n_features):
    if sensitivity == 2:
        kaze = cv2.KAZE_create(threshold=0.000001)
    elif sensitivity == 1:
        kaze = cv2.KAZE_create(threshold=0.00001)
    elif sensitivity == 0:
        kaze = cv2.KAZE_create(threshold=0.0001)
    return kaze


# http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html#gsc.tab=0
def ORB_detector(sensitivity, n_features):
    if sensitivity == 2:
        orb = cv2.ORB_create(n_features, edgeThreshold=1)
    elif sensitivity == 1:
        orb = cv2.ORB_create(n_features)
    elif sensitivity == 0:
        orb = cv2.ORB_create(n_features)
    return orb


# http://docs.opencv.org/trunk/d7/d19/classcv_1_1AgastFeatureDetector.html#gsc.tab=0
def Agast_detector(sensitivity, n_features):
    if sensitivity == 2:
        agast = cv2.AgastFeatureDetector_create(threshold=3)
    elif sensitivity == 1:
        agast = cv2.AgastFeatureDetector_create(threshold=6)
    elif sensitivity == 0:
        agast = cv2.AgastFeatureDetector_create(threshold=9)
    return agast


# http://docs.opencv.org/trunk/df/d21/classcv_1_1GFTTDetector.html#gsc.tab=0
def GFTT_detector(sensitivity, n_features):
    if sensitivity == 2:
        gftt = cv2.GFTTDetector_create(maxCorners=n_features)
    elif sensitivity == 1:
        gftt = cv2.GFTTDetector_create(maxCorners=n_features, minDistance=10, blockSize=5)
    elif sensitivity == 0:
        gftt = cv2.GFTTDetector_create(maxCorners=n_features, minDistance=20, blockSize=10)
    return gftt


# http://docs.opencv.org/trunk/df/d21/classcv_1_1GFTTDetector.html#gsc.tab=0
def MSER_detector(sensitivity, n_features):
    if sensitivity == 2:
        mser = cv2.MSER_create(_edge_blur_size=3, _delta=10, _min_area=3, _max_area=1000)
    elif sensitivity == 1:
        mser = cv2.GFTTDetector_create(_edge_blur_size=3, _min_area=3, _max_area=1000)
    elif sensitivity == 0:
        mser = cv2.GFTTDetector_create(_edge_blur_size=1, _min_area=3, _max_area=1000)
    return mser


# http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#brisk
def BRISK_detector(sensitivity, n_features):
    # the BRISK implementation is broken in openCV (https://github.com/opencv/opencv/pull/3383)
    return None


# cannot seem to get it working
# http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=surffeaturedetector#StarFeatureDetector%20:%20public%20FeatureDetector
def STAR_detector(sensitivity, n_features):
    return cv2.xfeatures2d.StarDetector_create(responseThreshold=5, lineThresholdProjected=5)


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
