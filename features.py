import cv2
import numpy as np

# Please Note: The detector parameters are tweaked by hand for the wart pictures, when using detectors for other purposes you may need to change the parameters

# symbols used for printing output
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

# global var to cache detectors
detectors = None


def get_features_array(images, detector, descriptor, dect_params=None, sensitivity=2, max_features=500, gray_detector=True, gray_descriptor=True, testing=False):
    """
    This function runs detector.detect() on all images and returns all features in a numpy array

    Parameters
    ---------
    images: array
        The array of images to run the feature detection on

    detector: string
        Should be able to respond to detect() and return keypoints.

    descriptor: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    dect_params: dict
        If supported these params will be applied on the detector create function

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
    featureVectors = []

    for i, filename in enumerate(images):
        if testing:
            if i > 0:
                print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
            print "--- Extracting features " + str(filename) + " ---"

        desc = get_features(filename, detector, descriptor, dect_params, sensitivity, max_features, gray_detector, gray_descriptor, testing)
        len_features = len(desc)

        if featureVectors is None:
            featureVectors = np.zeros((len(images) * 500, desc.shape[1]), dtype=np.float32)  # max 500 features per image

        featureVectors.append(desc)

        p_i = p_i + len_features

        continue

    if testing:
        print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE

    return featureVectors


def get_features(filename, detector, descriptor_name, dect_params=None, sensitivity=2, max_features=500, gray_detector=True, gray_descriptor=True, testing=False):
    image = cv2.imread(filename)

    if gray_detector or gray_descriptor:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray_detector:
        dect, kps = kps_with_detector(gray, detector, sensitivity, max_features, dect_params)
    else:
        dect, kps = kps_with_detector(image, detector, sensitivity, max_features, dect_params)

    if descriptor_name == detector:
        if gray_descriptor:
            _, desc = dect.compute(gray, kps)
        else:
            _, desc = dect.compute(image, kps)
    else:
        descriptor, _ = get_descriptor(descriptor_name, sensitivity, max_features)
        if gray_descriptor:
            _, desc = descriptor.compute(gray, kps)
        else:
            _, desc = descriptor.compute(image, kps)

    # if testing:
    #     print "Number of features: " + str(len(kps))
    #     test = cv2.drawKeypoints(gray, kps, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     cv2.imshow('test', test)
    #     cv2.waitKey(0)

    if desc is None:
        return np.array([])

    len_features = desc.shape[0]

    if len_features == 0:
        return np.array([])

    return desc


def kps_with_detector(img, name, sensitivity, n_features, params):
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
    return detector_func(img, sensitivity, n_features, params)


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


def get_descriptor(name, sensitivity, n_features, params=None):
    if name == 'SIFT':
        detector_func = SIFT_detector
    elif name == 'SURF':
        detector_func = SURF_detector
    elif name == 'AKAZE':  # AKAZE detector can only be used with the AKAZE descriptor
        detector_func = AKAZE_detector
    elif name == 'ORB':
        detector_func = ORB_detector
    elif name == 'BRIEF':
        return cv2.BriefDescriptorExtractor_create()
    elif name == 'FREAK':
        detector_func = FREAK_descriptor
    elif name == 'BRISK':
        return cv2.BRISK_create()

    return detector_func(None, sensitivity, n_features)


# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#sift-sift
def SIFT_detector(img, sensitivity, n_features, params=None):
    # params: sift = cv2.xfeatures2d.SIFT_create(nfeatures=10, contrastThreshold=0.02, edgeThreshold=2, sigma=0.4)
    if params is not None:
        sift = cv2.xfeatures2d.SIFT_create(**params)
    elif sensitivity == 2:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.01, edgeThreshold=20, sigma=0.4)
    elif sensitivity == 1:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.02, edgeThreshold=10, sigma=0.4)
    elif sensitivity == 0:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=n_features, contrastThreshold=0.02, edgeThreshold=5, sigma=0.4)  # of 7 as edgethreshold?

    if img is None:
        return sift, None

    return sift, sift.detect(img)


def get_max_features(img, thresh, thresh_arg, increment, create_func, max_features, arg_dict={}, max_rounds=50):
    global detectors

    arg_dict[thresh_arg] = thresh

    if img is None:
        return create_func(**arg_dict), None

    if detectors is None:
        detectors = []
        for i in range(max_rounds):
            arg_dict[thresh_arg] += increment * i
            detectors.append(create_func(**arg_dict))

    detector = detectors[0]

    j = 0
    n_kps = max_features + 1
    while n_kps > max_features and j < max_rounds:
        detector = detectors[j]
        kps = detector.detect(img)
        n_kps = len(kps)
        j += 1

    return detector, kps


# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#SURF%20:%20public%20Feature2D
def SURF_detector(img, sensitivity, max_features, params):
    return get_max_features(img,
                            thresh=10,
                            thresh_arg="hessianThreshold",
                            increment=15,
                            arg_dict={"nOctaves": 1},
                            create_func=cv2.xfeatures2d.SURF_create,
                            max_features=max_features)


# http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html#SURF%20:%20public%20Feature2D
def AKAZE_detector(img, sensitivity, max_features, params):
    return get_max_features(img,
                            thresh=0.000001,
                            thresh_arg="threshold",
                            increment=0.000001,
                            arg_dict={},
                            create_func=cv2.AKAZE_create,
                            max_features=max_features)


# http://docs.opencv.org/trunk/d3/d61/classcv_1_1KAZE.html#gsc.tab=0
def KAZE_detector(img, sensitivity, max_features, params):
    return get_max_features(img,
                            thresh=0.000001,
                            thresh_arg="threshold",
                            increment=0.000001,
                            arg_dict={},
                            create_func=cv2.KAZE_create,
                            max_features=max_features)


# http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#mser
# cannot seem to get MSER working for wart images
def MSER_detector(img, sensitivity, max_features, params):
    return get_max_features(img,
                            thresh=3,
                            thresh_arg="_max_area",
                            increment=1,
                            arg_dict={"_edge_blur_size":2, "_delta":10, "_min_area":3},
                            create_func=cv2.MSER_create,
                            max_features=max_features)


# http://docs.opencv.org/trunk/d7/d19/classcv_1_1AgastFeatureDetector.html#gsc.tab=0
def Agast_detector(img, sensitivity, max_features, params):
    return get_max_features(img,
                            thresh=2,
                            thresh_arg="threshold",
                            increment=1,
                            arg_dict={},
                            create_func=cv2.AgastFeatureDetector_create,
                            max_features=max_features)


# http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html#gsc.tab=0
def ORB_detector(img, sensitivity, n_features, params):
    if sensitivity == 2:
        orb = cv2.ORB_create(n_features, edgeThreshold=1)
    elif sensitivity == 1:
        orb = cv2.ORB_create(n_features)
    elif sensitivity == 0:
        orb = cv2.ORB_create(n_features)
    if img is None:
        return orb, None
    return orb, orb.detect(img)


# http://docs.opencv.org/trunk/df/d21/classcv_1_1GFTTDetector.html#gsc.tab=0
# J. Shi and C. Tomasi. Good Features to Track. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 593-600, June 1994.
def GFTT_detector(img, sensitivity, n_features, params):
    if sensitivity == 2:
        gftt = cv2.GFTTDetector_create(maxCorners=n_features)
    elif sensitivity == 1:
        gftt = cv2.GFTTDetector_create(maxCorners=n_features, minDistance=10, blockSize=5)
    elif sensitivity == 0:
        gftt = cv2.GFTTDetector_create(maxCorners=n_features, minDistance=20, blockSize=10)

    if img is None:
        return gftt, None

    return gftt, gftt.detect(img)


# http://docs.opencv.org/3.0-beta/modules/features2d/doc/feature_detection_and_description.html#brisk
def BRISK_detector(sensitivity, n_features, params):
    print "ERROR: BRISK detector not implemented"
    # the BRISK implementation is broken in openCV (https://github.com/opencv/opencv/pull/3383)
    return None, None


# cannot seem to get it working
# http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=surffeaturedetector#StarFeatureDetector%20:%20public%20FeatureDetector
def STAR_detector(img, sensitivity, n_features, params):
    return cv2.xfeatures2d.StarDetector_create(responseThreshold=5, lineThresholdProjected=5)


# http://docs.opencv.org/3.1.0/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html#gsc.tab=0
# NB only descriptor!
# diffusivity
def FREAK_descriptor(img, sensitivity, keypoints, params):
    if sensitivity == 2:
        freak = cv2.xfeatures2d.FREAK_create(patternScale=10.)
    elif sensitivity == 1:
        freak = cv2.xfeatures2d.FREAK_create(patternScale=7.)
    elif sensitivity == 0:
        freak = cv2.xfeatures2d.FREAK_create(patternScale=5.)
    return freak, None

# latch = cv2.xfeatures2d.LATCH_create()
# _, desc = latch.compute(gray, kps)
