# -*- coding: utf-8 -*-
"""
Author: Yuri Mejia Miranda and Hans Pinckaers
Date: 20160718
Goal: Detecting possible regions with warts to create training set for training HAAR classifiers.
Info:
img = "path to image"
"""

# Overview:
# 1. open image and convert to luv
# 2. execute mean shift filtering
# 3. segment mean shift with hierarchical clustering
# 4. find most skin-colored cluster (finger)
# 5. bilateral blur inside of finger
# 6. canny edge detection in finger
# 7. dilate + erode edges to 'close' nearby edges
# 8. export wart-regions

import numpy as np
import cv2
import time
import os.path
# import pudb; # this is the debugger
from utilities import segment_hclustering_sklearn, get_possible_finger_clusters


def find_warts(img_path, output, kernel_size=(8, 8),
               laplacian=(8, 8), morph_iterations=2, blur_size=[7, 60, 60],
               canny_params=[20, 80, 5], dist_threshold=0.5,
               dilation_iterations=3, percentile=50):
    # start timer to time execution
    st = time.time()

    print(img_path + " 0.00s - 1. open image and convert to luv...")

    img = cv2.imread(img_path)

    # load cache of mask if available
    if os.path.isfile("cache/" + img_path + "/mask.png"):
        mask = cv2.imread("cache/" + img_path + "/mask.png")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        print(img_path + " %.2f" % (time.time() - st) + "s" + " - 2, 3 ,4. loading cached mask")
    else:
        if output:
            cv2.imshow("original img", img)

        # width/height has to be able to divide by 4
        if img.shape[0] % 4 != 0 or img.shape[1] % 4 != 0:
            shape = np.array([img.shape[1], img.shape[0]])
            dsize = np.round(shape / 4) * 4
            img = cv2.resize(img, dsize=tuple(dsize), fx=0, fy=0)

        # http://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809

        # luv is a better color space to identify skin colors [citation needed]
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

        # parameters for mean shift
        hs = 16  # spatial bandwidth/window size
        hr = 16  # range (color) bandwidth/window size

        print(img_path + " %.2f" % (time.time() - st) + "s" + " - 2. execute mean shift filtering")

        shifted = cv2.pyrMeanShiftFiltering(luv, hs, hr, 0)

        print(img_path + " %.2f" % (time.time() - st) + "s" + " - 3. segment mean shift with hierarchical clustering")

        clusters = segment_hclustering_sklearn(shifted, img_path)

        print(img_path + " %.2f" % (time.time() - st) + "s" + " - 4. find most skin-colored cluster (finger)")

        shifted = cv2.cvtColor(shifted, cv2.COLOR_LUV2BGR)

        if output:
            cv2.imshow("shifted", shifted)

        skin_clusters = get_possible_finger_clusters(clusters, shifted, img_path)
        
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask[skin_clusters[0] == 0] = 0
        mask[skin_clusters[0] == 1] = 255
        
        if not os.path.exists("cache/" + img_path):
            os.makedirs("cache/" + img_path)
        
        # fill holes in mask
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        cv2.imwrite("cache/" + img_path + "/mask.png", mask)

        if output:
            cv2.imshow("mask", mask)

    # dilate the mask, we do not need the edges of the skin
    kernel = np.ones((5, 5), np.uint8)
    mask_erosion = cv2.erode(mask, kernel, iterations=1)

    print(img_path + " %.2f" % (time.time() - st) + "s" + " - 5. bilateral blur inside of finger")

    # blur = cv2.bilateralFilter(img,blur_size[0],blur_size[1], blur_size[2])

    # d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
    # sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
    # sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .

    # primitive implementation to remove hard shadows
    blur_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(blur_hsv)
    blur_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h2, s2, l2 = cv2.split(blur_hsl)

    print(img_path + " %.2f" % (time.time() - st) + "s" + " - 6. canny edge detection in finger")

    oneven = int(round(blur_size[1] / 2) * 2 + 1)
    blur = cv2.GaussianBlur(s, (oneven, oneven), 0)
    blur_v = cv2.GaussianBlur(v, (oneven, oneven), 0)
    blur = np.uint8(blur)
    blur_v = np.uint8(blur_v)

    nonzeros = np.nonzero(blur)
    nonzeros = blur[nonzeros]
    s_above_percentile = np.percentile(nonzeros, 50)

    _, black_thresh = cv2.threshold(blur_v, 100, 255, cv2.THRESH_BINARY)

    black_mask = np.ones(blur.shape)
    black_mask[blur <= s_above_percentile] = 0
    black_mask[np.where(black_thresh == 0)] = 0

    canny = cv2.Canny(blur, canny_params[0], canny_params[1], canny_params[2])
    canny[np.where(mask_erosion == 0)] = 0
    canny[np.where(black_mask == 0)] = 0

    if output:
        cv2.imshow(img_path + "canny", canny)
 
    print(img_path + " %.2f" % (time.time() - st) + "s" + " - 7. dilate + erode edges to 'close' nearby edges")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilation = cv2.dilate(canny, kernel, iterations=dilation_iterations)

    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dilation, contours, -1, 255, thickness=cv2.FILLED)

    # if output:
    #   cv2.imshow("dilation", dilation)

    dist_transform = cv2.distanceTransform(dilation, cv2.DIST_L2, 5)

    cv2.normalize(dist_transform, dist_transform, 0, 1., cv2.NORM_MINMAX)
    # print np.histogram(dist_transform, bins=10, range=(0,1))

    # if output:
    #    cv2.imshow(img_path+"dist_transform", dist_transform)

    thres_start = 0.2
    n_contours = 999
    while n_contours > 3 and thres_start < 1.1:
        # pu.db
        ret, sure_fg = cv2.threshold(dist_transform, thres_start, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        kernel = np.ones((10, 10), np.uint8)
        dilation = cv2.dilate(sure_fg, kernel, iterations=2)

        # find results
        im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_contours = len(contours)

        thres_start += 0.05

    # marker_image = np.zeros(dilation.shape)

    print img_path + " %.2f" % (time.time() - st) + "s" + " - elapsed time"

    return contours