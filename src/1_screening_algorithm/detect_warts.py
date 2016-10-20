# -*- coding: utf-8 -*-
"""
Author: Yuri Mejia Miranda and Hans Pinckaers
Date: 20160725
Goal: Detecting possible regions with warts to create training set for training HAAR classifiers.
Info:
    Run on folder: python detect-warts.py dir images
    Run on folder with parallelism: python detect-wart.py dir images [number-of-processes]
    Run on file: python detect-wart.py images/wart-on-skin.png
"""

from find_wart import find_warts
import cv2
import os
import fnmatch
import sys
import multiprocessing
import time

matches = []
output_result = False
params = {}

# default settings
kernel_size = (9, 9)  # mean_shift kernel
morph_iterations = 2  # for closing operation
blur_size = [15, 15, 15]  # blur image with certain size
canny_params = [5, 20, 40]  # canny 
dist_threshold = 8  # threshold for the distance converted canny image
laplacian = (5, 5)
dilation_iterations = 2
percentile = 75

if sys.argv[1] == "dir":
    for root, dirnames, filenames in os.walk(sys.argv[2]):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            matches.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.png'):
            matches.append(os.path.join(root, filename))
    output_result = True

else:
    matches.append(sys.argv[1])
    output_result = True

print matches


def wart_worker(filename):
    global params
    print filename

    regions = find_warts(filename, output_result, **params)

    img = cv2.imread(filename)
    filename = filename.replace("../", "/")

    img_with_regions = img.copy()

    if not os.path.exists("output/" + filename):
        os.makedirs("output/" + filename)

    region_size = int(img.shape[0] / 5)

    counter = 0

    for c in regions:
        x, y, w, h = cv2.boundingRect(c)
        counter += 1
        if w < region_size:
            x = max(int(x + w / 2 - region_size / 2), 0)
            w = region_size
        else:
            x = max(int(x - region_size / 4), 0)
            w += region_size / 2
        if h < region_size:
            y = max(int(y + h / 2 - region_size / 2), 0)
            h = region_size
        else:
            y = max(int(y - region_size / 4), 0)
            h += region_size / 2

        # if output_result:
        cv2.rectangle(img_with_regions, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imwrite("output/" + filename + "/roi-" + str(counter) + ".png", img[y:y + h, x:x + w])
        txtfilename = "output/" + filename + "/roi-" + str(counter) + ".csv"
        target = open(txtfilename, 'w')
        target.write(str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "\n")
        target.write(str(c))
        target.close()

    cv2.imwrite("output/" + filename + "/original.png", img)
    cv2.imwrite("output/" + filename + "/original-locs.png", img_with_regions)

    if output_result:
        cv2.namedWindow(filename)

# ENABLE TRACKBARS HERE TO TWEAK SETTINGS

        cv2.createTrackbar('1_kernel', filename, kernel_size[0], 150, callbackKernel)
        cv2.createTrackbar('2_morph', filename, morph_iterations, 20, callbackMorph)
        cv2.createTrackbar('3_blur', filename, blur_size[1], 300, callbackBlur)
        # cv2.createTrackbar('4_canny1', filename, canny_params[0], 200, callbackCanny1)
        # cv2.createTrackbar('5_canny2', filename, canny_params[1], 200, callbackCanny2)
        # cv2.createTrackbar('6_canny_a', filename, canny_params[2], 40, callbackCannyAperture)
        cv2.createTrackbar('7_theshold', filename, int(dist_threshold), 30, callbackThreshold)
        cv2.createTrackbar('8_dilation_iterations', filename, 3, 20, callbackDilationIterations)
        cv2.createTrackbar('9_laplacian', filename, 3, 20, callbackLaplacian)
        cv2.createTrackbar('10_percentile', filename, 50, 100, callbackPercentile)

        cv2.imshow(filename, img_with_regions)

    return


def callbackKernel(value):
    global kernel_size, params
    kernel_size = (value, value)
    resetParams()


def callbackMorph(value):
    global morph_iterations, params
    morph_iterations = value
    resetParams()


def callbackBlur(value):
    global blur_size, params
    blur_size = [blur_size[0], value, value]
    resetParams()


def callbackCanny1(value):
    global canny_params
    canny_params = [value, canny_params[1], canny_params[2]]
    resetParams()


def callbackCanny2(value):
    global canny_params
    canny_params = [canny_params[0], value, canny_params[2]]
    resetParams()


def callbackCannyAperture(value):
    global canny_params
    canny_params = [canny_params[0], canny_params[1], value]
    resetParams()


def callbackThreshold(value):
    global dist_threshold
    dist_threshold = value
    resetParams()


def callbackDilationIterations(value):
    global dilation_iterations
    dilation_iterations = value
    resetParams()


def callbackLaplacian(value):
    global laplacian
    laplacian = (value, value)
    resetParams()


def callbackPercentile(value):
    global percentile
    percentile = value
    resetParams()


def resetParams():
    global params
    params = {"percentile": percentile, "kernel_size": kernel_size, "morph_iterations": morph_iterations, "blur_size": blur_size, "canny_params": canny_params, "dist_threshold": dist_threshold, "dilation_iterations": dilation_iterations, "laplacian": laplacian}
    print params
    for filename in matches:
        wart_worker(filename)


params = {"percentile": percentile, "kernel_size": kernel_size, "morph_iterations": morph_iterations, "blur_size": blur_size, "canny_params": canny_params, "dist_threshold": dist_threshold, "dilation_iterations": dilation_iterations, "laplacian": laplacian}

if len(sys.argv) > 3:
    # start timer to time execution
    st = time.time()
    p = multiprocessing.Pool(int(sys.argv[3]))
    p.map(wart_worker, matches)
    print("Total: %.2f" % (time.time() - st) + "s\n")
else:
    st = time.time()

    for filename in matches:
        wart_worker(filename)

    print("Total: %.2f" % (time.time() - st) + "s\n")

if output_result:
    k = cv2.waitKey()
    cv2.destroyAllWindows()

quit()
