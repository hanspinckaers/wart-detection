# -*- coding: utf-8 -*-
"""
Author: Yuri Mejia Miranda and Hans Pinckaers
Date: 20160725
Goal: This script loops through the images in output/ and present a GUI to select the correctly identified warts.
Info:
"""

import os
import fnmatch
import sys
import cv2
import numpy as np
import math
import shutil
import pudb

matches = {}


def classifyImg(original, dirname, imgname):
    imgpath = os.path.join(dirname, imgname)
    img = cv2.imread(imgpath)
    if img is None:
        return ""

    # add padding to image for hstack
    height_diff = original.shape[0] - img.shape[0]
    width_diff = original.shape[1] - img.shape[1]

    top = int(math.floor(height_diff / 2.))
    bottom = int(math.ceil(height_diff / 2.))
    left = int(math.floor(width_diff / 2.))
    right = int(math.ceil(width_diff / 2.))

    bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    hstacked = np.hstack((original, bordered))
    cv2.destroyAllWindows()
    cv2.imshow(dirname, hstacked)
    k = cv2.waitKey()
    if k == 113:  # q
        quit()
    if k == 112:  # p
        print "go back"
        return "p"
    if k == 119:  # w
        print "save as wart"
        return "w"
    if k == 99:  # c
        print "save wart with cream"
        return "c"
    if k == 110:  # n
        print "no wart"
        return "n"
    if k == 100:  # d
        print "dubious"
        return "d"
    if k == 115:
        print "save original"
        return "s"

    print k
    return "-"


names = []
for root, dirnames, filenames in os.walk("../../images/train_set"):
    for filename in fnmatch.filter(filenames, '*.png'):
        names.append(filename)

np.random.seed(0)

all_filenames = []
for root, dirnames, filenames in os.walk("../../results/naive_algorithm_per_img"):
    for filename in fnmatch.filter(filenames, '*.png'):
        if root not in all_filenames:
            all_filenames.append(root)

original_names = []

images = []
for root, dirnames, filenames in os.walk("../../images/mining_set"):
    for filename in fnmatch.filter(filenames, '*.png'):
        images.append(filename)

i = 0
while i < len(images):
    imgfilename = images[i]

    original_names = []
    foldername = imgfilename.split('moi')[0][:-2].strip() + ".png"
    dirname = None
    dirname = filter(lambda x: foldername in x, all_filenames)
    original = cv2.imread(dirname[0] + "/original.png")
    k = "-"
    while k == "-" or k == "s":
        k = classifyImg(original, './mining_set', imgfilename)

    if k == "p":
        rm_arr = images[i - 1]
        if len(rm_arr) > 1:
            os.remove(rm_arr[1])
            images[i - 1] = rm_arr[0]
            i -= 1
            continue

    classified_as = ""

    if k == "":
        i += 1
        continue
    if k == "w":
        classified_as = "warts"
    if k == "d":
        classified_as = "dubious"
    if k == "n":
        classified_as = "negatives"
    if k == "c":
        classified_as = "warts_cream"

    old_path = os.path.join('./mining_set', imgfilename)
    new_path = os.path.join('classified_mining', classified_as)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    new_path = os.path.join(new_path, imgfilename)
    shutil.copyfile(old_path, new_path)

    images[i] = [imgfilename, new_path]

    i += 1
    print str(i) + " - " + str(len(images))
