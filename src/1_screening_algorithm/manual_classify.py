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
# import pudb

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


def newImgName(dirname, filename):
    last = os.path.split(dirname)[-1]
    name = os.path.splitext(last)[0]
    name += " - " + filename
    return str(name)


for root, dirnames, filenames in os.walk(sys.argv[1]):
    for filename in fnmatch.filter(filenames, '*.png'):
        folder = root
        if "original" in filename:
            continue
        if folder in matches:
            matches[folder].append(filename)
        else:
            matches[folder] = [filename]

last_img_path = ""  # needed to delete file when going back by pressing 'p'
images = []

for dirname in matches:
    for imgname in matches[dirname]:
        images.append([dirname, imgname])
                       
i = 0
while i < len(images):
    img_arr = images[i]
    ori_path = os.path.join(img_arr[0], "original-locs.png")
    
    original = cv2.imread(ori_path)
    k = "-"
    while k == "-" or k == "s":
        k = classifyImg(original, img_arr[0], img_arr[1])

        # when pressing s: save original for later analysis
        if k == "s":
            new_path_ori = os.path.join('classified', "originals")
            if not os.path.exists(new_path_ori):
                os.makedirs(new_path_ori)

            new_path_ori_file = os.path.join(new_path_ori, newImgName(img_arr[0], "original-locs.png"))
            shutil.copyfile(ori_path, new_path_ori_file)

            ori_no_squares_path = os.path.join(img_arr[0], "original.png")
            new_path_ori_file = os.path.join(new_path_ori, newImgName(img_arr[0], "original.png"))
            shutil.copyfile(ori_no_squares_path, new_path_ori_file)

    if k == "p":
        rm_arr = images[i - 1]
        if len(rm_arr) > 1:
            os.remove(rm_arr[2])
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

    saved = newImgName(img_arr[0], img_arr[1])
    old_path = os.path.join(img_arr[0], img_arr[1])
    new_path = os.path.join('classified', classified_as)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    new_path = os.path.join(new_path, saved)
    shutil.copyfile(old_path, new_path)

    images[i] = [img_arr[0], img_arr[1], new_path]

    i += 1
    print str(i) + " - " + str(len(images))
