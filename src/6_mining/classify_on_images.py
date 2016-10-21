import numpy as np
import os
import fnmatch
import shutil
import sys

sys.path.append('../5_apply_model/')
from classify import classify_img
from nms import nms
import cv2
import pudb

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

if not os.path.exists("mining_set/" + filename):
    os.makedirs("mining_set/" + filename)

original_names = []
for i in range(len(names)):
    file = names[i]
    foldername = file[:-12].strip() + ".png"
    original = None
    original = filter(lambda x: foldername in x, all_filenames)
    for loc in original:
        original_names.append(loc)

for original_name in original_names:
    compliant, val, preds = classify_img(original_name + "/original.png")
    regions = nms(preds)
    img = cv2.imread(original_name + "/original.png")
    name = original_name.split("/")[-1].split(".png")[0]
    counter = 0
    for r in regions:
        counter += 1
        x = int(r[0])
        y = int(r[1])
        w = int(r[3])
        h = int(r[4])
        subimg = img[y:y + h, x:x + w]
        new_filename = './mining_set/' + name + ' - moi-' + str(counter) + '.png'
        exists = os.path.isfile(new_filename)
        add = ''
        while exists:
            add = add + 'd'
            new_filename = './mining_set/' + name + ' - moi-' + str(counter) + '_' + add + '.png'
            exists = os.path.isfile(new_filename)
        cv2.imwrite(new_filename, subimg)
