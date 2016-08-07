# from tsne import bh_sne
import numpy as np
from skimage.transform import resize
# import cv2
import pudb
import math
import multiprocessing
# import pymunk
# from matplotlib import pyplot as plt
from functools import partial
import time

def calc_resolved_vector(i, coords):
    coord = coords[i]
    img_res = 100
    overlap_indices = coords[(np.abs(coords[:,0] - coord[0]) < img_res) & (np.abs(coords[:,1] - coord[1]) < img_res)]
    overall_vector = np.array([0,0], dtype=float)
    if i % 1000 == 0:
        print str(i)

    for collision_coord in overlap_indices:
        # if i == idx:
        #    continue

        # collision_coord = coords[idx]

        vec_diff = collision_coord[:2] - coord[:2]  # difference between the two coordinates
        max_diff = np.abs(vec_diff).max().astype(float)

        # we should add a very small random vec instead
        if max_diff == 0:
            continue
            # vec = np.array([1, 0])
            # max_diff = 1

        vec = vec_diff / max_diff
        vec = vec * img_res  # this is diff of the location of i without collision with idx

        vec_diff = (vec - vec_diff) / 2.  # calculate the diff to current idx location

        overall_vector += vec_diff

    if len(overlap_indices) > 1:
        overall_vector = overall_vector / (len(overlap_indices) - 1)

    # correct_loop[i][:2] = coord[:2] - overall_vector  # not sure why this should be minus ...

    return np.ceil(overall_vector).astype(int)
        
def gray_to_color(img):
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    return img


def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, img.shape[:2])
    if min([w, h]) != size:
        if h <= w:
            img = resize(img, (int(round((h / w) * size)), int(size)))
        else:
            img = resize(img, (int(size), int(round((w / h) * size))))
    return img


def image_scatter(f2d, images, img_res, res=6000, cval=1.):
    """
    Embeds images via tsne into a scatter plot.

    Parameters
    ---------
    features: numpy array
        Features to visualize

    images: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    img_res: float or int
        Resolution to embed images at

    res: float or int
        Size of embedding image in pixels

    cval: float or numpy array
        Background color value

    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    # features = np.copy(features).astype('float64')
    images = [gray_to_color(image) for image in images]
    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    # f2d = bh_sne(features)

    xx = f2d[:, 0]
    yy = f2d[:, 1]

    
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()

    # Fix the ratios
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = sx / float(sy) * res
        res_y = res
    else:
        res_x = res
        res_y = sy / float(sx) * res

    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)

    # make new coordinates
    new_coords = np.ones((len(xx), 3), dtype=int)

    for i, (x, y) in enumerate(zip(xx, yy)):
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        new_coords[i][0] = x_idx
        new_coords[i][1] = y_idx

    print "Resolve overlapping:"

    new_coords[:, 2] = np.arange(len(new_coords))  # add column to remember order 3
    correct_loop = new_coords.copy()

    running = True

    overall_movement = 0
    overall_n = 0
    j = 0
    while running:
        running = True
        print "Resolving " + str(j) + "x"

        i = 0
        while i < len(new_coords):
            if i % 1000 == 0:
                print str(i)
            coord = new_coords[i]
            overlap_indices = []

            # this takes the longest time
            overlap_indices = np.where((np.abs(new_coords[:,0] - coord[0]) < img_res) & (np.abs(new_coords[:,1] - coord[1]) < img_res))[0]
            overall_vector = np.array([0,0], dtype=float)
            for idx in overlap_indices:
                if i == idx:
                    continue

                collision_coord = new_coords[idx]

                vec_diff = collision_coord[:2] - coord[:2]  # difference between the two coordinates
                max_diff = np.abs(vec_diff).max().astype(float)

                # we should add a very small random vec instead
                if max_diff == 0:
                    continue
                    # vec = np.array([1, 0])
                    # max_diff = 1

                vec = vec_diff / max_diff
                vec = vec * img_res  # this is diff of the location of i without collision with idx

                vec_diff = (vec - vec_diff) / 2.  # calculate the diff to current idx location

                overall_vector += vec_diff

                overall_movement += np.abs(vec_diff)
                overall_n += 1

            if len(overlap_indices) > 1:
                overall_vector = overall_vector / (len(overlap_indices) - 1)

            correct_loop[i][:2] = (coord[:2] - overall_vector).astype(int)  # not sure why this should be minus ...

            i += 1
        j += 1

        print "- avg movement: " + str(overall_movement / overall_n)

        if np.sum(overall_movement / overall_n) < 4:
            running = False
        
        new_coords = correct_loop.copy()

    n_x_min, n_y_min, _ = new_coords.min(axis=0)
    new_coords[:,0] -= n_x_min
    new_coords[:,1] -= n_y_min
    n_x_max, n_y_max, _ = new_coords.max(axis=0)

    print "--- Making plot: (" + str(n_x_max) + "," + str(n_y_max) + ") ---"

    canvas = np.ones((n_x_max + max_width, n_y_max + max_height, 3)) * cval

    for x, y, image in zip(new_coords[:,0], new_coords[:,1], images):
        w, h = image.shape[:2]
        canvas[x:x + w, y:y + h] = image

    return canvas
