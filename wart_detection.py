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
# 8. watershed with markers set to closed edges
# 9. export wart-regions from watershed

import numpy as np
import cv2
import time

# import pudb; # this is the debugger

# scipy learn 
from sklearn.feature_extraction.image import grid_to_graph
import scipy.cluster.hierarchy as hac
from hierarchical_tweaked import AgglomerativeClusteringTreeMatrix 

def find_warts(img_path):
    # function to hierarchical cluster image
    def segment_hclustering_sklearn(image):
        small = cv2.resize(image, (0,0), fx=0.25, fy=0.25) 
 
        gray = cv2.cvtColor(small, cv2.COLOR_LUV2BGR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        X = np.reshape(gray, (-1, 1))
        connectivity = grid_to_graph(*gray.shape)
        ward = AgglomerativeClusteringTreeMatrix(n_clusters=0, linkage='ward',
                connectivity=connectivity, compute_full_tree=True)
        ward.fit(X)
 
        linkage_matrix = np.column_stack((ward.children_,ward.distance, ward.n_desc))
        max_distance =  np.amax(ward.distance)
        clusters = hac.fcluster(linkage_matrix, max_distance*0.9, 'distance')
        n_clusters = len(np.unique(clusters))
 
        label = np.reshape(clusters, gray.shape)
 
        # Plot the results on an image
        label = np.repeat(np.repeat(label, 4, axis=0), 4, axis=1)
        clusters = []
        for l in range(n_clusters):
            i_of_cluster = l+1 # first cluster (0) has label 1
            cluster = np.zeros(image.shape[:2])
            cluster[label == i_of_cluster] = 1;
            clusters.append(cluster)
 
        # maybe eliminate spatial regions containing less than minimum amount of pixels
        return clusters
 
    def get_possible_finger_clusters(clusters, image_BGR):
        img_hsv = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
 
        # D. Chai, and K.N. Ngan, "Face segmentation using skin-color map in videophone applications". IEEE Trans. on Circuits and Systems for Video Technology, 9(4): 551-564, June 1999.
        # skin_ycrcb_mint = np.array((0, 133, 77))
        # skin_ycrcb_maxt = np.array((255, 173, 127))
        # skin_mask = cv2.inRange(img_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
        # cv2.imshow("ycr mask", skin_mask)
            # total_skin = np.count_nonzero(skin_mask)
 
        skin_hue = 174 #constant
        skin_clusters = []
 
        pinkest = 0
        pinkest_level = 0
        pinkest_saturation = 0
 
        for i, cluster in enumerate(clusters):
            # get average color
            colors_i = np.nonzero(cluster)
            colors = img_hsv[colors_i]
            hues = colors[:,:1]
            saturation = colors[:,:2]
 
            average_hue = np.mean(hues)
            average_saturation = np.mean(saturation)
 
            if average_hue < 90: 
                average_hue = 179 + average_hue
 
            pink_level = (skin_hue - abs(skin_hue - average_hue)) / skin_hue
            sat_level = average_saturation / 255
 
            if pink_level > pinkest_level:
                pinkest = i
                pinkest_level = pink_level
                pinkest_saturation = sat_level
 
        skin_clusters.append(clusters[pinkest])
        return skin_clusters
 
    # start timer to time execution
    st = time.time()
 
    print("0.00s - 1. open image and convert to luv...")
 
    img = cv2.imread(img_path)
 
    # width/height has to be able to divide by 4
    if img.shape[0] % 4 != 0 or img.shape[1] % 4 != 0:
        shape = np.array([img.shape[1], img.shape[0]])
        dsize = np.round(shape/4)*4
        img = cv2.resize(img, dsize=tuple(dsize), fx=0, fy=0) 
 
    # http://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
    # maybe do a CLAHE to normalize brightness?
 
    # luv is a better color space to identify skin colors [citation needed] 
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
 
    # parameters for mean shift
    hs = 16 # spatial bandwidth/window size
    hr = 16 # range (color) bandwidth/window size
 
    print("%.2f" % (time.time() - st) + "s" + " - 2. execute mean shift filtering")
 
    shifted = cv2.pyrMeanShiftFiltering(luv, hs, hr, 0)
 
    print("%.2f" % (time.time() - st) + "s" + " - 3. segment mean shift with hierarchical clustering")
 
    clusters = segment_hclustering_sklearn(shifted)
    shifted = cv2.cvtColor(shifted, cv2.COLOR_LUV2BGR)
 
    print("%.2f" % (time.time() - st) + "s" + " - 4. find most skin-colored cluster (finger)")
 
    skin_clusters = get_possible_finger_clusters(clusters, shifted)
 
    mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask[skin_clusters[0] == 0] = 0 
    mask[skin_clusters[0] == 1] = 255 
 
    #dilate the mask, we do not need the edges of the skin
    kernel = np.ones((5,5),np.uint8)
    mask_erosion = cv2.erode(mask,kernel,iterations = 1) 
 
    print("%.2f" % (time.time() - st) + "s" + " - 5. bilateral blur inside of finger")
 
    blur = cv2.bilateralFilter(img,9,75,75)
 
    print("%.2f" % (time.time() - st) + "s" + " - 6. canny edge detection in finger")
 
    canny = cv2.Canny(blur, 20, 80, 3);
    canny[np.where(mask_erosion==0)] = 0 
    # cv2.imshow("canny1", canny)
 
    # ret,thresh = cv2.threshold(blur,127,255,0)
    # im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0,255,0), 2)
    # cv2.imshow('Found finger', img)
 
    print("%.2f" % (time.time() - st) + "s" + " - 7. dilate + erode edges to 'close' nearby edges")
 
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(canny,kernel,iterations=2)
    erosion = cv2.erode(dilation,kernel,iterations=2)
 
    print("%.2f" % (time.time() - st) + "s" + " - 8. watershed with markers set to closed edges")
    # Is watershed needed at all?
 
    sure_bg = cv2.dilate(canny,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(erosion,cv2.DIST_L2,5)
    cv2.normalize(dist_transform, dist_transform, 0.0, 1.0, cv2.NORM_MINMAX);
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    # cv2.imshow("sure_fg", sure_fg)
 
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
 
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
 
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
 
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    watershed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    watershed[markers == -1] = 255 
    watershed[markers != -1] = 0 
 
    # watershed also finds border around image, ignore it:
    watershed[0] = 0
    watershed[:,0] = 0
    watershed[img.shape[0]-1] = 0
    watershed[:,[img.shape[1]-1]] = 0
 
    # dilate watershed results:
    kernel = np.ones((15,15),np.uint8)
    dilation = cv2.dilate(watershed,kernel,iterations=2)
    erosion = cv2.erode(dilation,kernel,iterations=1)
 
    print("%.2f" % (time.time() - st) + "s" + " - 9. export wart-regions from watershed")
 
    # find results
    im2, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
 #   for c in contours:
 #       x,y,w,h = cv2.boundingRect(c)
 #       cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
 #

    print "%.2f" % (time.time() - st) + "s" + " - elapsed time" 

    return contours



