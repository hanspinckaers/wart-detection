import numpy as np
import cv2
import time
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as dis
from collections import defaultdict

img = cv2.imread("wart-on-skin.png")

# http://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
# maybe do a CLAHE to normalize brightness?
luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

hs = 32 # spatial bandwidth/window size
hr = 16 # range (color) bandwidth/window size

# pyrMeanShiftFiltering average color quite pretty
shifted = cv2.pyrMeanShiftFiltering(luv, hs, hr, 0)

cv2.imshow('shifted', shifted)

# function should work with all image representation
# must be 3 channels though
def unique_colors(image):
    colors = image.reshape(image.shape[0]*image.shape[1],image.shape[2])
    colors = np.vstack({tuple(row) for row in colors})
    return colors
    
#r_l, r_u en r_v define a region around 
def min_values_in_range(arr, r_l, r_u, r_v):
    # arr = np.array([[155,105,150],[155+13,105+13,150+13],[155+14,105+14,150+14],[155+27,105+27,150+27],[193,116,158],[122,103,151],[133,104,150],[163,105,150],[108,102,150],[134,104,151],[121,103,151]])
    n_of_values = np.zeros(len(arr))
    indices_of_values = []
    colors = []
    merged_colors = []
    for i, a in enumerate(arr):
        l = np.logical_and(arr[:,0]<=(a[0]+r_l/2), arr[:,0]>=(a[0]-r_l/2))
        u = np.logical_and(arr[:,1]<=(a[1]+r_u/2), arr[:,1]>=(a[1]-r_u/2))
        v = np.logical_and(arr[:,2]<=(a[2]+r_v/2), arr[:,2]>=(a[2]-r_v/2))
        values = np.logical_and.reduce((l,u,v))
        indices = np.where(values)
        n_of_values[i] = len(indices[0])
        indices_of_values.append(indices[0].tolist())

    filter_indices_of_values = indices_of_values[:]
    ranges = []

    # find max n_of_values
    max_index = n_of_values.argmax()
    max_count = n_of_values[max_index]
    while max_count > 0:
        colors.append(arr[max_index])
        merged_colors.append([arr[max_index]])
        ranges.append(arr[indices_of_values[max_index]])
        rm_indices = set(filter_indices_of_values[max_index])
        for i, a in enumerate(filter_indices_of_values):
            new_indices = [x for x in filter_indices_of_values[i] if x not in rm_indices]
            filter_indices_of_values[i] = new_indices
            n_of_values[i] = len(new_indices)
        max_index = n_of_values.argmax()
        max_count = n_of_values[max_index]

    # we could use graph theory here: calculate connected graph
    # 
    # ranges = [[1,2,3],[2,3],[4,5],[5],[5,6]]
    # merged_colors = [[0],[1],[2],[3],[4]]
    merged = True
    while merged == True:
        merged = False
        for i, c_range in enumerate(ranges):
            for j, r in enumerate(ranges):
                if i==j:
                    continue
                concat_range = np.concatenate((c_range, r))
                new_range = np.vstack({tuple(row) for row in concat_range})
                if (len(new_range) < len(concat_range)):
                    merge_colors = merged_colors[i]+merged_colors[j]
                    merged_colors[i] = merge_colors
                    ranges[i] = new_range
                    del ranges[j]
                    del merged_colors[j]
                    merged = True
                    break
            if merged:
                break
   
    return (colors, merged_colors)

def segment_threshold(image, hs, hr):
    colors = unique_colors(image)
    luv_range = 24 
    min_colors, overlapping = min_values_in_range(colors, luv_range, luv_range, luv_range)
    print len(min_colors)
    contours = []
    for color in min_colors:
        color_range = cv2.inRange(image, color-luv_range/2, color+luv_range/2)
        # find contours
        # label it using scipy 
        # http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html
        # http://stackoverflow.com/questions/9440921/identify-contiguous-regions-in-2d-numpy-array
        # cv2.imshow(str(color), color_range)

    # find overlapping contours/pixels
    # decide on color/spatial value to which contour they belong
    for color_arr in overlapping:
        thresholds = np.zeros((image.shape[0], image.shape[1]))
        for color in color_arr:
            color_range = cv2.inRange(image, color-luv_range/2, color+luv_range/2)
            boolean_matrix = (color_range > 0)
            thresholds += boolean_matrix

        boolean_thresholds = (thresholds>0)
        masked_img = cv2.bitwise_and(image, image, mask=boolean_thresholds.astype(np.uint8))
        cv2.imshow(str(color_arr), masked_img)

segment_threshold(shifted, 10, 10)
k = cv2.waitKey()
cv2.destroyAllWindows()

quit()

