import numpy as np
import cv2
import time
# import pudb; 
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as dis
from hierarchical_tweaked import AgglomerativeClusteringTreeMatrix 

st = time.time()

img = cv2.imread("wart-on-skin.png")

# http://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/24341809#24341809
# maybe do a CLAHE to normalize brightness?
luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

hs = 16 # spatial bandwidth/window size
hr = 16 # range (color) bandwidth/window size

# pyrMeanShiftFiltering average color quite pretty
shifted = cv2.pyrMeanShiftFiltering(luv, hs, hr, 0)

def img_with_coordinates_1d(image):
    # joint spatial color domain
    # create a 1D list of all pixels with in the 3rd and 4th dimension spatial information:
    #   l   u   v,  x   y
    # [
    # [119 103 151   0   0]
    # [120 103 151   1   0]
    # [120 103 151   2   0]
    # ..., 
    # [169 105 150 253 255]
    # [168 105 150 254 255]
    # [168 105 150 255 255]
    # ]
    start_time = time.time()

    x = np.arange(shifted.shape[1])
    y = np.arange(shifted.shape[0])
    xv, yv = np.meshgrid(x, y)
    join = np.insert(shifted, 3, xv, axis=2)
    join = np.insert(join, 4, yv, axis=2)
    img_1d = join.reshape(-1, join.shape[-1])
    return img_1d

## make segments based on mean shift filtering using hierchical clustering
def segment_hclustering_sklearn(image, name, original, n_clusters):
    small = cv2.resize(image, (0,0), fx=0.25, fy=0.25) 

    gray = cv2.cvtColor(small, cv2.COLOR_LUV2BGR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    
    X = np.reshape(gray, (-1, 1))
    print("Compute structured hierarchical clustering...")
    
    connectivity = grid_to_graph(*gray.shape)
    n_clusters = n_clusters  # number of regions
    # pu.db
    ward = AgglomerativeClusteringTreeMatrix(n_clusters=n_clusters, linkage='ward',
            connectivity=connectivity, compute_full_tree=True)
    ward.fit(X)

    linkage_matrix = np.column_stack((ward.children_,ward.distance, ward.n_desc))
    max_distance =  np.amax(ward.distance)
    clusters = hac.fcluster(linkage_matrix, max_distance*0.9, 'distance')
    n_clusters = len(np.unique(clusters))

    label = np.reshape(clusters, gray.shape)

    # Plot the results on an image
    label = np.repeat(np.repeat(label, 4, axis=0), 4, axis=1)
    for l in range(n_clusters):
        new_img = original.copy()
        cluster = l+1 
        new_img[label != cluster] = [0,0,0]
        cv2.imshow(name+str(cluster), new_img)
    im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = numpy.array((0, 133, 77))
    skin_ycrcb_maxt = numpy.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    #dm = dis.pdist(img_1d, dist)
    #z = hac.linkage(img_1d, metric=dm)

    #print("--- %s seconds ---" % (time.time() - start_time))

    # loop hues and make connect pixels into contour if smaller than hs and hr
    # use inRange of openCV and then findContours?

    # eliminate spatial regions containing less than minimum amount of pixels
    return image

#TODO: # http://stackoverflow.com/questions/26851553/sklearn-agglomerative-clustering-linkage-matrix

def segment_hclustering_scipy(image):
    img_1d = img_with_coordinates_1d(image)

    def dist(u, v):
        u_luv = u[0:4]
        u_xy = u[3:]

        v_luv = v[0]
        v_xy = v[4:]
        xs = np.linalg.norm(u_xy-v_xy) # spatial difference
        xr = np.linalg.norm(u_luv-v_luv) # range difference 
        
        return xs/hs + xr/hr # should we use the kernel here?

    dm = dis.pdist(img_1d, dist)
    print dm

img1 = segment_hclustering_sklearn(shifted, "mean shift", img, 4)
# segment_hclustering_scipy(shifted)
print "Elapsed time: ", time.time() - st

shifted = cv2.cvtColor(shifted, cv2.COLOR_LUV2BGR)
cv2.imshow('shifted', shifted)

#img2 = segment_hclustering(img, "original", 10, 10)

k = cv2.waitKey()
cv2.destroyAllWindows()

quit()

