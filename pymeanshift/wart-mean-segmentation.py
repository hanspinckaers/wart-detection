import numpy as np
import cv2
import pymeanshift as pms

img = cv2.imread("wart-on-skin.png")

(segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=6, 
                                                              range_radius=4.5, min_density=300)
print("done")

print(labels_image)

# find contours
# im2, contours, hierarchy = cv2.findContours(inverted, cv2.RETR_LIST, cv2.RETR_TREE)

# try to find finger with contour
# threshold original image using contour
# find median color per channel, use .nonzero
# find median color most close to skin!
# find biggest contour
# show finger

# draw finger on img
# cv2.drawContours(img, contours, -1, (0,255,0), -1)
cv2.imshow('segmented_image', segmented_image)
cv2.imshow('img', img)
cv2.imshow('labels_image', labels_image)

# dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L1, 3)
# ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(inverted,sure_fg)
# cv2.imshow('sure_fg', sure_fg)

k = cv2.waitKey()
cv2.destroyAllWindows()

quit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)

blurred = cv2.GaussianBlur(cl1, (15, 15), 0)

gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', gray)

ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('thres', thresh)

# apply a series of erosions and dilations to the mask
# using an elliptical kernel
dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
ret, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(inverted,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
 
#Add one to all labels so that sure background is not 0, but 1
markers = markers+1
    
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(shifted, markers)
img[markers == -1] = [255,0,0]

cv2.imshow('dist_transform', sure_fg)
cv2.imshow('img', img)

# cv2.imshow('erode', skinMask)

# cv2.imshow('thresh', thresh)

# cv2.imshow('edges', edges)

# skin boundaries

