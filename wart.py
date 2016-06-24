import numpy as np
import cv2

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

img = cv2.imread("wart-on-skin.png")

# pyrMeanShiftFiltering average color quite pretty (look up theory!)
shifted = cv2.pyrMeanShiftFiltering(img, 21, 30)
cv2.imshow('shifted', shifted)

# canny edge detection with automatic upper/lower bounds
edges = auto_canny(shifted)
cv2.imshow('edges', edges)

# connect loose edges
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
mask = cv2.dilate(edges, kernel, iterations = 1)
# mask = cv2.erode(mask, kernel, iterations = 1)

# convert mask
inverted = cv2.bitwise_not(mask)
cv2.imshow('inverted', inverted)

# find contours
im2, contours, hierarchy = cv2.findContours(inverted, cv2.RETR_LIST, cv2.RETR_TREE)

# try to find finger with contour
for contour in contours:
    # threshold original image using contour
    # find median color per channel, use .nonzero
    # find median color most close to skin!
    # find biggest contour
    # show finger
    for c in contours:
        if cv2.contourArea(c) > 300 and cv2.contourArea(c) < 90000:
          (x,y,w,h) = cv2.boundingRect(c)
          cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2)

# draw finger on img
# cv2.drawContours(img, contours, -1, (0,255,0), -1)
cv2.imshow('img', img)

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

