      #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #gray = np.zeros((gray.shape[0], gray.shape[1])) # init 2D numpy array

    #def weightedAverage(pixel):
    #     return 0.000*pixel[0] + 0.500*pixel[1] + 0.500*pixel[2]

    ## get row number
    #for rownum in range(len(img)):
    #    for colnum in range(len(img[rownum])):
    #        gray[rownum][colnum] = weightedAverage(img[rownum][colnum])

 
    # watershed on contours

    for c in contours:
        # center of contour
        M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

        cv2.circle(marker_image, (cX, cY), 15, (255, 255, 255), -1)

    cv2.imshow("marker_image", marker_image)

    img_watershed = img.copy()

    # noise removal
    kernel = np.ones((30,30), np.uint8)
     
     # sure background area
    sure_bg = cv2.dilate(marker_image, kernel,iterations=6)
     
    # Finding unknown region
    #sure_fg = np.uint8(marker_image)
    unknown = cv2.subtract(sure_bg,marker_image)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
     
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
     
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(blur, markers)
    img_watershed[markers == -1] = [255,0,0]

    # circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=35, param2=30,minRadius=10,maxRadius=100)

#     if circles is not None:
# 	# convert the (x, y) coordinates and radius of the circles to integers
# 	circles = np.round(circles[0, :]).astype("int")
#  
# 	# loop over the (x, y) coordinates and radius of the circles
# 	for (x, y, r) in circles:
#             # draw the circle in the output image, then draw a rectangle
#             # corresponding to the center of the circle
#             cv2.circle(s2, (x, y), r, (0, 255, 0), 4)
#             cv2.rectangle(s2, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#  
# 	# show the output image
# 	cv2.imshow("output", s2) 


    # gray = equalizeHistWithMask(r, mask)

    # gray = np.zeros((blur.shape[0], blur.shape[1])) # init 2D numpy array

    # def weightedAverage(pixel):
    #     return 0.000*pixel[0] + 0.000*pixel[1] + 1.000*pixel[2]

    # # get row number
    # for rownum in range(len(blur)):
    #     for colnum in range(len(blur[rownum])):
    #         gray[rownum][colnum] = weightedAverage(blur[rownum][colnum])

    # gray = np.uint8(gray)
    
    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # equ = cv2.equalizeHist(gray)
    # gray[np.where(mask_erosion==0)] = 0
    # ret,normalized_dark = cv2.threshold(gray,90,255,cv2.THRESH_BINARY_INV)
    # kernel = np.ones((3,3),np.uint8)
    # normalized_dark = cv2.dilate(normalized_dark,kernel,iterations=1)


def min_values_in_range(arr, r_l, r_u, r_v):
in 
#        if np.equal(color, fetched).all(1).any():
#            continue
#
#        l = np.logical_and(np_colors[:,0]<color[0]+r_l, np_colors[:,0]>color[0]-r_l)
#        u = np.logical_and(np_colors[:,1]<color[1]+r_u, np_colors[:,1]>color[1]-r_u)
#        v = np.logical_and(np_colors[:,2]<color[2]+r_v, np_colors[:,2]>color[2]-r_v)
#        values = np.logical_and.reduce((l,u,v))
#        indices = np.where(values)
#        overlapping_colors = np_colors[indices]
#        fetched = np.concatenate((fetched, overlapping_colors))
#        merged_colors.append(overlapping_colors)
 

### OLD ###

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
    # testing if colors are missed...
#    checking_missed_boolean = []
#    for i, a in enumerate(min_colors):
#        l = np.logical_and(colors[:,0]<=(a[0]+luv_range/2), colors[:,0]>=(a[0]-luv_range/2))
#        u = np.logical_and(colors[:,1]<=(a[1]+luv_range/2), colors[:,1]>=(a[1]-luv_range/2))
#        v = np.logical_and(colors[:,2]<=(a[2]+luv_range/2), colors[:,2]>=(a[2]-luv_range/2))
#        values = np.logical_and.reduce((l,u,v))
#        if len(checking_missed_boolean) == 0:
#            checking_missed_boolean = values
#        else:
#            checking_missed_boolean += values
#
#    indices = np.where((checking_missed_boolean==0))
#    colors_missed = colors[indices]
#    print "missed: "+str(colors_missed)
#
#    thresholds = np.zeros((image.shape[0], image.shape[1]))
#    for color in colors:
#        color_range = cv2.inRange(image, color, color)
#        thresholds += color_range
#    boolean_thresholds = (thresholds>0)
#    masked_img = cv2.bitwise_and(image, image, mask=boolean_thresholds.astype(np.uint8))
#    cv2.imshow("all colors", masked_img)
#
#    thresholds = np.zeros((image.shape[0], image.shape[1]))
#    for color in min_colors:
#        color_range = cv2.inRange(image, color-luv_range/2, color+luv_range/2)
#        thresholds += color_range
#    boolean_thresholds = (thresholds>0)
#    masked_img = cv2.bitwise_and(image, image, mask=boolean_thresholds.astype(np.uint8))
#    cv2.imshow("all merged", masked_img)


shifted = cv2.cvtColor(shifted, cv2.COLOR_LUV2BGR)

cv2.imshow('shifted', shifted)

# canny edge detection with automatic upper/lower bounds
# TODO: maybe use Hough tranform or RANSAC line improvements!
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


