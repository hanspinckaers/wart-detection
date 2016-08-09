# from tsne import bh_sne
import numpy as np
from skimage.transform import resize
import time
import pudb

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

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

def image_scatter(f2d, images, img_res=50, res=8000, cval=1., cmap=None, labels=None): 
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
    # images = [gray_to_color(image) for image in images]

    images = np.load("images_resized.npy")
    # images = [min_resize(image, img_res) for image in images]
    # np.save("images_resized", images)

    shapes = np.array([image.shape for image in images])

    img_widths = shapes[:,0] 
    img_heights = shapes[:,1] 

    max_width = max(img_widths)
    max_height = max(img_heights)

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
    coords = np.ones((len(xx), 2), dtype=int)

    for i, (x, y) in enumerate(zip(xx, yy)):
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        coords[i][0] = x_idx
        coords[i][1] = y_idx

    print "Resolve overlapping:"

    j = 0  # keep track of n runs
    running = True

    overall_vectors = np.zeros(coords.shape)

    # greedy algorithm (locally optimum choice at each stage) to minimize overlap
    overall_start_time = time.time()
    
    while running:
        start_time = time.time()

        overall_movement = 0  # keep track of overall movement
        overall_n = 0  # number of images moved

        i = 0
        overall_vectors = np.zeros(coords.shape)

        img_coords_w = coords[:,0] + img_widths
        img_coords_h = coords[:,1] + img_heights

        while i < len(coords):
            coord = coords[i]
            overall_vector = np.array([0,0], dtype=float)
            
            candidate_indices = np.where((np.abs(coords[:,0] - coord[0]) < img_res) & (np.abs(coords[:,1] - coord[1]) < img_res))[0]
            can_coords = coords[candidate_indices]

            can_w_plus_x = img_coords_w[candidate_indices]
            can_h_plus_y = img_coords_h[candidate_indices]

            item_w_plus_x = img_coords_w[i]
            item_h_plus_y = img_coords_h[i]

            x = ((can_coords[:,0] > coord[0]) & (can_coords[:,0] < item_w_plus_x))
            xx = ((can_w_plus_x > coord[0]) & (can_w_plus_x < item_w_plus_x))
            xxx = ((can_coords[:,0] <= coord[0]) & (can_w_plus_x >= item_w_plus_x))

            overlap_x = (x | xx | xxx)
            
            y = ((can_coords[:,1] > coord[1]) & (can_coords[:,1] < item_h_plus_y))
            yy = ((can_h_plus_y > coord[1]) & (can_h_plus_y < item_h_plus_y))
            yyy = ((can_coords[:,1] <= coord[1]) & (can_h_plus_y >= item_h_plus_y))

            overlap_y = (y | yy | yyy)

            check_indices = np.where(overlap_x & overlap_y)[0]

            if len(check_indices) == 1: 
                i += 1
                continue

            overlap_coords = can_coords[check_indices]
            vec_diff = overlap_coords - coord  # difference between the two coordinates
            max_diff = np.abs(vec_diff).max(axis=1).astype(float) 
            max_diff[max_diff == 0] = 1
            vec = vec_diff / max_diff[:,None]  

            x_less_zero = np.where(vec[:,0] < 0)
            x_grea_zero = np.where(vec[:,0] > 0)
            y_less_zero = np.where(vec[:,1] < 0)
            y_grea_zero = np.where(vec[:,1] > 0)

            vec[:,0][x_less_zero] *= img_widths[check_indices[x_less_zero]]
            vec[:,0][x_grea_zero] *= img_widths[i]
            vec[:,1][y_less_zero] *= img_heights[check_indices[y_less_zero]]
            vec[:,1][y_grea_zero] *= img_heights[i]

            vec_diff = (vec - vec_diff) / 2.  # calculate the diff to current idx location

            overall_vector = np.sum(vec_diff, axis=0)
            if len(check_indices) > 1:
                overall_vector = overall_vector / (len(check_indices) - 1)

            # always move by at least a pixel
            if overall_vector[0] < 0:
                overall_vector[0] = np.floor(overall_vector[0])
            else:
                overall_vector[0] = np.ceil(overall_vector[0])

            if overall_vector[1] < 0:
                overall_vector[1] = np.floor(overall_vector[1])
            else:
                overall_vector[1] = np.ceil(overall_vector[1])
            
            overall_movement += np.abs(overall_vector)
            overall_n += 1  

            overall_vectors[i] = overall_vector

            i += 1

        avg_movement = np.sum(overall_movement / overall_n)

        if j > 0:
            print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE) 

        print("--- run %s: %.3f seconds - %.3f avg movement - %s moved ---" % (j, time.time() - start_time, avg_movement, overall_n))

        if j > 500 or avg_movement <= 1:
            running = False

        coords = (coords - overall_vectors).astype(int)

        j += 1

    n_x_min, n_y_min = coords.min(axis=0)
    coords[:,0] -= n_x_min
    coords[:,1] -= n_y_min
    n_x_max, n_y_max = coords.max(axis=0)

    print "--- Making plot: (" + str(n_x_max) + "," + str(n_y_max) + ") ---"

    canvas = np.ones((n_x_max + max_width, n_y_max + max_height, 3)) * cval

    for x, y, image in zip(coords[:,0], coords[:,1], images):
        w, h = image.shape[:2]
        canvas[x:x + w, y:y + h] = image

    return canvas
