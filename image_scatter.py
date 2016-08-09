# from tsne import bh_sne
import numpy as np
from skimage.transform import resize
import time
import pudb

# symbols used for printing output
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


def scaled_coordinates(coordinates, new_size=8000):
    xx = coordinates[:,0]
    yy = coordinates[:,1]
    
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()

    # Fix the ratios
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = sx / float(sy) * new_size
        res_y = new_size
    else:
        res_x = new_size
        res_y = sy / float(sx) * new_size

    # create an array from x/y_min to x/y_max with res_x/y (width of img) points
    # you basically have a array where the index is the pixel location in the final img and
    # the value is the location relative to the tsne results / coordinates
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)

    # make new coordinates
    coords = np.ones((len(xx), 2), dtype=int)

    for i, (x, y) in enumerate(zip(xx, yy)):
        x_idx = np.argmin((x - x_coords) ** 2)  # ** 2 is to make everything positive
        y_idx = np.argmin((y - y_coords) ** 2)
        coords[i][0] = x_idx
        coords[i][1] = y_idx

    return coords


def image_scatter(coordinates, images, img_size=50, scatter_size=8000, cval=1., resolve_overlapping=1000):
    """
    Embeds images with coordinates (e.g. from tsne) into a scatter plot.

    Parameters
    ---------
    coordinates: numpy array
        Initial coordinates for images (e.g. from a tsne run)

    images: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    img_size: float or int
        Maximum width and height of image, function scales to ratio

    scatter_size: float or int
        *Minimum* width or height of embedding image in pixels

    cval: float or numpy array
        Background color value

    resolve_overlapping: float or int
        Number of maximum runs the algorithm should do to resolve overlapping
        If the plot has no overlapping images below this threshold the run is stopped
        0 to disable overlap resolvement
        
    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    # images = [gray_to_color(image) for image in images]

    coords = scaled_coordinates(coordinates, scatter_size)

    images = np.load("images_resized.npy")
    # images = [min_resize(image, img_size) for image in images]
    # np.save("images_resized", images)

    shapes = np.array([image.shape for image in images])

    img_widths = shapes[:,0]
    img_heights = shapes[:,1]

    max_width = max(img_widths)
    max_height = max(img_heights)

    if resolve_overlapping > 0:
        # greedy algorithm (locally optimum choice at each stage) to minimize overlap
        
        print "Resolve overlapping:"

        j = 0  # keep track of n runs
        running = True

        overall_vectors = np.zeros(coords.shape)

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
                
                candidate_indices = np.where((np.abs(coords[:,0] - coord[0]) < img_size) & (np.abs(coords[:,1] - coord[1]) < img_size))[0]
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

                # if images are perfectly overlapping seperate them by
                # giving them a push to left/right depending on their index
                perfect_overlap = np.where((max_diff == 0))[0]
                if len(perfect_overlap) > 1:
                    for p in perfect_overlap:
                        p_i = candidate_indices[check_indices[p]]  # get the original index of coords
                        # if p_i is i continue (this is ourselves)
                        if i < p_i:
                            vec_diff[p] = np.array([1,0])
                        elif i > p_i:
                            vec_diff[p] = np.array([0,1])
                
                max_diff[perfect_overlap] = 1
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

            if overall_n > 0:
                avg_movement = np.sum(overall_movement / overall_n)
            
            if overall_n == 0 or j > resolve_overlapping:  # or avg_movement <= 2  -> average movement of 2 means images are moving with 1 pixel diff
                running = False

            if j > 0:
                print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

            print("--- Run %s: %.3f seconds - %.3f avg movement - %s moved ---" % (j, time.time() - start_time, avg_movement, overall_n))

            coords = (coords - overall_vectors).astype(int)

            j += 1

        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print("--- Overall run (n=%s) took: %.2f min" % (j, (time.time() - overall_start_time) / 60))

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
