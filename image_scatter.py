# from tsne import bh_sne
import numpy as np
from skimage.transform import resize
import time
import os
import cv2

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
    h, w = map(float, img.shape[:2])
    if min([w, h]) != size:
        if h <= w:
            img = resize(img, (int(round((h / w) * size)), int(size)))  # rows / columns so y / x
        else:
            img = resize(img, (int(size), int(round((w / h) * size))))
    return img


def scaled_coordinates(coordinates, new_size=8000):
    xx = coordinates[:,1]
    yy = coordinates[:,0]

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


def image_scatter(coordinates, images, img_size=50, scatter_size=8000, cval=1., resolve_overlapping=1000, assume_same_img_size=False):
    """
    Embeds images with coordinates (e.g. from tsne) into a scatter plot.

    When resolving the overlap between images the function finds new
    coordinates for the images that minimize overlap. Distances between
    images will thus not be perfectly preserved, general overall structure
    should remain. The algorithm pushes the images out of eachother (in
    opposite directions of their overlap), like when in a multiple collision
    in a physics engine.


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

    assume_same_img_size: boolean
        If True the algorithm assumes that all the input images are of the same size (img_size x img_size).
        This provides a speed up of ~30%.

    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    coords = scaled_coordinates(coordinates, scatter_size)

    if os.path.exists('images_resized.npy'):  # load cache if it exists
        images = np.load("images_resized.npy")
        images = images[0:len(coords)]
    else:
        images = [gray_to_color(image) for image in images]
        images = [min_resize(image, img_size) for image in images]

    shapes = np.array([image.shape for image in images])

    img_widths = shapes[:,1]
    img_heights = shapes[:,0]

    max_width = max(img_widths)
    max_height = max(img_heights)

    if resolve_overlapping > 0:
        # greedy algorithm (locally optimum choice at each stage) to minimize overlap
        print "Resolve overlapping:"

        j = 0  # keep track of n runs
        running = True

        overall_start_time = time.time()

        n_x_max, n_y_max = coords.max(axis=0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_size = (int(n_y_max * 1.25), int(n_x_max * 1.25), 3)
        out = cv2.VideoWriter("scatter.mp4", fourcc, 7, (video_size[1], video_size[0]), True)

        while running:
            start_time = time.time()

            overall_movement = 0  # keep track of overall movement
            overall_n = 0  # number of images moved

            i = 0
            overall_vectors = np.zeros(coords.shape)

            img_coords_w = coords[:,0] + img_widths
            img_coords_h = coords[:,1] + img_heights

            while i < len(coords):
                coord = coords[i]  # coord of the current image
                overall_vector = np.array([0,0], dtype=float)

                # find coordinates who possibly collide with current image when
                # assuming same size for each images, these coords do collide.
                candidate_indices = np.where((np.abs(coords[:,0] - coord[0]) < img_size) & (np.abs(coords[:,1] - coord[1]) < img_size))[0]

                # if we only found ourselves -> go to next image
                if len(candidate_indices) == 1:
                    i += 1
                    continue

                can_coords = coords[candidate_indices]

                if not assume_same_img_size:
                    # select images that collide from possible collision candidates
                    can_w_plus_x = img_coords_w[candidate_indices]  # array of right edges of the candidate images (x + w)
                    can_h_plus_y = img_coords_h[candidate_indices]  # array of bottom edges of the candidate images (y + h)

                    item_w_plus_x = img_coords_w[i]  # left edge of the current image
                    item_h_plus_y = img_coords_h[i]  # bottom edge of the current image

                    # {-----} current image, |-----| candidate
                    x = ((can_coords[:,0] > coord[0]) & (can_coords[:,0] < item_w_plus_x))  # boolean array where candidate's left edge is between current image {---|--}---|
                    xx = ((can_w_plus_x > coord[0]) & (can_w_plus_x < item_w_plus_x))  # boolean array where candidate's right edge is between current |--{---|--}
                    xxx = ((can_coords[:,0] <= coord[0]) & (can_w_plus_x >= item_w_plus_x))  # boolean array where candidate's edges are overlapping current --|--{-----}-|--

                    overlap_x = (x | xx | xxx)  # combine boolean arrays

                    # same as above x-axis but for y
                    y = ((can_coords[:,1] > coord[1]) & (can_coords[:,1] < item_h_plus_y))
                    yy = ((can_h_plus_y > coord[1]) & (can_h_plus_y < item_h_plus_y))
                    yyy = ((can_coords[:,1] <= coord[1]) & (can_h_plus_y >= item_h_plus_y))

                    overlap_y = (y | yy | yyy)

                    check_indices = np.where(overlap_x & overlap_y)[0]
                    overlap_coords = can_coords[check_indices]
                else:
                    check_indices = np.arange(len(can_coords))
                    overlap_coords = can_coords

                # if we only found ourselves -> go to next image
                if len(check_indices) == 1:
                    i += 1
                    continue

                vec_diff = overlap_coords - coord  # vectors of difference between the overlapping coordinates and current image coord

                # if images are perfectly overlapping seperate them by
                # giving them a push to left/right depending on their index
                perfect_overlap = np.where((vec_diff[:,0] == 0) & (vec_diff[:,1] == 0))[0]
                if len(perfect_overlap) > 1:
                    for p in perfect_overlap:
                        p_i = candidate_indices[check_indices[p]]  # get the original index of coords
                        # if p_i is i continue (this is ourselves)
                        if i < p_i:
                            vec_diff[p] = np.array([1,0])
                        elif i > p_i:
                            vec_diff[p] = np.array([0,1])

                if not assume_same_img_size:
                    heights = img_heights[candidate_indices[check_indices]]
                    widths = img_widths[candidate_indices[check_indices]]

                    above = np.where((vec_diff[:,1] < 0) & (vec_diff[:,0] >= 0))[0]
                    left_up = np.where((vec_diff[:,1] < 0) & (vec_diff[:,0] < 0))[0]
                    left = np.where((vec_diff[:,1] >= 0) & (vec_diff[:,0] < 0))[0]
                    inside = np.where((vec_diff[:,1] >= 0) & (vec_diff[:,0] >= 0))[0]

                    for a_index in above:
                        vec_d = vec_diff[a_index].astype(float)
                        overlap_img_width = widths[a_index]
                        overlap_img_height = heights[a_index]

                        align_top = vec_d / np.abs(vec_d[1]) * (overlap_img_height - np.abs(vec_d[1]))
                        if vec_d[0] != 0:
                            align_left = vec_d / np.abs(vec_d[0]) * (img_widths[i] - np.abs(vec_d[0]))
                        else:
                            align_left = np.array([np.inf, np.inf])

                        if np.linalg.norm(align_top) < np.linalg.norm(align_left):
                            vec_diff[a_index] = align_top
                        else:
                            vec_diff[a_index] = align_left

                    for a_index in left_up:
                        vec_d = vec_diff[a_index].astype(float)
                        overlap_img_width = widths[a_index]
                        overlap_img_height = heights[a_index]

                        align_top = vec_d / np.abs(vec_d[1]) * (overlap_img_height - np.abs(vec_d[1]))
                        align_left = vec_d / np.abs(vec_d[0]) * (overlap_img_width - np.abs(vec_d[0]))

                        if np.linalg.norm(align_top) < np.linalg.norm(align_left):
                            vec_diff[a_index] = align_top
                        else:
                            vec_diff[a_index] = align_left

                    for a_index in left:
                        vec_d = vec_diff[a_index].astype(float)
                        overlap_img_width = widths[a_index]
                        overlap_img_height = heights[a_index]

                        if vec_d[1] != 0:
                            align_top = vec_d / np.abs(vec_d[1]) * (img_heights[i] - np.abs(vec_d[1]))
                        else:
                            align_top = np.array([np.inf, np.inf])

                        align_left = vec_d / np.abs(vec_d[0]) * (overlap_img_width - np.abs(vec_d[0]))

                        if np.linalg.norm(align_top) < np.linalg.norm(align_left):
                            vec_diff[a_index] = align_top
                        else:
                            vec_diff[a_index] = align_left

                    for a_index in inside:
                        vec_d = vec_diff[a_index].astype(float)
                        overlap_img_width = widths[a_index]
                        overlap_img_height = heights[a_index]
                        if vec_d[1] != 0:
                            align_top = vec_d / np.abs(vec_d[1]) * (img_heights[i] - np.abs(vec_d[1]))
                        else:
                            align_top = np.array([np.inf, np.inf])

                        if vec_d[0] != 0:
                            align_left = vec_d / np.abs(vec_d[0]) * (img_widths[i] - np.abs(vec_d[0]))
                        else:
                            align_left = np.array([np.inf, np.inf])
                        if vec_d[0] == 0 and vec_d[1] == 0:
                            vec_diff[a_index] = vec_d
                        elif np.linalg.norm(align_top) < np.linalg.norm(align_left):
                            vec_diff[a_index] = align_top
                        else:
                            vec_diff[a_index] = align_left

                    vec_diff = vec_diff / 2.
                else:
                    max_diff = np.abs(vec_diff).max(axis=1).astype(float)  # axis of maximum difference with current image coord
                    max_diff[perfect_overlap] = 1

                if assume_same_img_size:
                    vec = vec_diff / np.abs(max_diff[:,None])
                    vec *= img_size  # does this have to be negative? why?
                    vec_diff = (vec - vec_diff) / 2.  # calculate the diff to current idx location

                overall_vector = np.average(vec_diff, axis=0)

                if len(check_indices) > 0:
                    overall_vector = overall_vector / (len(check_indices) - 1)

                # always move by at least a pixel in either direction
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
            j += 1

            if overall_n > 0:
                avg_movement = np.sum(overall_movement / overall_n)

            if overall_n == 0 or j == resolve_overlapping:  # or avg_movement <= 2  -> average movement of 2 means images are moving with 1 pixel diff
                running = False

            # generate img per run
            n_x_min, n_y_min = coords.min(axis=0)
            coords[:,0] -= n_x_min
            coords[:,1] -= n_y_min
            n_x_max, n_y_max = coords.max(axis=0)

            canvas = np.ones((n_y_max + max_height, n_x_max + max_width, 3)) * cval

            for x, y, image in zip(coords[:,0], coords[:,1], images):
                h, w = image.shape[:2]
                canvas[y:y + h, x:x + w] = image

            canvas = canvas * 255
            if canvas.shape[0] < video_size[0] and canvas.shape[1] < video_size[1]:
                frame = np.ones(video_size) * 255
                frame[0:canvas.shape[0], 0:canvas.shape[1]] = canvas
                for i, (coord, vector) in enumerate(zip(coords, overall_vectors)):
                    if np.linalg.norm(vector) > 0:
                        tiplength = 2 / np.linalg.norm(vector)
                        center = np.array([coord[0] + img_widths[i] / 2, coord[1] + img_heights[i] / 2])
                        cv2.arrowedLine(frame, tuple(center), tuple((center - vector * 5).astype(int)), (0,0,0), thickness=2, tipLength=tiplength)

                out.write(np.uint8(frame))  # normalize size!

            coords = (coords - overall_vectors).astype(int)

            if j > 1:
                print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

            print("--- Run %s: %.3f seconds - %.3f avg movement - %s moved ---" % (j, time.time() - start_time, avg_movement, overall_n))

        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print("--- Overall run (n=%s) took: %.2f min" % (j, (time.time() - overall_start_time) / 60))

    n_x_min, n_y_min = coords.min(axis=0)
    coords[:,0] -= n_x_min
    coords[:,1] -= n_y_min
    n_x_max, n_y_max = coords.max(axis=0)

    print "--- Making plot: (" + str(n_x_max) + "," + str(n_y_max) + ") ---"

    canvas = np.ones((n_y_max + max_height, n_x_max + max_width, 3)) * cval

    for x, y, image in zip(coords[:,0], coords[:,1], images):
        h, w = image.shape[:2]
        canvas[y:y + h, x:x + w] = image

    return canvas
