# from tsne import bh_sne
import numpy as np
import time
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
    Resize an image so that the longest edge is equal to size.
    """
    h, w = map(float, img.shape[:2])
    if max([w, h]) != size:
        if h <= w:
            img = cv2.resize(img, (int(size), int(round((h / w) * size))))  # rows / columns so y / x
        else:
            img = cv2.resize(img, (int(round((w / h) * size)), int(size)))

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


def image_scatter(coordinates, images, img_size=50, scatter_size=8000, cval=1., resolve_overlapping=1000, assume_same_img_size=False, video_filename=None, video_fps=7, video_codec="mp4v", video_scale=0.5):
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

    video_filename: string
        If set to a string (default is None) a video file will be created showing every run of the algorithm in video_fps.

    video_fps: integer
        The frames per second used when create a video_codec

    video_codec: string
        openCV fourcc codes "mp4v"

    video_scale: float
        What scale the video should have relatively to scatter

    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    print "--- Making image scatter ---"

    assert len(coordinates) == len(images), "There should be the same amount of coordinates as images"

    if len(coordinates) == 0:
        return np.array()

    coords = scaled_coordinates(coordinates, scatter_size)

    # note: for all coordinates after this call the x-value is the *first* value in the array, y the second:
    # coord = [x, y]
    # this is different from some image libraries where y and height is always first

    # print "--- Resizing images ---"
    images = [gray_to_color(image) for image in images]
    images = [min_resize(image, img_size) for image in images]

    shapes = np.array([image.shape for image in images])

    img_widths = shapes[:,1]
    img_heights = shapes[:,0]

    max_width = max(img_widths)
    max_height = max(img_heights)

    if resolve_overlapping > 0:
        print "--- Resolve overlapping: ---"

        # greedy algorithm (locally optimum choice at each stage) to minimize overlap.
        # Basically every run it computes all vectors needed to solve each collision per image and then
        # computes an average vector per image from that. Then every image is moved according to their
        # individual calculated average vector and the run start again (checking if by moving the images
        # there are now new overlaps between the images.

        n_j = 0  # keep track of n runs
        running = True

        overall_start_time = time.time()

        if video_filename is not None:
            n_x_max, n_y_max = coords.max(axis=0)
            fourcc = cv2.VideoWriter_fourcc(*video_codec)
            frame_size = (int(n_y_max * 1.25), int(n_x_max * 1.25), 3)
            video_size = (int(frame_size[0] * video_scale), int(frame_size[1] * video_scale))
            out = cv2.VideoWriter(video_filename, fourcc, video_fps, (video_size[1], video_size[0]), True)

        while running:
            start_time = time.time()

            overall_movement = 0  # keep track of overall movement
            overall_n = 0  # number of images moved

            i = 0  # current image coord index
            overall_vectors = np.zeros(coords.shape)  # saved the average vectors, gets applied on coords in the end of the run

            img_coords_w = coords[:,0] + img_widths
            img_coords_h = coords[:,1] + img_heights

            while i < len(coords):
                coord = coords[i]  # coord of the *current image*
                overall_vector = np.array([0,0], dtype=float)

                # find coordinates who possibly collide with *current image*
                # when assuming the same size for each image, these coords do actually collide.
                candidate_indices = np.where((np.abs(coords[:,0] - coord[0]) < img_size) & (np.abs(coords[:,1] - coord[1]) < img_size))[0]

                # if we only found ourselves -> go to next image
                if len(candidate_indices) == 1:
                    i += 1
                    continue

                can_coords = coords[candidate_indices]

                if not assume_same_img_size:
                    # select images that actually collide from possible collision candidates
                    # the above candidate_indices is a "pre-screen" with images that could overlap based
                    # on the fact they are (function arg) img_size + img_size away from the current image.
                    # img_size should thus be the same size as the largest image

                    can_w_plus_x = img_coords_w[candidate_indices]  # array of right edges of the candidate images (x + w)
                    can_h_plus_y = img_coords_h[candidate_indices]  # array of bottom edges of the candidate images (y + h)

                    item_w_plus_x = img_coords_w[i]  # right edge of the current image
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

                    check_indices = np.where(overlap_x & overlap_y)[0]  # if there is an overlap on the x-axis and on the y there is overlap
                    overlap_coords = can_coords[check_indices]
                else:
                    # when we assume all images are of the same size can_coords has all the overlapping coords
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
                    # we need scale the difference vectors such that the images don't overlap anymore
                    # this means the individual relation the images have in term of locations is
                    # sort of preserved.
                    can_heights = img_heights[candidate_indices[check_indices]]
                    can_widths = img_widths[candidate_indices[check_indices]]

                    # figure out in which quadrant an overlapping image is
                    # this is important to determine of the overlapping image size
                    # needs to be used or the *current image* size to correct the overlapping
                    above = (vec_diff[:,1] < 0) & (vec_diff[:,0] >= 0)
                    left_up = (vec_diff[:,1] < 0) & (vec_diff[:,0] < 0)
                    left = (vec_diff[:,1] >= 0) & (vec_diff[:,0] < 0)
                    inside = (vec_diff[:,1] >= 0) & (vec_diff[:,0] >= 0)

                    #
                    #   ***********
                    #   * l- * a  *   l-u = left_up quadrant
                    #   * u  *    *   a = above quadrant
                    #   *****XXXXXX
                    #   * l  X i  X   l = left quadrant
                    #   *    X    X   i = inside quadrant
                    #   *****XXXXXX
                    #
                    #   X = *current image*
                    #   * = area of possible overlapping image
                    #
                    #   Difference vectors are scaled according to in which quadrant they end (they start at top-left coordinate
                    #   of *current image* and go to top-left coordinate of overlapping image.
                    #
                    #   e.g.: if (top-left) coordinate of the overlapping image is in l-u quadrant the difference vector
                    #   needs to be scaled by the width or the height of the overlapping image:
                    #
                    #                      ^++++++
                    #                      +\+++++
                    #     ^++++++          ++\++++
                    #     +\+++++          +++\+++                        X = current image
                    #     ++\++++          ++++\++                        + = overlapping image in l-u quadrant
                    #     +++XXXXXX            XXXXXX                    -> = difference vector (^ is head/direction)
                    #     +++X+++ X   --->     X    X
                    #        X    X            X    X
                    #        XXXXXX            XXXXXX
                    #
                    #  the algorithm tries to align both sides (horizontal / vectical) of the images and then selects
                    #  the difference vector for the side which is the smallest.

                    for v_index in range(len(vec_diff)):
                        vec_d = vec_diff[v_index].astype(float)  # we need floats because we are going to divide

                        if vec_d[0] == 0 and vec_d[1] == 0:  # this is the difference with ourselves
                            continue

                        if vec_d[1] != 0:  # align with horizontal side (except if diff is already 0)
                            if above[v_index] or left_up[v_index]:
                                align_top = vec_d / np.abs(vec_d[1]) * (can_heights[v_index] - np.abs(vec_d[1]))
                            elif left[v_index] or inside[v_index]:
                                align_top = vec_d / np.abs(vec_d[1]) * (img_heights[i] - np.abs(vec_d[1]))
                        else:
                            align_top = np.array([np.inf, np.inf])

                        if vec_d[0] != 0:  # align with vertical side (except if diff is already 0)
                            if above[v_index] or inside[v_index]:
                                align_left = vec_d / np.abs(vec_d[0]) * (img_widths[i] - np.abs(vec_d[0]))
                            elif left_up[v_index] or left[v_index]:
                                align_left = vec_d / np.abs(vec_d[0]) * (can_widths[v_index] - np.abs(vec_d[0]))
                        else:
                            align_left = np.array([np.inf, np.inf])

                        if np.linalg.norm(align_top) < np.linalg.norm(align_left):
                            vec_diff[v_index] = align_top
                        else:
                            vec_diff[v_index] = align_left

                else:
                    max_diff = np.abs(vec_diff).max(axis=1).astype(float)  # axis of maximum difference with current image coord
                    max_diff[perfect_overlap] = 1

                if assume_same_img_size:
                    vec = vec_diff / np.abs(max_diff[:,None])
                    vec *= img_size
                    vec_diff = (vec - vec_diff)  # calculate the diff to current idx location

                overall_vector = np.average(vec_diff, axis=0)  # get the average movement this image has to make to get out of collisions

                # limit step size
                norm = np.linalg.norm(overall_vector)
                if norm > 4.:
                    overall_vector = overall_vector / norm * 4.

                # always move by at least a pixel in either directions
                abs_vec = np.abs(overall_vector)
                if abs_vec[0] < 0.5 and abs_vec[0] >= abs_vec[1]:
                    if overall_vector[0] < 0:
                        overall_vector[0] = np.floor(overall_vector[0])
                    else:
                        overall_vector[0] = np.ceil(overall_vector[0])
                else:
                    overall_vector[0] = round(overall_vector[0])  # be aware: numpy rounds halfs to closest *even* number (0.5 becomes 0, 1.5 becomes 2)

                if abs_vec[1] < 0.5 and abs_vec[1] > abs_vec[0]:
                    if overall_vector[1] < 0:
                        overall_vector[1] = np.floor(overall_vector[1])
                    else:
                        overall_vector[1] = np.ceil(overall_vector[1])
                else:
                    overall_vector[1] = round(overall_vector[1])

                overall_movement += np.abs(overall_vector)
                overall_n += 1

                overall_vectors[i] = overall_vector

                i += 1
            n_j += 1

            if overall_n > 0:
                avg_movement = np.sum(overall_movement / overall_n)

            # stop running when nothing is moving or maximum amount of runs is reached
            if overall_n == 0 or n_j == resolve_overlapping:
                running = False

            if video_filename is not None:
                # generate img per run
                coords -= coords.min(axis=0)  # translate whole canvas to (0,0)
                n_x_max, n_y_max = coords.max(axis=0)

                canvas = np.ones((n_y_max + max_height, n_x_max + max_width, 3)) * cval * 255
                for x, y, image in zip(coords[:,0], coords[:,1], images):
                    h, w = image.shape[:2]
                    canvas[y:y + h, x:x + w] = image

                max_size = scatter_size * 1.25
                if canvas.shape[0] <= frame_size[0] and canvas.shape[1] <= frame_size[1]:
                    frame = np.ones(frame_size) * cval * 255
                    frame[0:canvas.shape[0], 0:canvas.shape[1]] = canvas

                    # draw a arrow per vector
                    for i, (coord, vector) in enumerate(zip(coords, overall_vectors)):
                        if np.linalg.norm(vector) > 0:
                            tiplength = 10 / np.linalg.norm(vector)
                            center = np.array([coord[0] + img_widths[i] / 2, coord[1] + img_heights[i] / 2])
                            cv2.arrowedLine(frame, tuple(center), tuple((center - vector).astype(int)), (0,0,0), thickness=2, tipLength=tiplength)

                    if video_size != 1.:
                        frame = cv2.resize(frame, (video_size[1], video_size[0]))

                    frame = np.uint8(frame)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    out.write(frame)
                else:
                    print("--- Canvas is too big %s for video frame %s x %s, frame not written! Try to make scatter_size bigger --- \n" % (str(canvas.shape), max_size, max_size))

            # finally apply the vectors
            coords = (coords - overall_vectors).astype(int)

            # if n_j > 1:
            #     print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

            print("--- Run %s: %.3f seconds - %.3f avg movement - %s moved ---" % (n_j, time.time() - start_time, avg_movement, overall_n))

        # print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
        print("--- Overall run (n=%s) took: %.2f min" % (n_j, (time.time() - overall_start_time) / 60))

    coords -= coords.min(axis=0)  # translate whole canvas to (0,0)
    n_x_max, n_y_max = coords.max(axis=0)

    print "--- Making plot: (" + str(n_x_max) + "," + str(n_y_max) + ") ---"

    canvas = np.ones((n_y_max + max_height, n_x_max + max_width, 3)) * cval * 255

    for x, y, image in zip(coords[:,0], coords[:,1], images):
        h, w = image.shape[:2]
        canvas[y:y + h, x:x + w] = image

    return canvas
