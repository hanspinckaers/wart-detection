from feature_detection import analyze_images
from joblib import Parallel, delayed

import multiprocessing

# arguments:
detector_name = 'SIFT'
descriptor_name = 'FREAK'

combinations = [
    ['SIFT', 'SIFT'],
    ['SIFT', 'FREAK'],
    ['SURF', 'SURF'],
    ['SURF', 'FREAK'],
    ['AKAZE', 'AKAZE'],  # AKAZE is the modern version than KAZE (Accelerated KAZE)
    ['KAZE', 'KAZE'],
    ['ORB', 'ORB'],
    ['Agast', 'SURF'],
    ['GFTT', 'SURF'],
    ['MSER', 'SURF'],
]

n_features = [10, 25, 50]
sensitivity = [1, 2]
bow_size = [200, 1000, 500]  # too small bag of words: not representative of all patches, too big: overfitting!

first = True

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(analyze_images)(det_dec[0], det_dec[1], n, s, b)
                                     for det_dec in combinations
                                     for n in n_features
                                     for s in sensitivity
                                     for b in bow_size)

# for det_dec in combinations:
#     for n in n_features:
#         for s in sensitivity:
#             for b in bow_size:
#                 try:
#                     analyze_images(det_dec[0], det_dec[1], n, s, b)
#                 except:
#                     print "Error in this loop!"
#                 print "----------------------------------------------\n\n"
