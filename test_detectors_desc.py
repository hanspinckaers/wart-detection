from feature_detection import analyze_images
from joblib import Parallel, delayed
import os, sys
import pudb
import multiprocessing

# arguments:
detector_name = 'SIFT'
descriptor_name = 'FREAK'

combinations = [
    ['SURF', 'SURF'],
    ['SURF', 'FREAK'],
    ['AKAZE', 'AKAZE'],  # AKAZE is the modern version than KAZE (Accelerated KAZE)
    ['KAZE', 'KAZE'],
    ['SIFT', 'SIFT'],
    ['SIFT', 'FREAK'],
    ['ORB', 'ORB'],
    ['Agast', 'SURF'],
    ['GFTT', 'SURF'],
    ['MSER', 'SURF'],
]

n_features = [10, 25, 50]
sensitivity = [1, 2]
bow_size = [200, 1000, 500]  # too small bag of words: not representative of all patches, too big: overfitting!

first = True

num_cores = int(round(multiprocessing.cpu_count() / 4))
if num_cores == 0:
    num_cores = 1


def analyze(args):
    child_pid = os.fork()
    if child_pid == 0:
        analyze_images(args[0], args[1], args[2], args[3], args[4])
	pu.db
        sys.exit(0)
    else:
        os.waitpid(child_pid, 0)
	pu.db

combi_args = []

for det_dec in combinations:
    for n in n_features:
        for s in sensitivity:
            for b in bow_size:
                analyze((det_dec[0], det_dec[1], n, s, b))

