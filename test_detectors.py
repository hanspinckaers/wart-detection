import os
import sys
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

# if len(sys.argv) > 1:
#    combinations = combinations[int(sys.argv[1]) * 2:int(sys.argv[1]) * 2 + 2]

n_features = [10, 25, 50]
sensitivity = [1, 2]
bow_size = [200, 1000, 500]  # too small bag of words: not representative of all patches, too big: overfitting!

first = True

num_cores = int(round(multiprocessing.cpu_count() / 4))
if num_cores == 0:
    num_cores = 1

combi_args = []

for det_dec in combinations:
    for n in n_features:
        for s in sensitivity:
            for b in bow_size:
                child_pid = os.fork()
                if child_pid == 0:
                    os.system("python tsne_scatter.py " + det_dec[0] + " " + det_dec[1] + " " + str(n) + " " + str(s) + " " + str(b))
                    sys.exit(0)
                else:
                    os.waitpid(child_pid, 0)
