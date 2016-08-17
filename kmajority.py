import numpy as np
import time
import pudb

from joblib import Parallel, delayed
import os
import multiprocessing

from subprocess import check_output
from random import randint


def kmajority(vectors, k):
    num_cores = multiprocessing.cpu_count()

    overall_start_time = time.time()
    np.random.seed(0)
    random_indices = np.random.rand(k)
    random_indices = np.round(random_indices * (len(vectors) - 1)).astype(int)
    centroids = vectors[random_indices]

    centroids_changed = True
    print("--- Start kmajority with n vectors = " + str(len(vectors)))
    j = 0
    np.savetxt('_t_vectors', vectors, fmt='%i')
    last_runs = False

    while centroids_changed:
        centroids_changed = False
        overall_bits_changed = 0
        start_time = time.time()

        np.savetxt('_t_centroids', centroids, fmt='%i')

        # divide the vectors and parallelize the search for closest centroids
        divide_by = float(num_cores)
        len_until = (np.floor(len(vectors) / divide_by) * divide_by).astype(int)
        dividable = vectors[0:len_until]
        split = np.split(dividable, int(divide_by))
        restover = vectors[len_until:]
        if len(restover) > 0:
            split[-1] = np.append(split[-1], restover, axis=0)
        results = Parallel(n_jobs=num_cores)(delayed(closest_dist)(vecs, centroids) for vecs in split)
        cen_per_vec = np.concatenate(results)

        # majority voting
        for c_i in np.arange(len(centroids)):
            vectors_i = np.where(cen_per_vec == c_i)[0]
            if len(vectors_i) == 0:
                continue
            hist = np.sum(vectors[vectors_i], axis=0)
            half = len(vectors_i) / 2
            hist[hist < half] = 0
            hist[hist >= half] = 1
            if np.any(hist != centroids[c_i]):
                centroids_changed = True
                if last_runs:
                    print str(c_i) + " changed " + str(np.sum(hist != centroids[c_i])) + " bits"
                overall_bits_changed += np.sum(hist != centroids[c_i])
                centroids[c_i] = hist
            else:
                print "--- Nothing found for centroid: " + str(c_i)
                # start from other random point?

        j += 1
        print("--- Run: %s changed %s bits in %.3f seconds" % (str(j), str(overall_bits_changed), time.time() - start_time))
        last_runs = (overall_bits_changed < 10)

    print("--- Overall kmajority %.3f seconds -" % (time.time() - overall_start_time))
    os.remove('_t_vectors')
    return centroids


def closest_dist(vecs, centroids):
    rand = str(randint(0,5000))
    np.savetxt('_t_vectors_' + rand, vecs, fmt='%i')
    output = check_output(["./kmajority", str(vecs.shape[1]), str(vecs.shape[0]), '1000', '_t_vectors_' + rand, '_t_centroids'])
    cen_per_vec = np.fromstring(output, dtype=int, sep=' ')
    os.remove('_t_vectors_' + rand)
    return cen_per_vec


# old python code:

# start_time = time.time()
# assign data to centroids

# print("Running kmajority")
# output = check_output(["./kmajority", str(vectors.shape[1]), str(vectors.shape[0]), str(k), '_t_vectors', '_t_centroids'])
# cen_per_vec = np.fromstring(output, dtype=int, sep=' ')
# print output
# cen_per_vec = np.genfromtxt(output, delimiter=",")
# print("--- Run %.3f seconds -" % (time.time() - start_time))
# pu.db
# np.set_printoptions(**opt)

# vectors_per_centroids = {}

# for r in results:
#     for r_c in r:
#         if r_c in vectors_per_centroids:
#             vectors_per_centroids[r_c].extend(r[r_c])
#         else:
#             vectors_per_centroids[r_c] = r[r_c]

# print("--- Run %.3f seconds -" % (time.time() - start_time))

# start_time = time.time()

# for v_i, v in enumerate(vectors):
#     closest_c_i = None
#     dists = np.sum(centroids != v, axis=1)
#     closest_c_i = np.argmin(dists)
#     if v_i % 10000 == 0:
#         print("v: " + str(v_i) + " closest c: " + str(closest_c_i))
#         print("will take approximate: %.3f s" % ((time.time() - start_time) / (v_i + 1) * len(vectors)))
#     if closest_c_i in vectors_per_centroids:
#         vectors_per_centroids[closest_c_i].append(v_i)
#     else:
#         vectors_per_centroids[closest_c_i] = [v_i]
# print("--- Run %.3f seconds -" % (time.time() - start_time))

# vectors_per_centroids = {}

# for v_i, v in enumerate(vecs):
#     closest_c_i = None
#     dists = np.sum(centroids != v, axis=1)
#     closest_c_i = np.argmin(dists)
#     if closest_c_i in vectors_per_centroids:
#         vectors_per_centroids[closest_c_i].append(v_i)
#     else:
#         vectors_per_centroids[closest_c_i] = [v_i]
# return vectors_per_centroids
