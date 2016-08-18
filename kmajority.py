import numpy as np
import time
import pudb

from joblib import Parallel, delayed
import os
import multiprocessing

from subprocess import check_output
from random import randint


def kmajority(vectors, k):
    # determine which amount of jobs would be ideal
    num_cores = multiprocessing.cpu_count()

    overall_start_time = time.time()

    # choose initial centroids at random (but reproducable with a seed)
    np.random.seed(0)
    random_indices = np.random.rand(k)
    random_indices = np.round(random_indices * (len(vectors) - 1)).astype(int)
    centroids = vectors[random_indices]

    centroids_changed = True

    print("--- Start kmajority with n vectors = " + str(len(vectors)))

    # we use files to send vectors to kmajority c binary
    np.savetxt('_t_vectors', vectors, fmt='%i')

    # when we are really close log which centroids are changing (check to see if we are converging)
    last_runs = False

    j = 0  # count the number of iterations
    while centroids_changed:
        centroids_changed = False

        overall_bits_changed = 0  # bits changed this run

        start_time = time.time()

        # save new centroids for kmajority c binary
        np.savetxt('_t_centroids', centroids, fmt='%i')

        # divide the vectors and parallelize the search for closest centroids
        divide_by = float(num_cores)
        dividable_len = (np.floor(len(vectors) / divide_by) * divide_by).astype(int)
        part_dividable = vectors[0:dividable_len]
        splitted_vectors = np.split(part_dividable, int(divide_by))  # we could also use lsplit to determine indices
        leftover = vectors[dividable_len:]
        if len(leftover) > 0:
            splitted_vectors[-1] = np.append(splitted_vectors[-1], leftover, axis=0)

        # start parallel jobs to find closest centroid for each vector
        results = Parallel(n_jobs=num_cores)(delayed(closest_dist)(vecs, unique_i, centroids, k) for unique_i, vecs in enumerate(splitted_vectors))
        cen_per_vec = np.concatenate(results)

        print("--- kmajority %.3f seconds" % (time.time() - start_time))

        # majority voting as in Graba et al. A Fast Approach for Integrating ORB Descriptors in the Bag of Words Model 2013
        bin_centroids = np.unpackbits(centroids.astype(np.ubyte), axis=1)
        bin_vectors = np.unpackbits(vectors.astype(np.ubyte), axis=1)

        for c_i in range(len(centroids)):
            vectors_i = np.where(cen_per_vec == c_i)[0]  # get all vectors that have this centroid as closest

            if len(vectors_i) == 0:
                print "--- Nothing found for centroid: " + str(c_i)  # start from other random point?
                continue

            if len(vectors_i) == 1:
                print "--- Only found 1 for centroid: " + str(c_i)  # start from other random point?

            hist = np.sum(bin_vectors[vectors_i], axis=0)  # init a new histogram for majority voting
            half = len(vectors_i) / 2  # if the majority of vectors have a 1 on a certain spot it becomes a 1, half = n for majority

            hist[hist < half] = 0  # if majority has a 0 then it should be a 0
            hist[hist >= half] = 1  # the paper decides to assign a 1 or 0 in a tie (hist == half) at random, for reproducability we always assign a 1

            # check if the new centroid (hist) is different from the centroid
            diff = (hist != bin_centroids[c_i])
            if np.any(diff):
                centroids_changed = True
                centroids[c_i] = np.packbits(hist)

                overall_bits_changed += np.sum(diff)

                if last_runs:
                    print str(c_i) + " changed " + str(np.sum(diff)) + " bits"

        j += 1
        print("--- Run: %s changed %s bits in %.3f seconds" % (str(j), str(overall_bits_changed), time.time() - start_time))

        last_runs = (overall_bits_changed < 10)

    print("--- Overall kmajority %.3f seconds -" % (time.time() - overall_start_time))

    os.remove('_t_vectors')
    os.remove('_t_centroids')

    return centroids


def closest_dist(vecs, unique_i, centroids, k):
    filename = '_t_vectors_job_' + str(unique_i)  # create random filename for kmajority

    np.savetxt(filename, vecs, fmt='%i')

    output = check_output(["./kmajority", str(vecs.shape[1]), str(vecs.shape[0]), str(k), filename, '_t_centroids'])  # run c binary (is faster than pure python)
    cen_per_vec = np.fromstring(output, dtype=int, sep=' ')  # convert to numpy array

    os.remove(filename)  # clean up file after job finished

    return cen_per_vec


def compute_hamming_hist(vecs, vocabulary):
    # for readability don't use the c-binary for histogram calculations
    hist = np.zeros(len(vocabulary))

    bin_vocabulary = np.unpackbits(vocabulary.astype(np.ubyte), axis=1)
    bin_vectors = np.unpackbits(vecs.astype(np.ubyte), axis=1)

    for v_i, v in enumerate(bin_vectors):
        closest_c_i = None
        dists = np.sum(bin_vocabulary != v, axis=1)
        closest_c_i = np.argmin(dists)
        hist[closest_c_i] += 1

    # normalize hist
    hist /= len(vecs)

    return hist

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
