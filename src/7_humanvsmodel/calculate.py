import os
import fnmatch
import numpy as np
import pudb


""" Computes the Fleiss' Kappa value as described in (Fleiss, 1971) """
DEBUG = True

def computeKappa(mat):
    """ Computes the Kappa value
    @param n Number of rating per subjects (number of human raters)
    @param mat Matrix[subjects][categories]
    @return The Kappa value """
    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])

    if DEBUG:
        print n, "raters."
        print N, "subjects."
        print k, "categories."

    # Computing p[]
    p = [0.0] * k
    for j in xrange(k):
        p[j] = 0.0
        for i in xrange(N):
            p[j] += mat[i][j]
            p[j] /= N * n
            if DEBUG:
                print "p =", p

    # Computing P[]
    P = [0.0] * N
    for i in xrange(N):
        P[i] = 0.0
        for j in xrange(k):
            P[i] += mat[i][j] * mat[i][j]
            P[i] = (P[i] - n) / (n * (n - 1))
            if DEBUG:
                print "P =", P

    # Computing Pbar
    Pbar = sum(P) / N
    if DEBUG:
        print "Pbar =", Pbar

    # Computing PbarE
    PbarE = 0.0
    for pj in p:
        PbarE += pj * pj
        if DEBUG:
            print "PbarE =", PbarE

    kappa = (Pbar - PbarE) / (1 - PbarE)
    if DEBUG:
        print "kappa =", kappa

    return kappa


def checkEachLineCount(mat):
    """ Assert that each line has a constant number of ratings
    @param mat The matrix checked
    @return The number of ratings
    @throws AssertionError If lines contain different number of ratings """
    n = sum(mat[0])

    assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    return n

# results files
all_filenames = []
for root, dirnames, filenames in os.walk('./results'):
    for filename in fnmatch.filter(filenames, '*.csv'):
        if root not in all_filenames:
            all_filenames.append(root)

# load results
results = []
for file in filenames:
    answers = {}
    lines = tuple(open('./results/' + file, 'r'))
    for line in lines:
        info = line.split(',')
        question = info[1]
        answer = info[-1].split('\n')[0]
        if answer == 'cream':
            answer = 1
        else:
            answer = 0
        answers[int(question)] = answer

    if question == '50':
        results.append(answers)

mat_results = np.zeros((50, len(results)))
for i, result in enumerate(results):
    for q in range(50):
        mat_results[q, i] = result[q + 1]

sums = np.sum(mat_results, axis=1)
kappa_results = np.zeros((mat_results.shape[0], 2))
for i in range(mat_results.shape[0]):
    kappa_results[i, 0] = sums[i]
    kappa_results[i, 1] = mat_results.shape[1] - sums[i]

kappa = computeKappa(kappa_results)


print sums / 13. * 100
for s in sums:
    print s / 13.
print mat_results

print "KAPPA:"
print kappa


