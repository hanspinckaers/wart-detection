import numpy as np
import fnmatch
import os
import sys
import math
import random
import shutil


def sample(n=5, should_seed=False, seed=1):
    filenames_per_subject = {}
    overall_n = 0
    overall_w = 0
    overall_c = 0
    overall_negative = 0
    for root, dirnames, filenames in os.walk(sys.argv[1]):
        for filename in fnmatch.filter(filenames, '*.png'):
            parts = filename.split(" - ")
            subject_id = parts[0]

            if 'originals' in root:
                continue
            if 'dubious' in root:
                continue

            if subject_id in filenames_per_subject:
                filenames_per_subject[subject_id].append(root + "/" + filename)
            else:
                filenames_per_subject[subject_id] = [root + "/" + filename]

            if 'cream' in root:
                overall_c += 1
            elif 'wart' in root:
                overall_w += 1
            else:
                overall_negative += 1

            overall_n += 1

    keys = []
    for key in filenames_per_subject:
        keys.append(key)

    n_participants = len(filenames_per_subject)
    test_n = int(math.ceil(n_participants / float(n)))

    if should_seed:
        np.random.seed(seed)

    random_indices = np.random.random_integers(0, high=len(filenames_per_subject) - 1, size=test_n)

    all_indices = np.arange(len(filenames_per_subject) - 1)

    bool_indices = np.ones(all_indices.shape)
    bool_indices[random_indices] = 0

    other_indices = all_indices[bool_indices.astype(bool)]

    n_photos = 0
    n_warts = 0
    n_cream = 0
    n_negative = 0

    filenames = []
    for i in random_indices:
        n_photos += len(filenames_per_subject[keys[i]])
        filenames.extend(filenames_per_subject[keys[i]])
        for filename in filenames_per_subject[keys[i]]:
            if 'cream' in filename:
                n_cream += 1
            elif 'wart' in filename:
                n_warts += 1
            else:
                n_negative += 1

    other_filenames = []

    for i in other_indices:
        other_filenames.extend(filenames_per_subject[keys[i]])

    perc = n_photos / float(overall_n) * 100

    perc_w = n_warts / float(n_photos) * 100
    perc_c = n_cream / float(n_photos) * 100
    perc_neg = n_negative / float(n_photos) * 100

    overall_w_perc = overall_w / float(overall_n) * 100
    overall_c_perc = overall_c / float(overall_n) * 100
    overall_neg_perc = overall_negative / float(overall_n) * 100

    print str(n_photos) + " of " + str(overall_n) + " = " + str(int(perc)) + "%"
    print str(int(perc_w)) + "% warts in test, " + str(int(overall_w_perc)) + "% in all"
    print str(int(perc_c)) + "% cream in test, " + str(int(overall_c_perc)) + "% in all"
    print str(int(perc_neg)) + "% negatives in test, " + str(int(overall_neg_perc)) + "% in all"

    return filenames, other_filenames


def divide_in(data, k, seed=1):
    data = np.array(data)
    random.seed(seed)
    data_len = len(data)
    k_len = int(round(data_len / k))
    parts = []
    for i in range(k):
        if i == k - 1:
            # last
            parts.append(data)
        else:
            data_len = len(data)
            # random_indices = np.random.random_integers(0, high=data_len - 1, size=k_len)
            random_indices = random.sample(range(data_len - 1), k_len)
            parts.append(data[random_indices])
            all_indices = np.arange(data_len)

            bool_indices = np.ones(all_indices.shape)
            bool_indices[random_indices] = 0
            data = data[bool_indices.astype(bool)]

    return parts


if __name__ == '__main__':
    test, train = sample(5., 1)
    for file in test:
        if 'cream' in file:
            shutil.copy(file, 'test_set/cream/')
        elif 'wart' in file:
            shutil.copy(file, 'test_set/wart/')
        else:
            shutil.copy(file, 'test_set/neg/')

    for file in train:
        if 'cream' in file:
            shutil.copy(file, 'train_set/cream/')
        elif 'wart' in file:
            shutil.copy(file, 'train_set/wart/')
        else:
            shutil.copy(file, 'train_set/neg/')
