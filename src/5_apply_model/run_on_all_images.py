from joblib import Parallel, delayed
import os
import fnmatch


def run_model(file, iden):
    print "starting " + file
    os.system("python classify.py -i \"" + file + "\" -d" + str(iden))
    return

all_filenames = []
for root, dirnames, filenames in os.walk("../../results/naive_algorithm_per_img"):
    for filename in fnmatch.filter(filenames, 'original.png'):
        if root not in all_filenames:
            all_filenames.append(root + "/" + filename)

print all_filenames

results = Parallel(n_jobs=16)(delayed(run_model)(img_filename, iden) for iden, img_filename in enumerate(all_filenames))
