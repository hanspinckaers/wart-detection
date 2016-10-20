from joblib import Parallel, delayed
import os
import fnmatch


def run_model(file):
    os.system("python classify.py -i " + file)
    return

all_filenames = []
for root, dirnames, filenames in os.walk("./run"):
    for filename in fnmatch.filter(filenames, 'original.png'):
        if root not in all_filenames:
            all_filenames.append(root)

print len(all_filenames)

results = Parallel(n_jobs=34)(delayed(run_model)(img_filename) for img_filename in enumerate(all_filenames))


