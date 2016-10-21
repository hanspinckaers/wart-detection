import numpy as np
import os
import fnmatch
import shutil

names = []
for root, dirnames, filenames in os.walk("../train_set"):
    for filename in fnmatch.filter(filenames, '*.png'):
        names.append(filename)

np.random.seed(0)

random_indices = np.random.random_integers(0, high=len(names) - 1, size=60)
counter = 0
for i in random_indices:
    counter = counter + 1
    file = names[i]
    print file
    foldername = file[:-12].strip() + ".png"
    original = None
    for root, dirnames, filenames in os.walk("../run"):
        for filename in fnmatch.filter(filenames, '*.png'):
            if foldername in root:
                original = root + "/original.png"
                break
        if original is not None:
            break
    shutil.copy(original, '../humanvsmodel/examples/')
    os.rename('../humanvsmodel/examples/original.png', '../humanvsmodel/examples/' + str(counter) + '.png')
    print original

    # shutil.copy(file, '../humanvsmodel/images/')
    # os.rename(file, '../humanvsmodel/images/' + str(i) + '.png')
