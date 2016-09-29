import numpy as np
import os
import fnmatch
import shutil

names = []
for root, dirnames, filenames in os.walk("../test_set"):
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
    shutil.copy(original, '../humanvsmodel/images/')
    os.rename('../humanvsmodel/images/original.png', '../humanvsmodel/images/' + str(counter) + '.png')
    print original

    # shutil.copy(file, '../humanvsmodel/images/')
    # os.rename(file, '../humanvsmodel/images/' + str(i) + '.png')

# 070 - 2015-03-26 - 9 - roi-2.png
# ../run/images5/user_070/date_1427404189/070 - 2015-03-26 - 9.png/original.png
# 056 - 2015-03-17 - 7 - roi-2.png
# ../run/images/user_056/date_1426580648/056 - 2015-03-17 - 7.png/original.png
# 030 - 2015-03-31 - 3 - roi-3.png
# ../run/images4/user_030/date_1427836651/030 - 2015-03-31 - 3.png/original.png
# 070 - 2015-02-15 - 6 - roi-1.png
# ../run/images5/user_070/date_1424031356/070 - 2015-02-15 - 6.png/original.png
# 003 - 2015-02-16 - 2 - roi-1.png
# ../run/images2/user_003/date_1424069163/003 - 2015-02-16 - 2.png/original.png
# 071 - 2015-02-21 - 7 - roi-2.png
# ../run/images5/user_071/date_1424512447/071 - 2015-02-21 - 7.png/original.png
# 030 - 2015-04-15 - 5 - roi-2.png
# ../run/images4/user_030/date_1429134726/030 - 2015-04-15 - 5.png/original.png
# 012 - 2015-02-13 - 5 - roi-1.png
# ../run/images3/user_012/date_1423826467/012 - 2015-02-13 - 5.png/original.png
# 030 - 2015-04-14 - 2 - roi-3.png
# ../run/images4/user_030/date_1429046324/030 - 2015-04-14 - 2.png/original.png
# 030 - 2015-04-23 - 5 - roi-3.png
# ../run/images4/user_030/date_1429823975/030 - 2015-04-23 - 5.png/original.png
# 030 - 2015-05-03 - 2 - roi-1.png
# ../run/images4/user_030/date_1430676130/030 - 2015-05-03 - 2.png/original.png
# 070 - 2015-02-14 - 6 - roi-3.png
# ../run/images5/user_070/date_1423938548/070 - 2015-02-14 - 6.png/original.png
# 039 - 2015-05-23 - 1 - roi-1.png
# ../run/images4/user_039/date_1432336684/039 - 2015-05-23 - 1.png/original.png
# 070 - 2015-02-14 - 8 - roi-1.png
# ../run/images5/user_070/date_1423938548/070 - 2015-02-14 - 8.png/original.png
# 051 - 2015-04-10 - 8 - roi-2.png
# ../run/images/user_051/date_1428682650/051 - 2015-04-10 - 8.png/original.png
# 070 - 2015-03-08 - 9 - roi-1.png
# ../run/images5/user_070/date_1425853369/070 - 2015-03-08 - 9.png/original.png
# 056 - 2015-03-11 - 7 - roi-1.png
# ../run/images/user_056/date_1426058437/056 - 2015-03-11 - 7.png/original.png
# 040 - 2015-05-08 - 2 - roi-2.png
# ../run/images4/user_040/date_1431040840/040 - 2015-05-08 - 2.png/original.png
# 044 - 2015-05-11 - 6 - roi-2.png
# ../run/images/user_044/date_1431317149/044 - 2015-05-11 - 6.png/original.png
# 074 - 2015-02-23 - 7 - roi-2.png
# ../run/images5/user_074/date_1424690606/074 - 2015-02-23 - 7.png/original.png
# 030 - 2015-04-12 - 1 - roi-1.png
# ../run/images4/user_030/date_1428872130/030 - 2015-04-12 - 1.png/original.png
# 051 - 2015-04-05 - 7 - roi-1.png
# ../run/images/user_051/date_1428248012/051 - 2015-04-05 - 7.png/original.png
# 074 - 2015-02-21 - 7 - roi-2.png
# ../run/images5/user_074/date_1424517184/074 - 2015-02-21 - 7.png/original.png
# 040 - 2015-05-05 - 1 - roi-2.png
# ../run/images4/user_040/date_1430856264/040 - 2015-05-05 - 1.png/original.png
# 071 - 2015-03-19 - 7 - roi-3.png
# ../run/images5/user_071/date_1426786882/071 - 2015-03-19 - 7.png/original.png
# 035 - 2015-04-03 - 8 - roi-2.png
# ../run/images4/user_035/date_1428088244/035 - 2015-04-03 - 8.png/original.png
# 040 - 2015-05-27 - 2 - roi-3.png
# ../run/images4/user_040/date_1432679880/040 - 2015-05-27 - 2.png/original.png
# 007 - 2015-03-18 - 1 - roi-1.png
# ../run/images2/user_007/date_1426715554/007 - 2015-03-18 - 1.png/original.png
# 070 - 2015-03-25 - 9 - roi-1.png
# ../run/images5/user_070/date_1427320957/070 - 2015-03-25 - 9.png/original.png
# 039 - 2015-05-15 - 1 - roi-3.png
# ../run/images4/user_039/date_1431642365/039 - 2015-05-15 - 1.png/original.png
# 035 - 2015-03-11 - 8 - roi-1.png
# ../run/images4/user_035/date_1426109209/035 - 2015-03-11 - 8.png/original.png
# 051 - 2015-05-06 - 10 - roi-1.png
# ../run/images/user_051/date_1430931365/051 - 2015-05-06 - 10.png/original.png
# 074 - 2015-02-23 - 7 - roi-3.png
# ../run/images5/user_074/date_1424690606/074 - 2015-02-23 - 7.png/original.png
# 012 - 2015-03-02 - 2 - roi-1.png
# ../run/images3/user_012/date_1425297568/012 - 2015-03-02 - 2.png/original.png
# 039 - 2015-04-30 - 1 - roi-1.png
# ../run/images4/user_039/date_1430345168/039 - 2015-04-30 - 1.png/original.png
# 056 - 2015-03-31 - 6 - roi-1.png
# ../run/images/user_056/date_1427785895/056 - 2015-03-31 - 6.png/original.png
# 070 - 2015-03-25 - 9 - roi-1.png
# ../run/images5/user_070/date_1427320957/070 - 2015-03-25 - 9.png/original.png
# 003 - 2015-02-01 - 2 - roi-2.png
# ../run/images2/user_003/date_1422786891/003 - 2015-02-01 - 2.png/original.png
# 070 - 2015-02-23 - 8 - roi-2.png
# ../run/images5/user_070/date_1424728482/070 - 2015-02-23 - 8.png/original.png
# 056 - 2015-03-01 - 6 - roi-1.png
# ../run/images/user_056/date_1425196382/056 - 2015-03-01 - 6.png/original.png
# 003 - 2015-02-25 - 1 - roi-2.png
# ../run/images2/user_003/date_1424845010/003 - 2015-02-25 - 1.png/original.png
# 039 - 2015-05-28 - 2 - roi-2.png
# ../run/images4/user_039/date_1432765147/039 - 2015-05-28 - 2.png/original.png
# 071 - 2015-02-20 - 6 - roi-1.png
# ../run/images5/user_071/date_1424458567/071 - 2015-02-20 - 6.png/original.png
# 070 - 2015-02-14 - 8 - roi-2.png
# ../run/images5/user_070/date_1423938548/070 - 2015-02-14 - 8.png/original.png
# 051 - 2015-04-09 - 7 - roi-2.png
# ../run/images/user_051/date_1428593793/051 - 2015-04-09 - 7.png/original.png
# 056 - 2015-03-05 - 7 - roi-2.png
# ../run/images/user_056/date_1425540647/056 - 2015-03-05 - 7.png/original.png
# 070 - 2015-03-12 - 8 - roi-1.png
# ../run/images5/user_070/date_1426199425/070 - 2015-03-12 - 8.png/original.png
# 051 - 2015-03-28 - 7 - roi-2.png
# ../run/images/user_051/date_1427561623/051 - 2015-03-28 - 7.png/original.png
# 016 - 2015-03-02 - 2 - roi-2.png
# ../run/images3/user_016/date_1425288847/016 - 2015-03-02 - 2.png/original.png
# 070 - 2015-03-23 - 7 - roi-1.png
# ../run/images5/user_070/date_1427150985/070 - 2015-03-23 - 7.png/original.png
# 


# 1, 8, 32, 37, 39, 50 is double
