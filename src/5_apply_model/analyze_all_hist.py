import os
import fnmatch
import json
import matplotlib.pyplot as plt

with open('filenames.json') as data_file:
    data = json.load(data_file)

# results files
all_filenames = []
for root, dirnames, filenames in os.walk('./result_all'):
    for filename in fnmatch.filter(filenames, '*.txt'):
        all_filenames.append(root + "/" + filename)

names = []
for root, dirnames, filenames in os.walk("./train_set"):
    for filename in fnmatch.filter(filenames, '*.png'):
        names.append(filename.split("oi")[0][:-4] + ".png")


# load results
compliant_counter = 0

results = []

pos_scores = []
neg_scores = []

fp = 0.
fn = 0.
tp = 0.
tn = 0.

per_participant = {}

threshold = 9.3

for i in range(8675):
    name = data[i].split("/")[5]
    if name not in names:
        continue
    answers = {}
    lines = tuple(open("./result_all/result_all_imgs/img_" + str(i) + ".txt", 'r'))
    number = data[i].split(" - ")[-1][:-17]
    score = float(lines[0])
    if score > threshold:
        compliant_counter += 1

    if number == "1" or number == "6":
        score = float(lines[0])
        neg_scores.append(score)
        if score < threshold:
            tn += 1.
        else:
            fp += 1.
    else:
        score = float(lines[0])
        pos_scores.append(score)
        if score > threshold:
            tp += 1.
        else:
            fn += 1.

print "sensitivity: " + str(tp / (tp + fn)) + " specificity: " + str(tn / (tn + fp))
print str(len(neg_scores)) + " negatives"
print str(len(pos_scores)) + " positives"

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# the histogram of the data
n, bins, patches = plt.hist(pos_scores, bins, normed=0, facecolor='green', alpha=0.75)

# add a 'best fit' line
plt.xlabel('Images')
plt.ylabel('Scores')
plt.axis([0, 50, 0, 800])
plt.grid(True)

n, bins, patches = plt.hist(neg_scores, bins, normed=0, facecolor='red', alpha=0.75)

# add a 'best fit' line
plt.xlabel('Images')
plt.ylabel('Scores')
plt.axis([0, 40, 0, 1500])
plt.grid(True)

plt.show()

print compliant_counter
print len(all_filenames)
