import os
import sys
import fnmatch

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import numpy as np

warts_per_participant = {}
negatives_per_participant = {}

for root, dirnames, filenames in os.walk(sys.argv[1]):
    for filename in fnmatch.filter(filenames, '*.png'):
        parts = filename.split(" - ")
        subject_id = parts[0]
        if 'wart' in root:
            if subject_id in warts_per_participant:
                warts_per_participant[subject_id] += 1
            else:
                warts_per_participant[subject_id] = 1
        elif 'negative' in root:
            if subject_id in negatives_per_participant:
                negatives_per_participant[subject_id] += 1
            else:
                negatives_per_participant[subject_id] = 1

ratio = {}
for key in warts_per_participant:
    if key in negatives_per_participant:
        ratio[key] = warts_per_participant[key] / float(warts_per_participant[key] + negatives_per_participant[key]) * 100

sum_pics = {}
for key in negatives_per_participant:
    sum_pics[key] = int(negatives_per_participant[key])

for key in warts_per_participant:
    if key in sum_pics:
        sum_pics[key] += int(warts_per_participant[key])
    else:
        sum_pics[key] = int(warts_per_participant[key])


def histogram(array, title, ylabel):
    plt.clf()

    x = []
    y = []
    for key in sorted(array, key=array.get, reverse=True):
        x.append(key)
        y.append(array[key])

    x_pos = np.arange(len(x))
    plt.title(title)

    rects = plt.bar(x_pos, y, color=cm.viridis(0.5))
    plt.xticks(x_pos, x)

    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(5))
    plt.gca().yaxis.grid(b=True, which='major', color=cm.gray(0.4), linestyle='-')
    plt.gca().yaxis.grid(b=True, which='minor', color=cm.gray(0.3), linestyle='--')

    plt.ylabel(ylabel)

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45, fontsize=10)

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, '%d' % int(height), ha='center', va='bottom', rotation=45, fontsize=10)

    autolabel(rects)

    fig = plt.gcf()
    fig.set_size_inches(1920 / 72, 1080 / 72)
    fig.savefig(title + '.png', dpi=72, bbox_inches='tight')


# histogram(warts_per_participant, "Classified warts per participant", "Number of pictures")
# histogram(negatives_per_participant, "Classified negatives per participant", "Number of pictures")
# histogram(ratio, "Percentage warts of all images per participant", "% of pictures is of wart")
histogram(sum_pics, "Number of pictures per participants", "Number of pictures")
