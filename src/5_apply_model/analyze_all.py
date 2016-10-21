# Hans Pinckaers, oct 2016
# This file generates a HTML table to 

import os
import fnmatch
import json

with open('../../results/model_results_all_imgs/filenames.json') as data_file:
    data = json.load(data_file)

with open('../../results/model_results_all_imgs/compliance_data_1423_raw.json') as data_file:
    compliance_data = json.load(data_file)


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
    participant = int(name.split(" - ")[0])
    key = str(participant)
    wart = int(name.split(" - ")[-1][:-4])
    day = name.split(" - ")[1]
    # if name not in names:
    #     continue
    answers = {}
    lines = tuple(open("./result_all/result_all_imgs/img_" + str(i) + ".txt", 'r'))
    number = data[i].split(" - ")[-1][:-17]
    score = float(lines[0])

    if key not in per_participant:
        per_participant[key] = {}

    if score > threshold:
        compliant_counter += 1
        if day not in per_participant[key]:
            per_participant[key][day] = 1
    else:
        if day not in per_participant[key]:
            per_participant[key][day] = 0

compliance_p_participants = {}
for comp in compliance_data:
    if comp["FIELD2"] not in compliance_p_participants:
        compliance_p_participants[comp["FIELD2"]] = []
        for c in range(41):
            compliance_p_participants[comp["FIELD2"]].append(int(comp["FIELD" + str(c + 3)]))


for participant in sorted(per_participant):
    for day in sorted(per_participant[participant]):
        compliance = per_participant[participant][day]
        for i in range(len(compliance_p_participants[participant])):
            if compliance_p_participants[participant][i] == 1:
                if compliance == 1:
                    compliance_p_participants[participant][i] = 2
                else:
                    compliance_p_participants[participant][i] = 3
                break

for participant in sorted(compliance_p_participants, key=int):
    print "<tr> <td align=\"right\"> " + str(participant) + " </td>"
    print "<td> " + str(participant) + " </td>"
    for i in range(len(compliance_p_participants[participant])):
        print "<td> " + str(compliance_p_participants[participant][i]) + " </td>"

    print("<td></td></tr>")
