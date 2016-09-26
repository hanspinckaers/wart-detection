import os
import fnmatch
import re

sigma_re = re.compile("u")

for root, dirnames, filenames in os.walk("output/"):
    for filename in fnmatch.filter(filenames, '*.out'):
        with open("output/" + filename, "r") as outputfile:
            data = outputfile.readlines()
            p = re.compile('[0-9.-]+')
            sigma = p.findall(data[4])[5]
            if float(sigma) < 1.:
                print data[4]
                print filename

            # print str(data[4])
            # m = sigma_re.match( )
            #    if m:
            #    print 'Match found: ', m.group()
                # print 'No match'

# train cream vs neg
# train wart vs neg

# train c+w vs neg
    # classifier c vs w

# cleanup data

# bayesian opt blur < 1
