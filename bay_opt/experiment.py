import os
from os import path
import sys
from subprocess import check_output

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


# run with ./spearmint/spearmint/bin/spearmint ./bay_opt/config.pb --drive=local --method=GPEIOptChooser --max-concurrent=2
def main(job_id, params):
    '''Params is a dictionary mapping from parameters specified in the
    config.json file to values that Spearmint has sugguested. This
    function will likely train a model on your training data, and
    return some function evaluated on your validation set. Small =
    better! Spearmint minimizes by default, so negate when it is so
    required.
    '''

    # parts = []
    # for root, dirnames, filenames in os.walk("train_set"):
    #     for filename in fnmatch.filter(filenames, '*.png'):
    #         part = filename.split(" - ")[0]
    #         if part not in parts:
    #             parts.append(part)

    # parts.sort()
    # dect_params = {
    #     "nfeatures": params['nfeatures'],
    #     "contrastThreshold": params['contrastThreshold'],
    #     "edgeThreshold": params['edgeThreshold'],
    #     "sigma": params['sigma']
    # }
    # model_params = {
    #     "C": 10.**params['svm_C'],
    #     "gamma": 10.**params['svm_gamma']
    # }
    print params

    # nfeatures = int(sys.argv[1])
    # bow_size = int(sys.argv[2])
    # svm_gamma = int(sys.argv[3])
    # edgeThreshold = int(sys.argv[4])
    # svm_C = int(sys.argv[5])
    # sigma = int(sys.argv[6])
    # contrastThreshold = int(sys.argv[7])

    # output = check_output(["python", "/Users/Hans/opencv3/train.py", str(params['nfeatures'][0]), str(params['bow_size'][0]), str(params['svm_gamma'][0]), str(params['edgeThreshold'][0]), str(params['svm_C'][0]), str(params['sigma'][0]), str(params['contrastThreshold'][0])])  # run c binary (is faster than pure python
    output = check_output(["python", "/Users/Hans/opencv3/train.py", str(params['svm_gamma'][0]), str(params['svm_C'][0]), str(params['weight'][0])])  # run c binary (is faster than pure python
    print output

    last_score = output.split("Final score:")[-1]
    score = float(last_score)

    # kappa = cross_validate_with_participants(5, parts, dect_params=dect_params, bow_size=params['bow_size'], model_params=model_params)
    return -score
