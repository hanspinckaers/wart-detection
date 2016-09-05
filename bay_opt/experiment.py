from ..analyze_training import cross_validate_with_participants
import fnmatch
import os


def main(job_id, params):
    '''Params is a dictionary mapping from parameters specified in the
    config.json file to values that Spearmint has sugguested. This
    function will likely train a model on your training data, and
    return some function evaluated on your validation set. Small =
    better! Spearmint minimizes by default, so negate when it is so
    required.
    '''

    parts = []
    for root, dirnames, filenames in os.walk("train_set"):
        for filename in fnmatch.filter(filenames, '*.png'):
            part = filename.split(" - ")[0]
            if part not in parts:
                parts.append(part)

    parts.sort()
    dect_params = {
        "n_features": params['n_features'],
        "contrastThreshold": params['contrastThreshold'],
        "edgeThreshold": params['edgeThreshold'],
        "sigma": params['sigma']
    }
    model_params = {
        "C": params['C'],
        "gamma": params['gamma']
    }
    print params
    kappa = cross_validate_with_participants(5, parts, dect_params=dect_params, bow_size=params['bow_size'], model_params=model_params)
    return -kappa
