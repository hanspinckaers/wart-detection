from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_score(z_func, title, z_axis, z_title, filename):
    n_points = 50

    spec = np.linspace(0., 1.0, n_points)
    sens = np.linspace(0., 1.0, n_points)

    x = np.zeros(len(spec) * len(sens))
    y = np.zeros(len(spec) * len(sens))
    z = np.zeros(len(spec) * len(sens))

    x, y = np.meshgrid(spec, sens)
    z = z_func(x, y)

    fig = plt.figure()
    fig.suptitle(title)
    plt.locator_params(nbins=10)
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.viridis, alpha=1)
    ax.view_init(elev=20., azim=-135.5)

    ax.set_xlabel('specificity')
    ax.set_ylabel('sensitivity')
    ax.set_zlabel(z_title)

    ax.w_xaxis.gridlines.set_lw(1.0)
    ax.w_yaxis.gridlines.set_lw(1.0)
    ax.w_zaxis.gridlines.set_lw(1.0)

    ax.w_xaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.4)}})
    ax.w_yaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.4)}})
    ax.w_zaxis._axinfo.update({'grid': {'color': (0, 0, 0, 0.4)}})

    fig.set_size_inches(25, 16)

    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_zticks(z_axis)

    ax.xaxis.labelpad = 60
    ax.yaxis.labelpad = 60
    ax.zaxis.labelpad = 60

    ax.tick_params(axis='z', which='major', pad=23)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.tick_params(axis='y', which='major', pad=15)

    mpl.rcParams.update({'font.size': 36})

    fig.subplots_adjust(top=1)

    plt.savefig(filename, dpi=72, bbox_inches='tight')


# plot_score(lambda spec, sens: (spec * 3 + sens) / 4, 'Weighted mean of specifity (w = 3) and sensitivity (w = 1)', np.arange(0, 1.2, 0.2), 'weighted mean', "weighted_normal_mean.png")
plot_score(lambda spec, sens: spec**1.75 * sens**0.75, 'Product using exponents ($\mathregular{spec^{1.75} * sens^{0.75}}$)', np.arange(0, 1.1, 0.1), 'exponent weighted', "exp_weighted_mean.png")
# plot_score(lambda spec, sens: spec + sens - 1, 'Youden\'s J statistic', np.arange(-1.0, 1.2, 0.2), 'Youden\'s J statistic', "youden.png")
# plot_score(lambda spec, sens: (3 + 1) * (spec * sens) / (spec + 3 * sens), 'Weighted harmonic mean of specifity (w = 3) and sensitivity (w = 1)', np.arange(0, 1.1, 0.1), 'weighted mean', "weighted_mean.png")
# plot_score(lambda spec, sens: sens / (sens + (1 - spec)), 'Positive Predictive Value', np.arange(0, 1.1, 0.1), 'Positive Predictive Value', "ppv.png")
# plot_score(lambda spec, sens: spec / (spec + (1 - sens)), 'Negative Predictive Value', np.arange(0, 1.1, 0.1), 'Negative Predictive Value', "npv.png")
# 
# 
# def cohen(sens, spec):
#     a = sens
#     b = 1 - spec
#     d = spec
#     c = 1 - sens
# 
#     total = (a + b + c + d)
# 
#     po = (a + d) / total
#     ma = ((a + b) * (a + c)) / (a + b + c + d)
#     mb = ((c + d) * (b + d)) / (a + b + c + d)
#     pe = (ma + mb) / total
#     cohen = (po - pe) / (1 - pe)
#     return cohen
# 
# plot_score(cohen, 'Cohen\'s kappa', np.arange(-1.0, 1.2, 0.2), 'Cohen\'s kappa', "cohen.png")
# plot_score(lambda spec, sens: (sens + spec) / (sens + spec + (1 - sens) + (1 - spec)), 'Accuracy', np.arange(0, 1.1, 0.1), 'accuracy', "accuracy.png")
# 
# 
# def f1(sens, spec):
#     a = sens
#     b = 1 - spec
#     c = 1 - sens
# 
#     f1 = (2) * a / ((2 * a) + (1 * c) + b)
#     return f1
# 
# plot_score(f1, '$\mathregular{F_1}$ score (harmonic mean of recall and precision)', np.arange(0.0, 1.1, 0.1), '$\mathregular{F_1}$ score', "f1_score.png")
# 
# 
# def f05(sens, spec):
#     a = sens
#     b = 1 - spec
#     c = 1 - sens
# 
#     f05 = (1 + 0.5**2) * a / ((1 + 0.5**2 * a) + (0.5**2 * c) + b)
#     return f05
# 
# plot_score(f05, '$\mathregular{F_{0.5}}$ score (weighs recall lower than precision)', np.arange(0.0, 1.1, 0.1), '$\mathregular{F_{0.5}}$ score', "f05_score.png")
# 
# 
# def f2(sens, spec):
#     a = sens
#     b = 1 - spec
#     c = 1 - sens
# 
#     f05 = (1 + 2**2) * a / ((1 + 2**2 * a) + (2**2 * c) + b)
#     return f05
# 
# plot_score(f2, '$\mathregular{F_2}$ score (weighs recall higher than precision)', np.arange(0.0, 1.1, 0.1), '$\mathregular{F_2}$ score', "f2_score.png")
