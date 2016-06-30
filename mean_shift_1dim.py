# -*- coding: utf-8 -*-
"""
Author: Yuri Mejia Miranda and Hans Pinckaers
Date: 20160629
Goal: Understand mean shift in 1 dimension.
Info:
K = Gaussian Kernel in 1 dimension
h = scaling factor (size of window)
"""
from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Gaussian Kernel
def Kernel(x):
    return math.exp(-math.pow(x,2)/math.sqrt(2))

#Shift and scale kernel
def Kernel_shift(x, vec, h, k):
    return [k((x-v)/h) for v in vec]

#Negative derivate of Kernel
def Kernel_g(x):
    return -Kernel(x)

#Density function
def f(x, vec, h):
    return np.mean(Kernel_shift(x,vec,h, Kernel))/h
    # Explanation of this function for Hans:
    # taking the mean is the same as taking the sum and multiplying with 1/n
    # since this is one dimension h^d = h

#Calculate shift towards the mean of the pdf 
def cal_mean_shift(x, vec, h):
    # we need a normalization factor?
    mean_values = np.sum([v*Kernel_g((x-v)/h) for v in vec])
    mean_weights = np.sum([Kernel_g((x-v)/h) for v in vec])
    return mean_values/mean_weights - x 

# Shift until convergence (or shift less than 0.01)
# the function returns all the shifts the algorithm took 
# the last one is the last maximum density
def mean_shift(x, vec, h):
    shifts = []
    shift = 1 # 1 is placehold value to get into the while loop
    # threshold on a shift of smaller than 0.01
    while abs(shift) > 0.01:
        shift = cal_mean_shift(x, vec, h)
        x += shift
        shifts.append(x)
    return shifts

#Vector of points
vec = [1,2,3,4,5,7,8,9,10,
        11,12,13,14,15,16,17,18,19,20,
        21,22,23,24,25,26,27,28,29,30,
        31,32,34,35,36,37,38,39,40,
        51,52,53,55,56,57,58,59,60,
        61,62,63,64,65,66,67,68,69,70,
        71,72,74,75,76,77,78,
        80,80,80,80,81,80,81,
        82,83,84,85,86,87,88,89,90,
        91,92,93,94,95,96,97,99,100,110,111,115,102,120,125,140]
vec = sorted(vec)

#Save figures
#For different window sizes, compute density function
with PdfPages('gaussian_kernel_1dim_shift.pdf') as pp:
    for h in [3, 5, 10, 20, 40]:
        y_vec = [f(x, vec, h) for x in vec]
        plt.plot(vec, y_vec, 'bo')
        
        # calculate mode of pdf starting from the beginning of the vector
        beginning = vec[5];
        s_from_beginning = mean_shift(beginning, vec, h)
        plt.axvline(x=s_from_beginning[len(s_from_beginning)-1], linewidth=1, color='r')
        plt.plot(s_from_beginning, [0.035 for s in s_from_beginning], 'r+')
        
        # calculate mode of pdf start from the ending of the vector
        ending = vec[len(vec)-2]
        s_from_ending = mean_shift(ending, vec, h)
        plt.axvline(x=s_from_ending[len(s_from_ending)-1], linewidth=1, color='r')
        plt.plot(s_from_ending, [0.035 for s in s_from_ending], 'r+')

        plt.ylim(0, 0.04)
        plt.xlabel('Points')
        plt.ylabel('Density function: f')
        plt.title('Gaussian kernel in 1 dimension with h = %s'%(h))
        plt.annotate('mean_shift_1dim.py',
                (0,0), (270, -25), xycoords='axes fraction', textcoords='offset points', va='top')
        pp.savefig()
        plt.close()

