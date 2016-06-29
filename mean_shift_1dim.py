# -*- coding: utf-8 -*-
"""
Author: Yuri Mejia Miranda
Date: 20160629
Goal: Understand mean shift in 1 dimension.
Info:
K = Gaussian Kernel in 1 dimension
h = scaling factor (size of window)
"""
import numpy as np
import math
from __future__ import division
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


#Gaussian Kernel
def Kernel(x):
    return math.exp(-math.pow(x,2)/math.sqrt(2))
    
#Shift and scale kernel
def Kernel_shift(x, vec, h):       
    return [Kernel((x-v)/h) for v in vec]

#Density function    
def f(x, vec, h):
    return np.mean(Kernel_shift(x,vec,h))/h

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
with PdfPages('C:\Users\ymejia\Desktop\GitHub\wart-detection\gaussian_kernel_1dim.pdf') as pp:    
    #For different window sizes, compute density function
    for h in [3, 5, 10, 20, 40]:    
        y_vec = [f(x, vec, h) for x in vec]
        plt.plot(vec, y_vec, 'bo')
        plt.ylim(0, 0.04)
        plt.xlabel('Points')
        plt.ylabel('Density function: f')
        plt.title('Gaussian kernel in 1 dimension with h = %s'%(h))
        plt.annotate('mean_shift_1dim.py',
                     (0,0), (270, -25), xycoords='axes fraction', textcoords='offset points', va='top')
        pp.savefig()
        plt.close()
       
