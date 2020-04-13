# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:28:35 2020

@author: wei.zheng
"""

import pylab as pl
import scipy.misc as sp
img = sp.face()
pl.imshow(img, cmap = pl.cm.gray)
pl.colorbar()

print("Hello python")

import sys
print('================Python import mode==========================')
print ('命令行参数为:')
for i in sys.argv:
    print (i)
print ('\n python 路径为',sys.path)

import matplotlib.pyplot as plt
import numpy as np
import math
x = np.linspace(1, 5, 5, endpoint=True)
y = np.power(x, 2)

plt.plot(x, y, ls="-", lw="2", label="plot figure")

plt.legend()
plt.show()

if True:
    print("")
else:
    print("")
    
a = 10
while a > 0:
    print(a)
    a = a-1
    
array= [1,2,4,5]
for i in range(len(array)):
    print(i, array[i])
    
    
