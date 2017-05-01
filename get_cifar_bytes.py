#!/usr/bin/env python2

from num2verilog import nums2verilog1d as n2v
import numpy as np
from weights2bytes import np2hex

destfile = 'cifarbytes'

datadir = '/home/ricson/data/cifar_data/out/'

testdata = np.load(datadir+'traindata.npy')
testlabels = np.load(datadir+'trainlabels.npy')

N = 10
S = []
for i in range(N):
    img = testdata[i]
    #do some transposes here
    img = np.transpose(img, (2, 0, 1)).flatten()
    S.append(n2v(img, 'img%d' % i)+'\n')

f = open(destfile, 'w')
for i in range(N):
    f.write(S[i])

f.close()
