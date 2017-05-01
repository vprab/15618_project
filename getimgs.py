#!/usr/bin/env python2

import cPickle
import numpy as np
from scipy.misc import imread, imsave

datadir = '/home/ricson/data/cifar_data/'
outdir = '/home/ricson/data/cifar_data/out/'

def unpickle(file):
    import cPickle
    with open(datadir+file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

datas = []
labels = []
for i in range(1, 6):
    dict_ = unpickle('data_batch_%d' % i)
    data = np.transpose(np.reshape(dict_['data'], (10000, 3, 32, 32)), (0, 2, 3, 1))
    datas.append(data)
    labels.extend(dict_['labels'])
datas = np.concatenate(datas, axis = 0)
labels = np.array(labels)

np.save(outdir+'traindata', datas)
np.save(outdir+'trainlabels', labels)

dict_ = unpickle('test_batch')
data = np.transpose(np.reshape(dict_['data'], (10000, 3, 32, 32)), (0, 2, 3, 1))
np.save(outdir+'testdata', data)
np.save(outdir+'testlabels', dict_['labels'])
