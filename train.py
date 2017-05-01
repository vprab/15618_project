#!/usr/bin/env python2

import numpy as np
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

datadir = '/home/ricson/data/cifar_data/out/'
outdir = 'weights'

def label2cat(labels, C = 10):
    N = len(labels)
    ohvs = np.zeros((N, C), dtype = np.float32)
    for i, label in enumerate(labels):
        ohvs[i][label] = 1.0
    return ohvs
        
        
def loaddata():
    imgs = np.load(datadir+'traindata.npy')[:10]
    labels = label2cat(np.load(datadir+'trainlabels.npy')[:10])
    return imgs, labels

def buildmodel(dodiv = False):
    if dodiv:
        divide = lambda: Lambda(lambda x: x/1024.0)
    else:
        divide = lambda: Lambda(lambda x: x)
        
    inp = Input(shape = (32, 32, 3))
    net = inp
    net = Convolution2D(filters = 8, kernel_size = 3, activation = 'relu', padding = 'same')(net)
    net = divide()(net)
    net = MaxPooling2D(pool_size = (2,2))(net) #16
    net = Convolution2D(filters = 16, kernel_size = 3, activation = 'relu', padding = 'same')(net)
    net = divide()(net)    
    net = MaxPooling2D(pool_size = (2,2))(net) #8
    net = Convolution2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same')(net)
    net = divide()(net)    
    net = MaxPooling2D(pool_size = (2,2))(net) #4
    net = Convolution2D(filters = 16, kernel_size = 1, activation = 'relu', padding = 'same')(net)
    net = divide()(net)    
    net = Convolution2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same')(net)
    net = divide()(net)    
    net = MaxPooling2D(pool_size = (2,2))(net) #2
    net = Convolution2D(filters = 16, kernel_size = 1, activation = 'relu', padding = 'same')(net)
    net = divide()(net)    
    net = Convolution2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same')(net)
    net = divide()(net)
    net = MaxPooling2D(pool_size = (2,2))(net) #1
    net = Convolution2D(filters = 16, kernel_size = 1, activation = 'relu', padding = 'same')(net)
    net = divide()(net)
    net = Convolution2D(filters = 10, kernel_size = 1, activation = None, padding = 'same')(net)
    net = Activation('softmax')(net)
    net = Flatten()(net)
    M = Model(inp, net)

    opt = Adam(lr = 0.001)
    M.compile(optimizer=opt, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
    return M

def train_and_get_weights():
    M = buildmodel()
    X, Y = loaddata()

    M.fit(X, Y, epochs = 2000, shuffle = True, batch_size = 32)

    weights = {}
    for layer in M.layers:
        weights[layer.name] = layer.get_weights()
    return weights

def save_weights(weightdict):
    for name, weights in weightdict.items():
        np.save(outdir+'/'+name, weights)

if __name__ == '__main__':
    save_weights(train_and_get_weights())
