#!/usr/bin/env python2

import numpy as np
import keras
from train import buildmodel, loaddata

datadir = 'weights/'


def uncat(vec):
    return np.argmax(vec)

if __name__ == '__main__':
    M = buildmodel(True)
    for layer in M.layers:
        wb = np.load(datadir+layer.name+'.npy')
        if not len(wb):
            continue
        
        weights, biases = wb
        #do modification
        weights *= 1024
        biases *= 1024
        weights = weights.astype(np.int32)
        biases = biases.astype(np.int32)        
        #finish modification

        #weights = np.transpose(weights, (1,0,2,3))
        
        layer.set_weights((weights, biases))

    X, Y = loaddata()

    M.layers.pop()
    M.layers.pop()
    M.outputs = [M.layers[-1].output]
    M.layers[-1].outbound_nodes = []
    
    pred = M.predict_on_batch(X[:10])
    print pred
    print Y[:10]
    print map(uncat, pred)
    print map(uncat, Y[:10])
