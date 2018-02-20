import scipy.io as spio
import pickle
import numpy as np
"""
Below are taken from:
https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

Needed to properly scrape the dictionaries from matconvnet model 
"""
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


"""
Now, we need to Scrape relevant data 
from model dictionary 
"""
def model_creation(path_to_model):
    data = loadmat(path_to_model)
    model = []

    for i in data['net']['layers']:
        if i.type == 'conv':
            model.append({'weights': i.weights[0], 'bias': i.weights[1], 'stride': i.stride, 'padding': i.pad, 'momentum': i.momentum,'lr': i.learningRate,'weight_decay': i.weightDecay})

        # RELU
        else:
            if i.type == 'softmaxloss':
                data = i.__dict__['class'] # matconv model helpfully names something a class, which causes errors in python
                model.append(['softmax', data])

            else:
                model.append('relu')

    pickle.dump(model, open("model.p", "wb" ))


    print('Successfully scrapped data from MatConvNet Model')
    print('Generating TensorFlow Model')


model_creation('net.mat')