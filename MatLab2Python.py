import scipy.io as spio
import pickle
import numpy as np
"""
Below are taken from:
https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
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
Scrape data from matconvnet mat file 
"""
def rotate_weights(weights):
    new_matrx = []

    for i in range(np.size(weights, 2)):
        row0 = [weights[0][0][i], weights[1][0][i], weights[2][0][i], weights[3][0][i]]
        row1 = [weights[0][1][i], weights[1][1][i], weights[2][1][i], weights[3][1][i]]
        row2 = [weights[0][2][i], weights[1][2][i], weights[2][2][i], weights[3][2][i]]
        row3 = [weights[0][3][i], weights[1][3][i], weights[2][3][i], weights[3][3][i]]

        new_matrx.append(np.array([row0, row1, row2, row3]).transpose())

    return new_matrx


def model_creation(path_to_model):
    data = loadmat(path_to_model)
    model = []

    for i in data['net']['layers']:
        if i.type == 'conv':
            weights = rotate_weights(i.weights[0])
            model.append({'weights': weights, 'bias': i.weights[1], 'stride': i.stride, 'padding': i.pad, 'momentum': i.momentum,'lr': i.learningRate,'weight_decay': i.weightDecay})

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