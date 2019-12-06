import numpy as np

######################
### Synthetic Data ###
######################

def synthData1():
    '''
    Returns synthetic data for perceptron 1
    '''
    np.random.seed(sum([ord(c) for c in 'Neural Network']))
    m1 = 200
    x1 = np.random.random((2, m1))
    y1 = np.sum(x1.T*[2, 1], axis=1) > 1.5
    y1 = y1.reshape(1, m1)
    return [x1, y1]

def synthData2():
    '''
    Returns synthetic data for perceptron 2
    '''
    np.random.seed(sum([ord(c) for c in 'Neural Network']))
    m2 = 100
    x2 = np.random.randn(2, m2) + [[0], [3]]
    x2 = np.concatenate([x2, np.random.randn(2, m2)], axis=1)
    y2 = np.array([[1]*m2 + [0]*m2])
    return [x2, y2]