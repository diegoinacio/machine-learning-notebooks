import numpy as np

########################
### Distance metrics ###
########################

ldist = ['euclidian', 'manhattan', 'chebyshev', 'canberra', 'cosine']

class Distance(object):
    def __init__(self, metrica='euclidian'):
        super(Distance, self).__init__()
        if metrica not in ldist:
            raise ValueError('Metric does not exist! Choose between: {}'.format(ldist))
        self._metrica = metrica
    @property
    def metrica(self):
        return self._metrica
    @metrica.setter
    def metrica(self, m):
        if m not in ldist:
            raise ValueError('Metric does not exist! Choose between: {}'.format(ldist))
        self._metrica = m
    def distance(self, p, q):
        if self._metrica == 'manhattan':
            return np.sum(np.absolute(p - q), axis=1)
        if self._metrica == 'chebyshev':
            return np.max(np.absolute(p - q), axis=1)
        if self._metrica == 'canberra':
            num = np.absolute(p - q)
            den = np.absolute(p) + np.absolute(q)
            return np.sum(num/den, axis=1)
        if self._metrica == 'cosine':
            if p.ndim == 1:
                p = p[np.newaxis]
            num = np.sum(p*q, axis=1)
            den = np.sum(p**2, axis=1)**0.5
            den = den*np.sum(q**2, axis=1)**0.5
            return 1 - num/den
        return np.sum((p - q)**2, axis=1)**0.5


######################
### Synthetic Data ###
######################

def synthData1():
    '''
    Returns synthetic data for classification 1
    '''
    np.random.seed(sum([ord(c) for c in 'k-nearest neighbors']))
    N = 100
    Q1 = np.random.uniform(0, 1, N)
    Q2 = np.random.uniform(0, 1, N)
    CL = np.random.randint(0, 3, N)
    return [Q1, Q2, CL]

def synthData2():
    '''
    Returns synthetic data for classification 2
    '''
    np.random.seed(sum([ord(c) for c in 'k-nearest neighbors']))
    N = 2048
    P1 = np.random.uniform(0, 1, N)
    P2 = np.random.uniform(0, 1, N)
    return [P1, P2]

def synthData3():
    '''
    Returns synthetic data for regression 1
    '''
    np.random.seed(sum([ord(c) for c in 'k-nearest neighbors']))
    M, N = 32j, 32j
    X, Y = np.mgrid[-3:3:N*8, -3:3:M*8]
    Z = (1 - X + Y*X**3 + Y**5)*np.exp(-X**2 - Y**2)
    return [X, Y, Z]

def synthData4():
    '''
    Returns synthetic data for regression 2
    '''
    np.random.seed(sum([ord(c) for c in 'k-nearest neighbors']))
    N = 512
    Q1 = np.random.uniform(-3, 3, N)
    Q2 = np.random.uniform(-3, 3, N)
    VL = (1 - Q1 + Q2*Q1**3 + Q2**5)*np.exp(-Q1**2 - Q2**2)
    return [Q1, Q2, VL]

def synthData5():
    '''
    Returns synthetic data for regression 3
    '''
    np.random.seed(sum([ord(c) for c in 'k-nearest neighbors']))
    N = 2048
    P = np.random.uniform(-3, 3, (N, 2))
    xi, yi = np.mgrid[-3:3:1024j, -3:3:1024j]
    return [P, xi, yi]