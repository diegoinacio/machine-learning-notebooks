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

def synthData():
    '''
    Returns synthetic data for clustering algorithms
    '''
    np.random.seed(sum([ord(e) for e in 'clustering']))

    # Gaussian mixture
    x1 = np.array([])
    y1 = np.array([])
    for i in range(3):
        mu1 = np.random.uniform(-2, 2)
        mu2 = np.random.uniform(-2, 2)
        sigma = np.random.uniform(0, 0.5)
        x1 = np.concatenate([x1, np.random.normal(mu1, sigma, 32)])
        y1 = np.concatenate([y1, np.random.normal(mu2, sigma, 32)])

    # Ring with center
    N = 1000
    t = np.random.uniform(0, 2*np.pi, N)
    x2 = np.cos(t)*10 + np.random.random(N)
    y2 = np.sin(t)*10 + np.random.random(N)
    x2 = np.concatenate([x2, np.random.uniform(-2.5, 2.5, N//4)])
    y2 = np.concatenate([y2, np.random.uniform(-2.5, 2.5, N//4)])

    return [x1, y1, x2, y2]