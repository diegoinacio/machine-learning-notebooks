import numpy as np

def correlation(X, Y):
    '''
    Correlation function
    '''
    N = X.size
    mu_X = X.sum()/N
    mu_Y = Y.sum()/N
    cov = ((X - mu_X)*(Y - mu_Y)).sum()/N
    sigma_X = X.std()
    sigma_Y = Y.std()
    return cov/(sigma_X*sigma_Y)

######################
### Synthetic Data ###
######################

def synthData1():
    '''
    Returns synthetic data for linear regression simple
    '''
    np.random.seed(sum([ord(c) for c in 'Regression']))
    N = 20
    x = np.linspace(0, 1, N)
    yA = x
    yB = x + (np.random.random(N)*2 - 1)*0.15
    yC = x + (np.random.random(N)*2 - 1)*0.5
    yD = np.random.random(N)
    return [x, yA, yB, yC, yD]

def synthData2(M):
    '''
    Returns synthetic data for linear regression multiple
    '''
    np.random.seed(sum([ord(c) for c in 'Regression']))
    N = complex(0, M)
    s, t = np.mgrid[-1:1:N, -1:1:N]
    x1 = s.reshape(1, -1)[0]
    x2 = t.reshape(1, -1)[0]
    y = (x1 + x2)*0.5 + (np.random.random(int(N.imag**2))*2 - 1)*0.75
    return [s, t, x1, x2, y]

def synthData3():
    '''
    Returns synthetic data for linear regression gradient descent
    '''
    np.random.seed(sum([ord(c) for c in 'Regression']))
    N = 20
    x = np.linspace(0, 1, N)
    x_ = np.linspace(-5, 5, N)
    y = x + (np.random.random(N)*2 - 1)*0.25
    return [x, x_, y]

def synthData4():
    '''
    Returns synthetic data for linear regression non-linear analysis
    Anscombes quartet
    '''
    x1 = [10,   8,    13,   9,    11,   14,   6,    4,    12,   7,    5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84,4.82, 5.68]
    x2 = [10,   8,    13,   9,    11,   14,   6,    4,    12,   7,    5]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    x3 = [10,   8,    13,   9,    11,   14,   6,    4,    12,   7,    5]
    y3 = [7.46, 6.77, 12.74,7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8,    8,    8,    8,    8,    8,    8,    19,   8,    8,    8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50,5.56, 7.91, 6.89]
    return np.array([x1, y1, x2, y2, x3, y3, x4, y4])

def synthData5():
    '''
    Returns synthetic data for logistic regression
    '''
    np.random.seed(sum([ord(c) for c in 'Regression']))
    N = 512
    x1 = np.random.normal(1, 0.3, N//2)
    x1 = np.concatenate([x1, np.random.normal(2, 0.3, N//2)])
    x2 = np.random.normal(0, 0.3, N//2)
    x2 = np.concatenate([x2, np.random.normal(0.25, 0.3, N//2)])
    y = np.zeros(N//2, np.int8)
    y = np.concatenate([y, np.ones(N//2, np.int8)])
    return [x1, x2, y]

def synthData6():
    '''
    Returns synthetic data for polynomial regression
    '''
    np.random.seed(sum([ord(c) for c in 'Regression']))
    N = 21
    x = np.random.uniform(-3, 3, N)
    y = x**3 - 3*x**2 + x + 1 + np.random.uniform(-3, 3, N)
    return [x, y]