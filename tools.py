import numpy as np


def design_matrix(x,y,degree):

    #fitting without intercept
    colums = []

    colums.append(x)
    colums.append(y)

    if degree>1:
        colums.append(x**2)
        colums.append(y**2)
        colums.append(x*y)

    if degree>2:
        colums.append(x**3)
        colums.append(y**3)
        colums.append(y**2*x)
        colums.append(x**2*y)

    if degree>3:
        colums.append(x**4)
        colums.append(y**4)
        colums.append(y**3*x)
        colums.append(x**3*y)
        colums.append(y**2*x**2)

    if degree>4:
        colums.append(x**5)
        colums.append(y**5)
        colums.append(y**4*x)
        colums.append(x**4*y)
        colums.append(y**3*x**2)
        colums.append(x**3*y**2)

    if degree>5:
        exit()

    X = np.array(colums)
    X = X.transpose() #desgin matrix where each column corrosponds to a feature

    return X


def calc_MSE(z,z_tilde):

    n = z.size
    return 1/n*np.sum((z-z_tilde)**2)

def calc_R2(z,z_tilde):

    mean = np.mean(z)
    return 1 - np.sum((z-z_tilde)**2)/ np.sum((z-mean)**2)


# code copied from project description and expanded to include a noise term
def FrankeFunction(x,y,noise_coeff=0.1):

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4 + np.random.normal(loc=0, scale=noise_coeff, size=len(x)) #incldues a stochastic noise according to the normal distribution N(0,1)

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


