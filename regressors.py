import numpy as np
import sklearn
from sklearn import linear_model

from tools import *


def OLS_train_and_test(x,y,z,degree):

    #set up design matrix (fitting without intercept)
    X = design_matrix(x,y,degree)

    #split into test and train
    split = int(x.size*2/3)
    X_train = X[:split,:]
    X_test = X[split:,:]
    z_train, z_test = np.array_split(z,[split])

    X_train_mean = np.mean(X_train, axis=0)
    z_train_mean = np.mean(z_train)

    #center data (labels and also features)
    z_train = z_train - z_train_mean
    X_train = X_train - X_train_mean
    X_test = X_test - X_train_mean #also need to transform data for prediction

    #fit. OLS: B = (X^T*X)^(-1)X^Ty
    inverse = np.linalg.pinv(X_train.T@X_train)
    B_hat = inverse@X_train.T@z_train

    #make prediction
    intercept = np.mean(z_train) #reconstruct intercept (left out of fitting) for prediction

    z_pred = B_hat@X_test.T + intercept
    test_MSE = calc_MSE(z_test,z_pred)
    test_R2 = calc_R2(z_test,z_pred)

    z_pred = B_hat@X_train.T + intercept
    train_MSE = calc_MSE(z_train,z_pred)
    train_R2 = calc_R2(z_train,z_pred)

    return train_MSE, test_MSE



def OLS(x,y,z,degree,bv_tradeoff=False):

    #set up design matrix (fitting without intercept)
    X = design_matrix(x,y,degree)

    #split into test and train
    split = int(x.size*2/3)
    X_train = X[:split,:]
    X_test = X[split:,:]
    z_train, z_test = np.array_split(z,[split])

    X_train_mean = np.mean(X_train, axis=0)
    z_train_mean = np.mean(z_train)

    #center data (labels and also features)
    z_train = z_train - z_train_mean
    X_train = X_train - X_train_mean
    X_test = X_test - X_train_mean #also need to transform data for prediction

    #fit. OLS: B = (X^T*X)^(-1)X^Ty
    inverse = np.linalg.pinv(X_train.T@X_train)
    B_hat = inverse@X_train.T@z_train

    #make prediction
    intercept = np.mean(z_train) #reconstruct intercept (left out of fitting) for prediction

    z_pred = B_hat@X_test.T + intercept

    #evaluation
    if not bv_tradeoff:
        MSE = calc_MSE(z_test,z_pred)
        R2 = calc_R2(z_test,z_pred)
        return MSE,R2,B_hat

    else:
        bias = np.mean((z_test - np.mean(z_pred))**2 )
        var = np.var(z_pred)
        return bias,var





#b) Ridge regression analysis using polynomials in and up to fifth order with centering of data (i.e. we are fitting wihtout an intercept)

def Ridge(x,y,z,degree,lambda_hp):

    #set up design matrix (fitting without intercept)
    X = design_matrix(x,y,degree)

    #split into test and train
    split = int(x.size*2/3)
    X_train = X[:split,:]
    X_test = X[split:,:]
    z_train, z_test = np.array_split(z,[split])

    X_train_mean = np.mean(X_train, axis=0)
    z_train_mean = np.mean(z_train)
    #center data (labels and also features)
    z_train = z_train - z_train_mean
    z_test = z_test - z_train_mean
    X_train = X_train - X_train_mean
    X_test = X_test - X_train_mean #also need to transform data for prediction

    #fit. Ridge: B = (X^T*X + lambda*I)^(-1)X^Ty
    inverse = np.linalg.pinv(X_train.transpose()@X_train + lambda_hp * np.identity(X_train.shape[1]))
    B_hat = inverse@X_train.transpose()@z_train

    #make prediction
    intercept = np.mean(z_train) #reconstruct intercept (left out of fitting) for prediction
    z_pred = B_hat@X_test.T + intercept

    #evaluation
    MSE = calc_MSE(z_test,z_pred)
    R2 = calc_R2(z_test,z_pred)

    return MSE,R2,B_hat


#c) Lasso regression analysis using polynomials in and up to fifth order with centering of data with scikitlearn

def Lasso(x,y,z,degree,lambda_hp):

    #set up design matrix (fitting without intercept)
    X = design_matrix(x,y,degree)

    #split into test and train
    split = int(x.size*2/3)
    X_train = X[:split,:]
    X_test = X[split:,:]
    z_train, z_test = np.array_split(z,[split])

    #set up model and fit
    model = linear_model.Lasso(alpha=lambda_hp)
    model.fit(X_train,z_train)

    #predict
    z_pred = model.predict(X_test)

    #evaluation
    MSE = calc_MSE(z_test,z_pred)
    R2 = calc_R2(z_test,z_pred)

    return MSE,R2




# e) bias and variance trade-off as function of your model complexity (the degree of the polynomial) and the number of data points, and possibly also your training and test data using the bootstrap resampling method

def bootstrap_bv_tradeoff(sample_amount,degree,datapoint_amount):

    # Make data (code copied from project description and expanded to include a noise term)
    x = np.linspace(0, 1, datapoint_amount)
    y = np.linspace(0, 1, datapoint_amount)

    # make grid
    x, y = np.meshgrid(x,y) #generate cross values
    x = x.flatten() #make cross values 1D
    y = y.flatten() #make cross values 1D

    # shuffle
    x, y = unison_shuffled_copies(x,y)
    # calculate targets
    z = FrankeFunction(x,y)

    bias_mean = 0
    var_mean = 0

    for i in range(sample_amount):

        sample_x = np.zeros(len(z))
        sample_y = np.zeros(len(z))
        sample_z = np.zeros(len(z))

        for i in range(len(sample_x)):

            idx = np.random.randint(0, len(z), 1)

            sample_x[i] = x[idx]
            sample_y[i] = y[idx]
            sample_z[i] = z[idx]

        bias,var = OLS(sample_x,sample_y,z,degree,bv_tradeoff=True)

        bias_mean += bias
        var_mean += var

    bias_mean = bias_mean/sample_amount
    var_mean = bias_mean/sample_amount

    return bias_mean,var_mean

