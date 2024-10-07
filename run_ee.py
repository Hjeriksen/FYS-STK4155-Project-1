import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from imageio import imread

from plot import *
from tools import *
from regressors import *

# load data
terrain_1 = imread('data/SRTM_data_Norway_1.tif')
terrain_2 = imread('data/SRTM_data_Norway_2.tif')


# set up
ols = linear_model.LinearRegression()
lasso_1 = linear_model.Lasso(alpha=10)
lasso_2 = linear_model.Lasso(alpha=1)
lasso_3 = linear_model.Lasso(alpha=0.1)
lasso_4 = linear_model.Lasso(alpha=0.01)
lasso_5 = linear_model.Lasso(alpha=0.001)
ridge_1 = linear_model.Ridge(alpha=10)
ridge_2 = linear_model.Ridge(alpha=1)
ridge_3 = linear_model.Ridge(alpha=0.1)
ridge_4 = linear_model.Ridge(alpha=0.01)
ridge_5 = linear_model.Ridge(alpha=0.001)

terrains = [terrain_1, terrain_2]
regs = [ols, ridge_1, ridge_2, ridge_3, ridge_4, ridge_5, lasso_1, lasso_2, lasso_3, lasso_4, lasso_5]


means = []
variances = []

for terrain in terrains:

    # make data
    N = 1000
    terrain = terrain[:N,:N]

    # Creates mesh of image pixels
    x = np.linspace(0, 1, np.shape(terrain)[0])
    y = np.linspace(0, 1, np.shape(terrain)[1])
    x, y = np.meshgrid(x,y)

    x = x.flatten()
    y = y.flatten()

    X = design_matrix(x, y, degree=5)
    z = terrain.flatten()

    for reg in regs:

        cv_results = sklearn.model_selection.cross_validate(reg, X, z, cv=10, scoring='neg_mean_squared_error')
        means.append(np.mean(-cv_results['test_score']))
        variances.append(np.var(-cv_results['test_score']))

print(means)
print(variances)
