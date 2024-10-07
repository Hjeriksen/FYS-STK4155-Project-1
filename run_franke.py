import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

from plot import *
from tools import *
from regressors import *

np.random.seed(2000)

# Make data 
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

# make grid
x, y = np.meshgrid(x,y) #generate cross values
x = x.flatten() #make cross values 1D
y = y.flatten() #make cross values 1D

# shuffle
x, y = unison_shuffled_copies(x,y)
# calculate targets
z = FrankeFunction(x,y)


# OLS

complexity_values = [i for i in range(1,6)]
MSE_values = []
R2_values = []
B_hat_values = []
for i in complexity_values:

    MSE,R2,B_hat = OLS(x,y,z,i)

    MSE_values.append(MSE)
    R2_values.append(R2)
    B_hat_values.append(B_hat)

plt.plot(complexity_values, MSE_values)
plt.ylabel("MSE")
plt.xlabel("Polynomial degree")
plt.savefig("OLS_MSE.png")
plt.clf()

plt.plot(complexity_values, R2_values)
plt.ylabel(r"$R^2$")
plt.xlabel("Polynomial degree")
plt.savefig("OLS_R2.png")
plt.clf()

complexity_values = [i for i in range(1,6)]
train_MSE_values = []
test_MSE_values = []
for i in complexity_values:

    train_MSE, test_MSE = OLS_train_and_test(x,y,z,i)

    train_MSE_values.append(train_MSE)
    test_MSE_values.append(test_MSE)

plt.plot(complexity_values, train_MSE_values, label='Train')
plt.plot(complexity_values, test_MSE_values, label='Test')
plt.ylabel("MSE")
plt.xlabel("Polynomial degree")
plt.legend(loc="best")
plt.savefig("OLS_test_train.png")
plt.clf()


# Ridge

complexity_values = [i for i in range(1,6)]
lambda_hp_values = [10,1,0.1,0.01,0.001] #testing 5 values in log scale

MSE_values = []
R2_values= []
for i in complexity_values:

    MSE_list = []
    R2_list = []

    for j in lambda_hp_values:

        MSE,R2,B_hat = Ridge(x,y,z,i,j)

        MSE_list.append(MSE)
        R2_list.append(R2)

    MSE_values.append(MSE_list)
    R2_values.append(R2_list)

plot_heatmap(np.array(MSE_values), lambda_hp_values, complexity_values, r'$\lambda$', r'$n_{pol}$', 'ridge_heat_mse.png')
plot_heatmap(np.array(R2_values), lambda_hp_values, complexity_values, r'$\lambda$', r'$n_{pol}$', 'ridge_heat_r2.png')

# LASSO

complexity_values = [i for i in range(1,6)]
lambda_hp_values = [10,1,0.1,0.01,0.001] #testing 5 values in log scale

MSE_values = []
R2_values= []
for i in complexity_values:

    MSE_list = []
    R2_list = []

    for j in lambda_hp_values:

        MSE,R2 = Lasso(x,y,z,i,j)

        MSE_list.append(MSE)
        R2_list.append(R2)

    MSE_values.append(MSE_list)
    R2_values.append(R2_list)

plot_heatmap(np.array(MSE_values), lambda_hp_values, complexity_values, r'$\lambda$', r'$n_{pol}$', 'lasso_heat_mse.png')
plot_heatmap(np.array(R2_values), lambda_hp_values, complexity_values, r'$\lambda$', r'$n_{pol}$', 'lasso_heat_r2.png')

# Bias Variance

complexity_values = [i for i in range(1,6)]
bootstrap_sample_values = [10, 100, 250]
datapoint_values = [5,10,15,20]
for i in bootstrap_sample_values:

    bias_mean_values = []
    var_mean_values = []

    for j in complexity_values:

        bias_mean_list = []
        var_mean_list = []

        for k in datapoint_values:

            bias_mean,var_mean = bootstrap_bv_tradeoff(i,j,k)

            bias_mean_list.append(bias_mean)
            var_mean_list.append(var_mean)

        bias_mean_values.append(bias_mean_list)
        var_mean_values.append(var_mean_list)

    plot_heatmap(np.array(bias_mean_values), [_**2 for _ in datapoint_values], complexity_values, r'Number of datapoints', r'Polynomial order', 'bv_decomp_bias_bootstrap' + str(i) + '.png')
    plot_heatmap(np.array(var_mean_values), [_**2 for _ in datapoint_values], complexity_values, r'Number of datapoints', r'Polynomial order', 'bv_decomp_variance_bootstrap' + str(i) + '.png')

# Cross validation

ols = linear_model.LinearRegression()
lasso = linear_model.Lasso(alpha=0.01)
ridge = linear_model.Ridge(alpha=0.01)

regs = [ols, lasso, ridge]

X = design_matrix(x, y, degree=3)

for j in regs:

    means = []
    variances = []
    for i in [5, 6, 7, 8, 9, 10]:


        cv_results = sklearn.model_selection.cross_validate(j, X, z, cv=i, scoring='neg_mean_squared_error')
        means.append(np.mean(-cv_results['test_score']))
        variances.append(np.var(-cv_results['test_score']))

    print(str(j))
    print(means)
    print(variances)
