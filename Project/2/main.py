import random
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn import preprocessing, cluster, model_selection, metrics


synInputData = pd.read_csv('input.csv',header=None).values
synOutputData = pd.read_csv('output.csv',header=None).values
letorInputData = pd.read_csv('Querylevelnorm_X.csv',header=None).values
letorOutputData = pd.read_csv('Querylevelnorm_t.csv',header=None).values

# X_train, X_test, y_train, y_test = model_selection.train_test_split(synInputData, synOutputData, test_size=0.20, shuffle=False)
# X_validate, X_test, y_validate, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.50, shuffle=False)


X_train, X_test, y_train, y_test = model_selection.train_test_split(letorInputData, letorOutputData, test_size=0.20, shuffle=False)
X_validate, X_test, y_validate, y_test = model_selection.train_test_split(X_test, y_test, test_size=0.50, shuffle=False)

def compute_design_matrix(X_train, k_clusters):
    N,D = X_train.shape
    kmeans = cluster.KMeans(k_clusters).fit(X_train)
    centers = kmeans.cluster_centers_
    centers = centers[:,np.newaxis,:]
    spreads = []
    covM = np.cov(X_train.T)
    for _ in range(0,k_clusters): spreads.append(covM)
    spreads = np.array(spreads)
    X = X_train[np.newaxis,:,:]
    basis_func_outputs = np.exp(np. sum(np.matmul(X - centers, spreads) * (X - centers), axis=2) / (-2) ).T
    return np.insert(basis_func_outputs, 0, 1, axis=1)

def closed_form_solution(l2_lamda, designMatrix, outValues):
    weights = np.dot(np.dot(np.linalg.inv(np.dot(designMatrix.T,designMatrix)),designMatrix.T),outValues)
    return weights

def SGD(learningRate, l2_lamda, designMatrix, outValues, miniBatchSize, numEpochs,k_clusters):
    N=designMatrix.shape[0]
    weights = np.zeros([1, k_clusters+1])
    for epoch in range(numEpochs):
        for i in range(int(N / miniBatchSize)):
            lower_bound = i * miniBatchSize
            upper_bound = min((i+1)*miniBatchSize, N)
            Phi = designMatrix[lower_bound : upper_bound, :]
            t = outValues[lower_bound : upper_bound, :]
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T,Phi )
            E = (E_D + l2_lamda * weights) / miniBatchSize
            weights = weights - learningRate * E
    return weights.flatten()

def rms_error(weights, designMatrix, Y, l2_lamda):
    y_calc = np.dot(designMatrix,weights)
#     print(y_calc.shape)
    E_D_W = 0.5*metrics.mean_squared_error(Y,y_calc)*designMatrix.shape[0]
    E_W_W = 0.5*np.dot(weights.T,weights)
    E_W = E_D_W + l2_lamda*E_W_W
    train_error = np.sqrt(2*E_W/designMatrix.shape[0])
#     print(train_error.shape)
    return train_error

k_cluster = 16

l2_lamda = 0.9

print("From closed form solution.")
designMatrix = compute_design_matrix(X_train,k_cluster)
weights_train0 = closed_form_solution(l2_lamda, designMatrix, y_train)
train_error0 = rms_error(weights_train0,designMatrix,y_train,l2_lamda)
print("Training error:",train_error0)

designMatrix2 = compute_design_matrix(X_validate,k_cluster)
train_error2 = rms_error(weights_train0,designMatrix2,y_validate,l2_lamda)
print("Test error (Validation set):",train_error2)


#Test error
designMatrix3 = compute_design_matrix(X_test,k_cluster)
train_error3 = rms_error(weights_train0,designMatrix3,y_test,l2_lamda)
print("Test error (test set):",train_error3)


l2_lamda = 0.9
learning_rate = 0.001
print("From SGD.")
designMatrix = compute_design_matrix(X_train,k_cluster)
weights_train0 = SGD(learning_rate, l2_lamda, designMatrix,y_train, designMatrix.shape[0],5000,k_cluster)
train_error0 = rms_error(weights_train0,designMatrix,y_train,l2_lamda)
print("Training error:",train_error0)

designMatrix2 = compute_design_matrix(X_validate,k_cluster)
train_error2 = rms_error(weights_train0,designMatrix2,y_validate, l2_lamda)
print("Test error (Validation set):",train_error2)

#Test error
designMatrix3 = compute_design_matrix(X_test,k_cluster)
train_error3 = rms_error(weights_train0,designMatrix3,y_test, l2_lamda)
print("Test error (test set):",train_error3)
