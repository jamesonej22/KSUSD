"""
KSUSD.py

Kernel Smoothing Methods For Unevenly-Spaced Data

Methods Implemented:
    Piecewise Constant 
    Piecewise Linear
    KNN Moving Average 
    Kernel Weighted Moving Average
    Locally Weighted Linear Regression
    Locally Weighted Quadratic Regression
    Variable Bandwidth KDE Regression 

Kernel Functions Implemented:
    Gaussian
    Tri-cube
    Epanechnikov

Utility Functions Implemented:
    Find K-Nearest Neighbors
    Silverman's Rule of Thumb
    Variable Bandwidth KDE 
    Integrate KDE
    
Error Metrics Implemented:
    Ground Truth RMSE
    Leave-One-Out RMSE

Parameter Selection Methods Implemented:
    Find Optimal Parameters
    
Author: Eric Jameson
"""

import numpy as np
from itertools import product
from scipy.interpolate import interp1d


#==============================================================================
# UTILITY FUNCTIONS 
#==============================================================================

# Utility function to compute the indices of the K "closest" points of array X in the 
# data-matrix array (based on x-coordinate)
def FindKNN(X, data_matrix, K):
    KNN_Indices = np.zeros((K, X.shape[1]), dtype=np.uint8)
    for j in range(KNN_Indices.shape[1]):
        KNN_Indices[:,j] = np.argsort(np.abs(X[0, j] - data_matrix[0, :]))[:K]

    return KNN_Indices

# Gaussian Kernel with parameter lambda
def GaussianKernel(x, x0, K_lambda):
    t = np.abs(x - x0) / K_lambda
    return np.exp(- t**2 / 2)


# Tricube Kernel with parameter lambda
def TricubeKernel(x, x0, K_lambda):
    t = np.abs(x - x0) / K_lambda
    if t > 1:
        return 0
    return (1 - t**3)**3

# Epanechnikov Kernel with parameter lambda
def EpanechnikovKernel(x, x0, K_lambda):
    t = np.abs(x - x0) / K_lambda
    if t > 1:
        return 0
    return 0.75*(1 - t**2)

# Silverman's Rule of Thumb for Variable Bandwidth KDE
def silverman_rule_of_thumb(X):
    data_x = X[0,:]
    IQR = np.percentile(data_x, [25, 75])
    A = min(np.std(data_x), (IQR[1]-IQR[0])/1.34) 
    
    silverman_bandwidth = 0.9 * A * X.shape[1]**(-0.2)
    return silverman_bandwidth


# Variable Bandwidth KDE Creation
def VBKDE(X, data_matrix, K_frac, Kernel):
    K_N = int(K_frac * data_matrix.shape[1])
    NN_Indices = FindKNN(data_matrix, data_matrix, K_N)
    
    lambdas = []
    
    for i in range(data_matrix.shape[1]):
        lambdas.append(silverman_rule_of_thumb(data_matrix[:, NN_Indices[:, i]]))
    
    lambdas = np.array(lambdas)
    KDE = np.zeros((2, X.shape[1]))
    KDE[0, :] = X[0, :]
    
    if Kernel == GaussianKernel:
        factor = 1 / np.sqrt(2 * np.pi)
    elif Kernel == TricubeKernel:
        factor = 70 / 81
    else:
        factor = 1
        
    for i in range(X.shape[1]):
        kernel_sum = 0
        for j in range(data_matrix.shape[1]):
            kernel_sum += 1 / lambdas[j] * factor * Kernel(X[0, i], data_matrix[0, j], lambdas[j])
        
        kernel_sum /= data_matrix.shape[1]
        KDE[1, i] = kernel_sum
    
    return KDE 

# Integrate the given KDE to each of the given points to give the estimated 
# value of the CDF at that point.
def IntegrateKDE(KDE, Points):
    f = interp1d(KDE[0,:], KDE[1,:])
    Values = np.zeros((2, Points.shape[1]))
    Values[0, :] = Points[0, :]

    for i in range(Points.shape[1]):
        insert_index_upper = np.searchsorted(KDE[0,:], Points[0,i])
        KDE_approx = f(Points[0,i]) 
    
        combined_points = np.concatenate((KDE[0, :insert_index_upper], [Points[0,i]]))
        combined_data = np.concatenate((KDE[1, :insert_index_upper],[KDE_approx]))
        Values[1, i] = np.trapz(combined_data, x=combined_points)

    return Values


#==============================================================================
# ESTIMATION METHODS
#==============================================================================
    
# Piecewise constant
def PiecewiseConstant(X, data_matrix):
    NN_Indices = FindKNN(X, data_matrix, 1)
    Values = np.zeros((2, X.shape[1]))
    Values[0,:] = X[0,:]

    for j in range(Values.shape[1]):
        Values[1, j] = data_matrix[1, NN_Indices[0,j]]
    
    return Values


# Piecewise linear
def PiecewiseLinear(X, data_matrix):
    Values = np.zeros((2, X.shape[1]))
    Values[0, :] = X[0, :]

    for i in range(data_matrix.shape[1] - 1):
        x1, y1 = data_matrix[0, i], data_matrix[1, i]
        x2, y2 = data_matrix[0, i+1] , data_matrix[1, i+1]

        slope = (y2 - y1) / (x2 - x1)
        indices = np.logical_and(X[0, :] >= data_matrix[0, i], X[0, :] <= data_matrix[0, i+1])
        Values[1, indices] = slope * (X[0, indices] - x1) + y1

    return Values


# KNN Moving Average
def MovingAverage(X, data_matrix, K_frac):
    K = int(K_frac * data_matrix.shape[1])

    NN_Indices = FindKNN(X, data_matrix, K)
    
    Values = np.zeros((2, X.shape[1]))
    Values[0, :] = X[0, :]

    for j in range(Values.shape[1]):
        Values[1, j] = np.mean(data_matrix[1, NN_Indices[:,j]])
    
    return Values


# Kernel-Weighted Moving Average
def KernelSmoothing(X, data_matrix, Kernel, K_lambda):
    Values = np.zeros((2, X.shape[1]))
    Values[0, :] = X[0, :]

    for j in range(Values.shape[1]):
        K_sum_y = 0
        K_sum = 0
        for i in range(data_matrix.shape[1]):
            K = Kernel(Values[0, j], data_matrix[0, i], K_lambda)
            
            K_sum_y += K*data_matrix[1, i]
            K_sum += K

        Values[1, j] = K_sum_y / K_sum
    
    return Values


# Locally Weighted Linear Regression
def LocalLinear(X, data_matrix, Kernel, K_lambda):
    N = data_matrix.shape[1]
    Values = np.zeros((2, X.shape[1]))
    Values[0, :] = X[0, :]
       
    for j in range(Values.shape[1]):
        B = np.vstack((np.ones((1,N)), np.zeros((1,N)))).T
        W = np.zeros((N, N))
        y = np.zeros((1, N)).T
        for i in range(data_matrix.shape[1]):
            W[i,i] = Kernel(Values[0, j], data_matrix[0, i], K_lambda)
            B[i,1] = data_matrix[0, i]
            y[i,0] = data_matrix[1, i]

        y0 = np.matmul(np.array([1, Values[0,j]]).T, np.linalg.inv(np.matmul(np.matmul(B.T, W), B)))
        y0 = np.matmul(np.matmul(y0, np.matmul(B.T, W)), y)
        Values[1, j] = y0    
    return Values


# Locally Weighted Quadratic Regression
def LocalQuadratic(X, data_matrix, Kernel, K_lambda):
    N = data_matrix.shape[1]
    Values = np.zeros((2, X.shape[1]))
    Values[0, :] = X[0, :]
       
    for j in range(Values.shape[1]):
        B = np.vstack((np.ones((1,N)), np.zeros((1,N)), np.zeros((1,N)))).T
        W = np.zeros((N, N))
        y = np.zeros((1, N)).T
        for i in range(data_matrix.shape[1]):
            W[i,i] = Kernel(Values[0, j], data_matrix[0, i], K_lambda)
            B[i,1] = data_matrix[0, i]
            B[i,2] = data_matrix[0, i]**2
            y[i,0] = data_matrix[1, i]

        y0 = np.matmul(np.array([1, Values[0,j], Values[0,j]**2]).T, np.linalg.inv(np.matmul(np.matmul(B.T, W), B)))
        y0 = np.matmul(np.matmul(y0, np.matmul(B.T, W)), y)
        Values[1, j] = y0    
    return Values


# Variable Bandwidth KDE Regression
def VBKDERegression(X, KDE_points, data_matrix, K_frac, KDE_Kernel, Method, params):
    KDE = VBKDE(KDE_points, data_matrix, K_frac, KDE_Kernel)
    cdf_desired_points = IntegrateKDE(KDE, X)

    Values = Method(np.reshape(cdf_desired_points[1, :], (-1,1)).T, data_matrix, **params)
    Values[0, :] = X[0, :]
    
    return Values
    
#==============================================================================
# EVALUATION METHODS
#==============================================================================
    
# Ground Truth RMSE Calculation
def EvaluateGroundTruth(ground_truth, method, **kwargs):
    EstimatedValues = method(**kwargs)
    indices = []
    if EstimatedValues.shape[1] < ground_truth.shape[1]:
        for i in range(EstimatedValues.shape[1]):
          indices.append(np.where(ground_truth[0,:] == EstimatedValues[0, i])[0][0])
        RMSE = np.sqrt(np.mean((EstimatedValues[1, :] - ground_truth[1, indices])**2))
    else:
        for i in range(ground_truth.shape[1]):
            indices.append(np.where(EstimatedValues[0, :] == ground_truth[0, i])[0][0])
        RMSE = np.sqrt(np.mean((EstimatedValues[1, indices] - ground_truth[1,:])**2))
    return RMSE


# Leave-one-out RMSE Calculation
def EvaluateLOO(X, LOOdata_matrix, method, **kwargs):
    SSE = 0
    if X.shape[1] >= LOOdata_matrix.shape[1]:
        for i in range(LOOdata_matrix.shape[1]):
            tempdata = np.delete(LOOdata_matrix, i, 1)
            EstimatedValues = method(X, data_matrix=tempdata, **kwargs)
            idx = np.where(EstimatedValues[0,:] == LOOdata_matrix[0,i])[0][0]
            SSE += (EstimatedValues[1, idx] - LOOdata_matrix[1,i])**2
        RMSE = np.sqrt(SSE / LOOdata_matrix.shape[1])
    else:
        for i in range(X.shape[1]):
            idx1 = np.where(LOOdata_matrix[0,:] == X[0,i])[0][0]
            tempdata = np.delete(LOOdata_matrix, idx1, 1)
            EstimatedValues = method(X, data_matrix=tempdata, **kwargs)
            idx2 = np.where(EstimatedValues[0,:] == X[0,i])[0][0]
            SSE += (EstimatedValues[1,idx2] - LOOdata_matrix[1, idx1])**2
        RMSE = np.sqrt(SSE / X.shape[1])
    return RMSE


#==============================================================================
# PARAMETER SELECTION
#==============================================================================
    
# Find the optimal parameters from the provided list for the proposed RMSE metric 
def FindOptimalParams(X, data_matrix, method, evaluation_method, ground_truth=None, **params):
    param_keys = list(params.keys())
    list_keys = [param_key for param_key in param_keys if type(params[param_key]) == type([0])]
    param_lists = []
    for i in range(len(list_keys)):
        param_lists.append(params[list_keys[i]])
    
    min_error = 100
    min_error_param1 = 0
    min_error_param2 = 0

    if evaluation_method == EvaluateGroundTruth:
        params['X'] = X
        params['data_matrix'] = data_matrix
    
    if len(list_keys) == 1:
        for p1 in param_lists[0]:
            params[list_keys[0]] = p1
            if evaluation_method == EvaluateGroundTruth:
                temp_error = EvaluateGroundTruth(ground_truth, method, **params)
            else:
                temp_error = EvaluateLOO(X, data_matrix, method, **params)

            if temp_error < min_error:
                min_error = temp_error
                min_error_param1 = p1    

        return min_error, min_error_param1
    
    elif len(list_keys) == 2:
        for p1, p2 in product(param_lists[0], param_lists[1]):
            params[list_keys[0]] = p1
            params[list_keys[1]] = p2
            if evaluation_method == EvaluateGroundTruth:
                temp_error = EvaluateGroundTruth(ground_truth, method, **params)
            else:
                temp_error = EvaluateLOO(X, data_matrix, method, **params)

            if temp_error < min_error:
                min_error = temp_error
                min_error_param1 = p1 
                min_error_param2 = p2
        
        return min_error, min_error_param1, min_error_param2    
    