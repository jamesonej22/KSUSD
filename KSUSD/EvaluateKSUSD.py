"""
EvaluateKSUSD.py

Example of evaluation methods implemented in KSUSD.py.

For a fixed numpy random seed, draw 20 random observations from a Uniform[0,1]
random variable, and create a dataset X = sin(4t) + epsilon, where epsilon is 
drawn from a Normal(0, 1/3) random variable.

For each of the methods in KSUSD.py, fit the model to the random data for these
three cases:
    1. No interpolation between observations
    2. Midpoint interpolation between observations
    3. 5-point interpolation between observations
    
In each of the above cases, calculate the ground truth RMSE in comparison to
X = sin(4t). Additionally, calculate the LOO RMSE for each model. Note that the
number of evaluation points does not affect the value of the LOO RMSE and so we
only compute it once. 

Prints the RMSE values for each method as they are computed.

Author: Eric Jameson
"""

from KSUSD import *
import numpy as np

if __name__ == "__main__":
    np.random.seed(12345)
    t = np.linspace(0, 1, 100)
    y = np.sin(4 * t)
    t_sparse = np.sort(np.random.uniform(0, 1, 20))
    t_true = np.unique(np.concatenate((t, t_sparse)))
    y_true = np.sin(4 * t_true)
    y_sparse = np.sin(4 * t_sparse)
    y_noise = y_sparse + np.random.normal(0, 1/3, t_sparse.shape)
    TrueData = np.vstack((t_sparse, y_sparse))
    FullData = np.vstack((t_sparse, y_noise))

    test1 = np.reshape(t_sparse, (-1,1)).T
    True1 = np.vstack((t_sparse, y_sparse))
    
    test2 = [np.linspace(t_sparse[i], t_sparse[i+1], 3).tolist()[j] for j in range(3) for i in range(t_sparse.shape[0] - 1)]
    test2 = np.array(sorted(list(set(test2))))
    y2 = np.sin(4*test2)
    True2 = np.vstack((test2, y2))
    test2 = np.reshape(test2, (-1,1)).T 

    test3 = [np.linspace(t_sparse[i], t_sparse[i+1], 7).tolist()[j] for j in range(7) for i in range(t_sparse.shape[0] - 1)]
    test3 = np.array(sorted(list(set(test3))))
    y3 = np.sin(4*test3)
    True3 = np.vstack((test3, y3))
    test3 = np.reshape(test3, (-1,1)).T   

    test = np.linspace(-0.3,1.3,1000)
    test = np.reshape(test, (-1,1)).T

    
    # Piecewise Constant
    gt1 = EvaluateGroundTruth(True1, PiecewiseConstant, **{'X': test1, 'data_matrix': FullData})
    gt2 = EvaluateGroundTruth(True2, PiecewiseConstant, **{'X': test2, 'data_matrix': FullData})
    gt3 = EvaluateGroundTruth(True3, PiecewiseConstant, **{'X': test3, 'data_matrix': FullData})

    loo = EvaluateLOO(test1, FullData, PiecewiseConstant, **{})
    
    print(f"Piecewise Constant: {gt1:.4f}, {gt2:.4f}, {gt3:.4f}")
    print(f"Piecewise Constant LOO: {loo:.4f}")

    # Piecewise Linear
    gt1 = EvaluateGroundTruth(True1, PiecewiseLinear, **{'X': test1, 'data_matrix': FullData})
    gt2 = EvaluateGroundTruth(True2, PiecewiseLinear, **{'X': test2, 'data_matrix': FullData})
    gt3 = EvaluateGroundTruth(True3, PiecewiseLinear, **{'X': test3, 'data_matrix': FullData})

    loo = EvaluateLOO(test1, FullData, PiecewiseLinear, **{})
    print(f"Piecewise Constant: {gt1:.4f}, {gt2:.4f}, {gt3:.4f}")
    print(f"Piecewise Constant LOO: {loo:.4f}")

    # Moving Average
    gt1 = EvaluateGroundTruth(True1, MovingAverage, **{'X': test1, 'data_matrix': FullData, 'K_frac': 0.3})
    gt2 = EvaluateGroundTruth(True2, MovingAverage, **{'X': test2, 'data_matrix': FullData, 'K_frac': 0.3})
    gt3 = EvaluateGroundTruth(True3, MovingAverage, **{'X': test3, 'data_matrix': FullData, 'K_frac': 0.3})

    loo = EvaluateLOO(test1, FullData, MovingAverage, **{'K_frac': 0.3})
    
    print(f"Moving Average: {gt1:.4f}, {gt2:.4f}, {gt3:.4f}")
    print(f"Moving Average LOO: {loo:.4f}")

    # Gaussian Kernel
    t1 = EvaluateGroundTruth(True1, KernelSmoothing, **{'X': test1, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, KernelSmoothing, **{'X': test2, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, KernelSmoothing, **{'X': test3, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    
    loo1 = EvaluateLOO(test1, FullData, KernelSmoothing, **{'Kernel': GaussianKernel, 'K_lambda': 0.2})
            
    print(f"Gaussian Kernel: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Gaussian Kernel LOO: {loo:.4f}")

    # Tricube Kernel
    t1 = EvaluateGroundTruth(True1, KernelSmoothing, **{'X': test1, 'data_matrix': FullData, 'Kernel': TricubeKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, KernelSmoothing, **{'X': test2, 'data_matrix': FullData, 'Kernel': TricubeKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, KernelSmoothing, **{'X': test3, 'data_matrix': FullData, 'Kernel': TricubeKernel, 'K_lambda': 0.2})
    
    loo = EvaluateLOO(test1, FullData, KernelSmoothing, **{'Kernel': TricubeKernel, 'K_lambda': 0.2})
    
    print(f"Tricube Kernel: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Tricube Kernel LOO: {loo:.4f}")

    # Epanechnikov Kernel
    t1 = EvaluateGroundTruth(True1, KernelSmoothing, **{'X': test1, 'data_matrix': FullData, 'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, KernelSmoothing, **{'X': test2, 'data_matrix': FullData, 'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, KernelSmoothing, **{'X': test3, 'data_matrix': FullData, 'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
    
    loo = EvaluateLOO(test1, FullData, KernelSmoothing, **{'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
            
    print(f"Epanechnikov Kernel: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Epanechnikov Kernel LOO: {loo:.4f}")

    # Gaussian Linear
    t1 = EvaluateGroundTruth(True1, LocalLinear, **{'X': test1, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, LocalLinear, **{'X': test2, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, LocalLinear, **{'X': test3, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    
    loo1 = EvaluateLOO(test1, FullData, LocalLinear, **{'Kernel': GaussianKernel, 'K_lambda': 0.2})
    
    print(f"Gaussian Linear: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Gaussian Linear LOO: {loo:.4f}")

    # Tricube Linear
    t1 = EvaluateGroundTruth(True1, LocalLinear, **{'X': test1, 'data_matrix': FullData, 'Kernel': TricubeKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, LocalLinear, **{'X': test2, 'data_matrix': FullData, 'Kernel': TricubeKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, LocalLinear, **{'X': test3, 'data_matrix': FullData, 'Kernel': TricubeKernel, 'K_lambda': 0.2})
    
    loo = EvaluateLOO(test1, FullData, LocalLinear, **{'Kernel': TricubeKernel, 'K_lambda': 0.2})
  
    print(f"Tricube Linear: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Tricube Linear LOO: {loo:.4f}")

    # Epanechnikov Linear
    t1 = EvaluateGroundTruth(True1, LocalLinear, **{'X': test1, 'data_matrix': FullData, 'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, LocalLinear, **{'X': test2, 'data_matrix': FullData, 'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, LocalLinear, **{'X': test3, 'data_matrix': FullData, 'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})
    
    loo = EvaluateLOO(test1, FullData, LocalLinear, **{'Kernel': EpanechnikovKernel, 'K_lambda': 0.2})

    print(f"Epanechnikov Linear: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Epanechnikov Linear LOO: {loo:.4f}")

    # Gaussian Quadratic
    t1 = EvaluateGroundTruth(True1, LocalQuadratic, **{'X': test1, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    t2 = EvaluateGroundTruth(True2, LocalQuadratic, **{'X': test2, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    t3 = EvaluateGroundTruth(True3, LocalQuadratic, **{'X': test3, 'data_matrix': FullData, 'Kernel': GaussianKernel, 'K_lambda': 0.2})
    
    loo = EvaluateLOO(test1, FullData, LocalQuadratic, **{'Kernel': GaussianKernel, 'K_lambda': 0.2})
  
    print(f"Gaussian Quadratic: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"Gaussian Quadratic LOO: {loo:.4f}")

    # VBKDE
    t1 = EvaluateGroundTruth(True1, VBKDERegression, **{'X': test1, 'KDE_points': test, 'data_matrix': FullData, 'K_frac': 0.3, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': 0.1}})
    t2 = EvaluateGroundTruth(True2, VBKDERegression, **{'X': test2, 'KDE_points': test, 'data_matrix': FullData, 'K_frac': 0.3, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': 0.1}})
    t3 = EvaluateGroundTruth(True3, VBKDERegression, **{'X': test3, 'KDE_points': test, 'data_matrix': FullData, 'K_frac': 0.3, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': 0.1}})

    loo = EvaluateLOO(test1, FullData, VBKDERegression, **{'K_frac': 0.3, 'KDE_points': test, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': 0.1}})

    print(f"VBKDE: {t1:.4f}, {t2:.4f}, {t3:.4f}")
    print(f"VBKDE LOO: {loo:.4f}")