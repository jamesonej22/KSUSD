"""
SimulationKSUSD.py

Simulation study using the trend estimation methods in KSUSD.py.

For a fixed number of trials, draw 20 random observations from a Uniform[0,1]
random variable, and create a dataset X = sin(4t) + epsilon, where epsilon is 
drawn from a Normal(0, 1/3) random variable.

For each of the methods in KSUSD.py, fit the model to the random data for these
three cases:
    1. No interpolation between observations
    2. Midpoint interpolation between observations
    3. 5-point interpolation between observations
    
In each of the above cases, find the parameters for each model which minimize
the ground truth RMSE in comparison to X = sin(4t). Additionally, find the 
parameters which minimize the LOO RMSE for each model. Note that these 
parameters need not be the same. 

Store the minimized RMSE values for each model in each trial in the text 
documents "Simulation1.txt", "Simulation2.txt", "Simulation3.txt", and 
"Simulation4.txt" for later use.

Additionally, plot boxplots showing the distribution of the RMSE Values for
each model.

In experimentation, both the tri-cube and Epanechnikov Kernels were prone to
producing singular matrices during the matrix inversion process for locally
weighted linear regression. These methods are wrapped in a try-except block for
now, and if a matrix is found to be singular, the model is skipped for that 
trial. The total number of failures for both methods are printed at the end of 
the analysis. 

Author: Eric Jameson
"""

from KSUSD import *
import time
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    NUM_TRIALS = 2

    PWConstant1, PWConstant2, PWConstant3, PWConstantLOO = [], [], [], []
    PWLinear1, PWLinear2, PWLinear3, PWLinearLOO = [], [], [], []
    MAverage1, MAverage2, MAverage3, MAverageLOO = [], [], [], []
    GaussKReg1, GaussKReg2, GaussKReg3, GaussKRegLOO = [], [], [], []
    TricubeKReg1, TricubeKReg2, TricubeKReg3, TricubeKRegLOO = [], [], [], []
    counttrikernel = 0
    EpanechnikovKReg1, EpanechnikovKReg2, EpanechnikovKReg3, EpanechnikovKRegLOO = [], [], [], []
    countepkernel = 0
    GaussLinear1, GaussLinear2, GaussLinear3, GaussLinearLOO = [], [], [], []
    TricubeLinear1, TricubeLinear2, TricubeLinear3, TricubeLinearLOO = [], [], [], []
    counttrilinear = 0
    EpanechnikovLinear1, EpanechnikovLinear2, EpanechnikovLinear3, EpanechnikovLinearLOO = [], [], [], []
    counteplinear = 0
    GaussQuad1, GaussQuad2, GaussQuad3, GaussQuadLOO = [], [], [], []
    VBKDEReg1, VBKDEReg2, VBKDEReg3, VBKDERegLOO = [], [], [], []

    for trial in range(NUM_TRIALS):
        time0 = time.time()
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

        K_fracs = np.linspace(0.1, 0.95, 20).tolist()
        K_lambdas = np.linspace(0.2, 2, 20).tolist()
        K_lambda_gauss = np.linspace(0.05, 2, 20).tolist()

        # Piecewise Constant
        gt1 = EvaluateGroundTruth(True1, PiecewiseConstant, **{'X': test1, 'data_matrix': FullData})
        gt2 = EvaluateGroundTruth(True2, PiecewiseConstant, **{'X': test2, 'data_matrix': FullData})
        gt3 = EvaluateGroundTruth(True3, PiecewiseConstant, **{'X': test3, 'data_matrix': FullData})

        loo = EvaluateLOO(test1, FullData, PiecewiseConstant, **{})
        
        PWConstant1.append(gt1)
        PWConstant2.append(gt2)
        PWConstant3.append(gt3)
        PWConstantLOO.append(loo)

        # Piecewise Linear
        gt1 = EvaluateGroundTruth(True1, PiecewiseLinear, **{'X': test1, 'data_matrix': FullData})
        gt2 = EvaluateGroundTruth(True2, PiecewiseLinear, **{'X': test2, 'data_matrix': FullData})
        gt3 = EvaluateGroundTruth(True3, PiecewiseLinear, **{'X': test3, 'data_matrix': FullData})

        loo = EvaluateLOO(test1, FullData, PiecewiseLinear, **{})
        
        PWLinear1.append(gt1)
        PWLinear2.append(gt2)
        PWLinear3.append(gt3)
        PWLinearLOO.append(loo)

        # Moving Average 
        err1, _ = FindOptimalParams(test1, FullData, MovingAverage, EvaluateGroundTruth, True1, **{'K_frac': K_fracs})
        err2, _ = FindOptimalParams(test2, FullData, MovingAverage, EvaluateGroundTruth, True2, **{'K_frac': K_fracs})
        err3, _ = FindOptimalParams(test3, FullData, MovingAverage, EvaluateGroundTruth, True3, **{'K_frac': K_fracs})

        err4, _ = FindOptimalParams(test1, FullData, MovingAverage, EvaluateLOO, **{'K_frac': K_fracs})

        MAverage1.append(err1)
        MAverage2.append(err2)
        MAverage3.append(err3)            

        MAverageLOO.append(err4)

        # Kernel Smoothing
        # Gaussian Kernel 
        err1, _ = FindOptimalParams(test1, FullData, KernelSmoothing, EvaluateGroundTruth, True1, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        err2, _ = FindOptimalParams(test2, FullData, KernelSmoothing, EvaluateGroundTruth, True2, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        err3, _ = FindOptimalParams(test3, FullData, KernelSmoothing, EvaluateGroundTruth, True3, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})

        err4, _ = FindOptimalParams(test1, FullData, KernelSmoothing, EvaluateLOO, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
    
        GaussKReg1.append(err1)
        GaussKReg2.append(err2)
        GaussKReg3.append(err3)

        GaussKRegLOO.append(err4)

        # Tricube Kernel
        try:
            err1, _ = FindOptimalParams(test1, FullData, KernelSmoothing, EvaluateGroundTruth, True1, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(test2, FullData, KernelSmoothing, EvaluateGroundTruth, True2, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
            err3, _ = FindOptimalParams(test3, FullData, KernelSmoothing, EvaluateGroundTruth, True3, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})

            err4, _ = FindOptimalParams(test1, FullData, KernelSmoothing, EvaluateLOO, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})    
        except:
            err1 = np.nan
            err2 = np.nan
            err3 = np.nan
            err4 = np.nan
            counttrikernel += 1
            
        TricubeKReg1.append(err1)
        TricubeKReg2.append(err2)
        TricubeKReg3.append(err3)

        TricubeKRegLOO.append(err4)
        
        # Epanechnikov Kernel
        try:
            err1, _ = FindOptimalParams(test1, FullData, KernelSmoothing, EvaluateGroundTruth, True1, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(test2, FullData, KernelSmoothing, EvaluateGroundTruth, True2, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
            err3, _ = FindOptimalParams(test3, FullData, KernelSmoothing, EvaluateGroundTruth, True3, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})

            err4, _ = FindOptimalParams(test1, FullData, KernelSmoothing, EvaluateLOO, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
        except:
            err1 = np.nan
            err2 = np.nan
            err3 = np.nan
            err4 = np.nan
            countepkernel += 1
        
        EpanechnikovKReg1.append(err1)
        EpanechnikovKReg2.append(err2)
        EpanechnikovKReg3.append(err3)

        EpanechnikovKRegLOO.append(err4)

        # # Local Linear Smoothing
        # Gaussian Linear 
        err1, glam1 = FindOptimalParams(test1, FullData, LocalLinear, EvaluateGroundTruth, True1, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        err2, glam2 = FindOptimalParams(test2, FullData, LocalLinear, EvaluateGroundTruth, True2, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        err3, glam3 = FindOptimalParams(test3, FullData, LocalLinear, EvaluateGroundTruth, True3, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})

        err4, glam4 = FindOptimalParams(test1, FullData, LocalLinear, EvaluateLOO, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
    
        GaussLinear1.append(err1)
        GaussLinear2.append(err2)
        GaussLinear3.append(err3)

        GaussLinearLOO.append(err4)

        # Tricube Linear 
        try:
            err1, _ = FindOptimalParams(test1, FullData, LocalLinear, EvaluateGroundTruth, True1, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(test2, FullData, LocalLinear, EvaluateGroundTruth, True2, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
            err3, _ = FindOptimalParams(test3, FullData, LocalLinear, EvaluateGroundTruth, True3, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})

            err4, _ = FindOptimalParams(test1, FullData, LocalLinear, EvaluateLOO, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
        except:
            err1 = np.nan 
            err2 = np.nan
            err3 = np.nan
            err4 = np.nan
            counttrilinear += 1

        TricubeLinear1.append(err1)
        TricubeLinear2.append(err2)
        TricubeLinear3.append(err3)

        TricubeLinearLOO.append(err4)

        # Epanechnikov Linear 
        try:
            err1, _ = FindOptimalParams(test1, FullData, LocalLinear, EvaluateGroundTruth, True1, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(test2, FullData, LocalLinear, EvaluateGroundTruth, True2, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
            err3, _ = FindOptimalParams(test3, FullData, LocalLinear, EvaluateGroundTruth, True3, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})

            err4, _ = FindOptimalParams(test1, FullData, LocalLinear, EvaluateLOO, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
        except:
            err1 = np.nan
            err2 = np.nan 
            err3 = np.nan
            err4 = np.nan
            counteplinear += 1

        EpanechnikovLinear1.append(err1)
        EpanechnikovLinear2.append(err2)
        EpanechnikovLinear3.append(err3)

        EpanechnikovLinearLOO.append(err4)
        
        # Gaussian Quadratic 
        err1, _ = FindOptimalParams(test1, FullData, LocalQuadratic, EvaluateGroundTruth, True1, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        err2, _ = FindOptimalParams(test2, FullData, LocalQuadratic, EvaluateGroundTruth, True2, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        err3, _ = FindOptimalParams(test3, FullData, LocalQuadratic, EvaluateGroundTruth, True3, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})

        err4, _ = FindOptimalParams(test1, FullData, LocalQuadratic, EvaluateLOO, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
    
        GaussQuad1.append(err1)
        GaussQuad2.append(err2)
        GaussQuad3.append(err3)

        GaussQuadLOO.append(err4)

        # VBKDE Regression 
        err1, _ = FindOptimalParams(test1, FullData, VBKDERegression, EvaluateGroundTruth, True1, **{'KDE_points': test, 'K_frac': K_fracs, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': glam1}})
        err2, _ = FindOptimalParams(test2, FullData, VBKDERegression, EvaluateGroundTruth, True2, **{'KDE_points': test, 'K_frac': K_fracs, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': glam2}})
        err3, _ = FindOptimalParams(test3, FullData, VBKDERegression, EvaluateGroundTruth, True3, **{'KDE_points': test, 'K_frac': K_fracs, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': glam3}})
        
        err4, _ = FindOptimalParams(test1, FullData, VBKDERegression, EvaluateLOO, **{'KDE_points': test, 'K_frac': K_fracs, 'KDE_Kernel': GaussianKernel, 'Method': LocalLinear, 'params': {'Kernel': GaussianKernel, 'K_lambda': glam4}})
        
        VBKDEReg1.append(err1)
        VBKDEReg2.append(err2)
        VBKDEReg3.append(err3)
        
        VBKDERegLOO.append(err4)

        print(f"Iteration {trial}, Time taken: {time.time()-time0:0.4f} seconds")

    # Plot Boxplots and print numerical results
    PWConstant1, PWConstant2, PWConstant3, PWConstantLOO = np.asarray(PWConstant1), np.asarray(PWConstant2), np.asarray(PWConstant3), np.asarray(PWConstantLOO)
    PWLinear1, PWLinear2, PWLinear3, PWLinearLOO = np.asarray(PWLinear1), np.asarray(PWLinear2), np.asarray(PWLinear3), np.asarray(PWLinearLOO)
    MAverage1, MAverage2, MAverage3, MAverageLOO = np.asarray(MAverage1), np.asarray(MAverage2), np.asarray(MAverage3), np.asarray(MAverageLOO)
    GaussKReg1, GaussKReg2, GaussKReg3, GaussKRegLOO = np.asarray(GaussKReg1), np.asarray(GaussKReg2), np.asarray(GaussKReg3), np.asarray(GaussKRegLOO)
    TricubeKReg1, TricubeKReg2, TricubeKReg3, TricubeKRegLOO = np.asarray(TricubeKReg1), np.asarray(TricubeKReg2), np.asarray(TricubeKReg3), np.asarray(TricubeKRegLOO)
    EpanechnikovKReg1, EpanechnikovKReg2, EpanechnikovKReg3, EpanechnikovKRegLOO = np.asarray(EpanechnikovKReg1), np.asarray(EpanechnikovKReg2), np.asarray(EpanechnikovKReg3), np.asarray(EpanechnikovKRegLOO)
    GaussLinear1, GaussLinear2, GaussLinear3, GaussLinearLOO = np.asarray(GaussLinear1), np.asarray(GaussLinear2), np.asarray(GaussLinear3), np.asarray(GaussLinearLOO)
    TricubeLinear1, TricubeLinear2, TricubeLinear3, TricubeLinearLOO = np.asarray(TricubeLinear1), np.asarray(TricubeLinear2), np.asarray(TricubeLinear3), np.asarray(TricubeLinearLOO)
    EpanechnikovLinear1, EpanechnikovLinear2, EpanechnikovLinear3, EpanechnikovLinearLOO = np.asarray(EpanechnikovLinear1), np.asarray(EpanechnikovLinear2), np.asarray(EpanechnikovLinear3), np.asarray(EpanechnikovLinearLOO)
    GaussQuad1, GaussQuad2, GaussQuad3, GaussQuadLOO = np.asarray(GaussQuad1), np.asarray(GaussQuad2), np.asarray(GaussQuad3), np.asarray(GaussQuadLOO)
    VBKDEReg1, VBKDEReg2, VBKDEReg3, VBKDERegLOO = np.asarray(VBKDEReg1), np.asarray(VBKDEReg2), np.asarray(VBKDEReg3), np.asarray(VBKDERegLOO)

    data1 = [PWConstant1, PWLinear1, MAverage1, GaussKReg1, TricubeKReg1, EpanechnikovKReg1, GaussLinear1, TricubeLinear1, EpanechnikovLinear1, GaussQuad1, VBKDEReg1]
    data2 = [PWConstant2, PWLinear2, MAverage2, GaussKReg2, TricubeKReg2, EpanechnikovKReg2, GaussLinear2, TricubeLinear2, EpanechnikovLinear2, GaussQuad2, VBKDEReg2]
    data3 = [PWConstant3, PWLinear3, MAverage3, GaussKReg3, TricubeKReg3, EpanechnikovKReg3, GaussLinear3, TricubeLinear3, EpanechnikovLinear3, GaussQuad3, VBKDEReg3]
    dataLOO = [PWConstantLOO, PWLinearLOO, MAverageLOO, GaussKRegLOO, TricubeKRegLOO, EpanechnikovKRegLOO, GaussLinearLOO, TricubeLinearLOO, EpanechnikovLinearLOO, GaussQuadLOO, VBKDERegLOO]

    with open("Simulation1.txt", "w") as f:
        for arr in data1:
            for j in range(arr.shape[0]):
                f.write(str(arr[j]) + " ")
            f.write("\n")

    with open("Simulation2.txt", "w") as f:
        for arr in data2:
            for j in range(arr.shape[0]):
                f.write(str(arr[j]) + " ")
            f.write("\n")

    with open("Simulation3.txt", "w") as f:
        for arr in data3:
            for j in range(arr.shape[0]):
                f.write(str(arr[j]) + " ")
            f.write("\n")

    with open("Simulation4.txt", "w") as f:
        for arr in dataLOO:
            for j in range(arr.shape[0]):
                f.write(str(arr[j]) + " ")
            f.write("\n")

    data1 = [PWConstant1, PWLinear1, MAverage1, GaussKReg1, TricubeKReg1[~np.isnan(TricubeKReg1)], EpanechnikovKReg1[~np.isnan(EpanechnikovKReg1)], GaussLinear1, TricubeLinear1[~np.isnan(TricubeLinear1)], EpanechnikovLinear1[~np.isnan(EpanechnikovLinear1)], GaussQuad1, VBKDEReg1]
    data2 = [PWConstant2, PWLinear2, MAverage2, GaussKReg2, TricubeKReg2[~np.isnan(TricubeKReg2)], EpanechnikovKReg2[~np.isnan(EpanechnikovKReg2)], GaussLinear2, TricubeLinear2[~np.isnan(TricubeLinear2)], EpanechnikovLinear2[~np.isnan(EpanechnikovLinear2)], GaussQuad2, VBKDEReg2]
    data3 = [PWConstant3, PWLinear3, MAverage3, GaussKReg3, TricubeKReg3[~np.isnan(TricubeKReg3)], EpanechnikovKReg3[~np.isnan(EpanechnikovKReg3)], GaussLinear3, TricubeLinear3[~np.isnan(TricubeLinear3)], EpanechnikovLinear3[~np.isnan(EpanechnikovLinear3)], GaussQuad3, VBKDEReg3]
    dataLOO = [PWConstantLOO, PWLinearLOO, MAverageLOO, GaussKRegLOO, TricubeKRegLOO[~np.isnan(TricubeKRegLOO)], EpanechnikovKRegLOO[~np.isnan(EpanechnikovKRegLOO)], GaussLinearLOO, TricubeLinearLOO[~np.isnan(TricubeLinearLOO)], EpanechnikovLinearLOO[~np.isnan(EpanechnikovLinearLOO)], GaussQuadLOO, VBKDERegLOO]

    plt.figure(figsize=(14, 5))
    plt.boxplot(data1, labels=["PW Const", "PW Linear", "Mov. Ave", "Gauss K Ave", "Tri K Ave", "Ep K Ave", "Gauss Lin", "Tri Lin", "Ep Lin", "Gauss Quad", "VBKDE"])
    plt.title(f"{NUM_TRIALS} Trials Error Calculations, {test1.shape[1]} Evaluation Points")
    plt.xlabel("Method")
    plt.ylabel("Ground Truth RMSE")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.boxplot(data2, labels=["PW Const", "PW Linear", "Mov. Ave", "Gauss K Ave", "Tri K Ave", "Ep K Ave", "Gauss Lin", "Tri Lin", "Ep Lin", "Gauss Quad", "VBKDE"])
    plt.title(f"{NUM_TRIALS} Trials Error Calculations, {test2.shape[1]} Evaluation Points")
    plt.xlabel("Method")
    plt.ylabel("Ground Truth RMSE")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.boxplot(data3, labels=["PW Const", "PW Linear", "Mov. Ave", "Gauss K Ave", "Tri K Ave", "Ep K Ave", "Gauss Lin", "Tri Lin", "Ep Lin", "Gauss Quad", "VBKDE"])
    plt.title(f"{NUM_TRIALS} Trials Error Calculations, {test3.shape[1]} Evaluation Points")
    plt.xlabel("Method")
    plt.ylabel("Ground Truth RMSE")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.boxplot(dataLOO, labels=["PW Const", "PW Linear", "Mov. Ave", "Gauss K Ave", "Tri K Ave", "Ep K Ave", "Gauss Lin", "Tri Lin", "Ep Lin", "Gauss Quad", "VBKDE"])
    plt.title(f"{NUM_TRIALS} Trials Error Calculations, Leave One Out")
    plt.xlabel("Method")
    plt.ylabel("LOO RMSE")
    plt.tight_layout()
    plt.show()

    print(f"Tricube Fails: KReg {counttrikernel}, Linear {counttrilinear}")
    print(f"Epanech Fails: KReg {countepkernel}, Linear {counteplinear}")