"""
SimulationStocksKSUSD.py

Simulation study using the trend estimation methods in KSUSD.py and S&P 500
Index data, found in data_csv.csv.

For a fixed number of trials, randomly select 10% of the full data set as the
"observed data," and discard the rest. 

For each of the methods in KSUSD.py, fit the model to this random data and 
estimate the trend at each of the known ground truth values (that were 
previously discarded). 
    
For each model, find the parameters which minimize the ground truth RMSE in 
comparison to the full dataset. Additionally, find the parameters which 
minimize the LOO RMSE for each model. Note that these parameters need not be 
the same. 

Store the minimized RMSE values for each model in each trial in the text 
documents "Simulation5.txt", and "Simulation6.txt" for later use.

Additionally, plot boxplots showing the distribution of the RMSE Values for
each model.

In experimentation, several methods were found to fail in different cases.
These methods are wrapped in a try-except block for now, and if an exception
is thrown, the model is skipped for that trial. The total number of failures 
for each method are printed at the end of the analysis. 

NOTE: This script in particular is likely in need of a lot of attention, since
the results seem to be meaningless. Due to the project time constraints, I was
unable to solve these issues before the deadline.

Author: Eric Jameson
"""

from KSUSD import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    NUM_TRIALS = 1

    PWConstantGT, PWConstantLOO = [], []
    PWLinearGT, PWLinearLOO = [], []
    MAverageGT, MAverageLOO = [], []
    GaussKRegGT, GaussKRegLOO = [], []
    countgausskernel = 0
    TricubeKRegGT, TricubeKRegLOO = [], []
    counttrikernel = 0
    EpanechnikovKRegGT, EpanechnikovKRegLOO = [], []
    countepkernel = 0
    GaussLinearGT, GaussLinearLOO = [], []
    countgausslinear = 0
    TricubeLinearGT, TricubeLinearLOO = [], []
    counttrilinear = 0
    EpanechnikovLinearGT, EpanechnikovLinearLOO = [], []
    counteplinear = 0
    GaussQuadGT, GaussQuadLOO = [], []
    countgaussquad = 0
    VBKDERegGT, VBKDERegLOO = [], []
    countvbkde = 0

    DATA = pd.read_csv("data_csv.csv")
    
    DATA = np.asarray(DATA['SP500'])
    DATA = DATA[1248:1764]

    if False:
        plt.figure()
        plt.title("S&P 500 Monthly Data Jan 1, 1975 - Dec 1, 2017")
        plt.ylabel("S&P Index")
        plt.xlabel("Month")
        plt.plot(DATA, label="S&P Index")
        plt.legend(loc="upper left")


    for trial in range(NUM_TRIALS):
        time0 = time.time()
        t = np.arange(DATA.shape[0])
        y = DATA
        t_sparse = sorted(list(set(np.random.randint(0, DATA.shape[0], size=int(DATA.shape[0]*0.1)).tolist())))
        y_sparse = DATA[t_sparse]
        TrueData = np.vstack((t, y))
        FullData = np.vstack((t_sparse, y_sparse))
   
        MAX_DIST = t_sparse[-1] - t_sparse[0] + 1
          
            
        t = np.reshape(t, (-1,1)).T
        t_sparse = np.reshape(t_sparse, (-1,1)).T
        test = np.linspace(-100,600,1000)
        test = np.reshape(test, (-1,1)).T


        K_fracs = np.linspace(0.1, 0.55, 5).tolist()
        K_lambdas = np.linspace(MAX_DIST, MAX_DIST*2, 5).tolist()
        K_lambda_gauss = np.linspace(MAX_DIST, MAX_DIST*2, 5).tolist()   
        
        # Piecewise Constant
        print("\tPiecewise Constant...")
        gt = EvaluateGroundTruth(TrueData, PiecewiseConstant, **{'X': t, 'data_matrix': FullData})
        loo = EvaluateLOO(t, FullData, PiecewiseConstant, **{})
        
        PWConstantGT.append(gt)
        PWConstantLOO.append(loo)

        # Piecewise Linear
        print("\tPiecewise Linear...")
        gt = EvaluateGroundTruth(TrueData, PiecewiseLinear, **{'X': t, 'data_matrix': FullData})
        loo = EvaluateLOO(t, FullData, PiecewiseLinear, **{})
        
        PWLinearGT.append(gt)
        PWLinearLOO.append(loo)

        # Moving Average 
        print("\tMoving Average...")
        err1, _ = FindOptimalParams(t, FullData, MovingAverage, EvaluateGroundTruth, TrueData, **{'K_frac': K_fracs})
        err2, _ = FindOptimalParams(t, FullData, MovingAverage, EvaluateLOO, **{'K_frac': K_fracs})

        MAverageGT.append(err1)
        MAverageLOO.append(err2)

        # Kernel Smoothing
        # Gaussian Kernel 
        print("\tKernel Smoothing...")
        try:
            err1, glam1 = FindOptimalParams(t, FullData, KernelSmoothing, EvaluateGroundTruth, TrueData, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
            err2, glam2 = FindOptimalParams(t, FullData, KernelSmoothing, EvaluateLOO, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        except:
            err1 = np.nan
            err2 = np.nan
            
            countgausskernel += 1
        
        GaussKRegGT.append(err1)
        GaussKRegLOO.append(err2)

        # Tricube Kernel
        try:
            err1, _ = FindOptimalParams(t, FullData, KernelSmoothing, EvaluateGroundTruth, TrueData, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(t, FullData, KernelSmoothing, EvaluateLOO, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})    
        except:
            err1 = np.nan
            err2 = np.nan
           
            counttrikernel += 1
            
        TricubeKRegGT.append(err1)
        TricubeKRegLOO.append(err2)
        
        # Epanechnikov Kernel
        try:
            err1, _ = FindOptimalParams(t, FullData, KernelSmoothing, EvaluateGroundTruth, TrueData, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(t, FullData, KernelSmoothing, EvaluateLOO, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
        except:
            err1 = np.nan
            err2 = np.nan
            
            countepkernel += 1
        
        EpanechnikovKRegGT.append(err1)
        EpanechnikovKRegLOO.append(err2)

        # # Local Linear Smoothing
        # Gaussian Linear 
        print("\tLinear Smoothing...")
        try:
            err1, _ = FindOptimalParams(t, FullData, LocalLinear, EvaluateGroundTruth, TrueData, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
            err2, _ = FindOptimalParams(t, FullData, LocalLinear, EvaluateLOO, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        except:
            err1 = np.nan
            err2 = np.nan
            
            countgausslinear += 1
        
        GaussLinearGT.append(err1)
        GaussLinearLOO.append(err2)

        # Tricube Linear 
        try:
            err1, _ = FindOptimalParams(t, FullData, LocalLinear, EvaluateGroundTruth, TrueData, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(t, FullData, LocalLinear, EvaluateLOO, **{'Kernel': TricubeKernel, 'K_lambda': K_lambdas})
        except:
            err1 = np.nan 
            err2 = np.nan

            counttrilinear += 1

        TricubeLinearGT.append(err1)
        TricubeLinearLOO.append(err2)

        # Epanechnikov Linear 
        try:
            err1, _ = FindOptimalParams(t, FullData, LocalLinear, EvaluateGroundTruth, TrueData, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
            err2, _ = FindOptimalParams(t, FullData, LocalLinear, EvaluateLOO, **{'Kernel': EpanechnikovKernel, 'K_lambda': K_lambdas})
        except:
            err1 = np.nan
            err2 = np.nan 
            
            counteplinear += 1

        EpanechnikovLinearGT.append(err1)
        EpanechnikovLinearLOO.append(err2)
        
        # Gaussian Quadratic 
        print("\tQuadratic...")
        try:
            err1, _ = FindOptimalParams(t, FullData, LocalQuadratic, EvaluateGroundTruth, TrueData, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
            err2, _ = FindOptimalParams(t, FullData, LocalQuadratic, EvaluateLOO, **{'Kernel': GaussianKernel, 'K_lambda': K_lambda_gauss})
        except:
            err1 = np.nan
            err2 = np.nan
            
            countgaussquad += 1
            
        GaussQuadGT.append(err1)
        GaussQuadLOO.append(err2)

        # VBKDE Regression 
        print("\tVBKDE...")
        try:
            err1, _ = FindOptimalParams(t, FullData, VBKDERegression, EvaluateGroundTruth, TrueData, **{'KDE_points': test, 'K_frac': K_fracs, 'KDE_Kernel': GaussianKernel, 'Method': KernelSmoothing, 'params': {'Kernel': GaussianKernel, 'K_lambda': 0.2}})
            err2, _ = FindOptimalParams(t, FullData, VBKDERegression, EvaluateLOO, **{'KDE_points': test, 'K_frac': K_fracs, 'KDE_Kernel': GaussianKernel, 'Method': KernelSmoothing, 'params': {'Kernel': GaussianKernel, 'K_lambda': 0.2}})
        except:
            err1 = np.nan
            err2 = np.nan
            
            countvbkde += 1
            
        VBKDERegGT.append(err1)            
        VBKDERegLOO.append(err2)

        print(f"Iteration {trial}, Time taken: {time.time()-time0:0.4f} seconds")

    # Plot Boxplots and print numerical results
    PWConstantGT, PWConstantLOO = np.asarray(PWConstantGT), np.asarray(PWConstantLOO)
    PWLinearGT, PWLinearLOO = np.asarray(PWLinearGT), np.asarray(PWLinearLOO)
    MAverageGT, MAverageLOO = np.asarray(MAverageGT), np.asarray(MAverageLOO)
    GaussKRegGT, GaussKRegLOO = np.asarray(GaussKRegGT), np.asarray(GaussKRegLOO)
    TricubeKRegGT, TricubeKRegLOO = np.asarray(TricubeKRegGT), np.asarray(TricubeKRegLOO)
    EpanechnikovKRegGT, EpanechnikovKRegLOO = np.asarray(EpanechnikovKRegGT), np.asarray(EpanechnikovKRegLOO)
    GaussLinearGT, GaussLinearLOO = np.asarray(GaussLinearGT), np.asarray(GaussLinearLOO)
    TricubeLinearGT, TricubeLinearLOO = np.asarray(TricubeLinearGT), np.asarray(TricubeLinearLOO)
    EpanechnikovLinearGT, EpanechnikovLinearLOO = np.asarray(EpanechnikovLinearGT), np.asarray(EpanechnikovLinearLOO)
    GaussQuadGT, GaussQuadLOO = np.asarray(GaussQuadGT), np.asarray(GaussQuadLOO)
    VBKDERegGT, VBKDERegLOO = np.asarray(VBKDERegGT), np.asarray(VBKDERegLOO)

    dataGT = [PWConstantGT, PWLinearGT, MAverageGT, GaussKRegGT, TricubeKRegGT, EpanechnikovKRegGT, GaussLinearGT, TricubeLinearGT, EpanechnikovLinearGT, GaussQuadGT, VBKDERegGT]
    dataLOO = [PWConstantLOO, PWLinearLOO, MAverageLOO, GaussKRegLOO, TricubeKRegLOO, EpanechnikovKRegLOO, GaussLinearLOO, TricubeLinearLOO, EpanechnikovLinearLOO, GaussQuadLOO, VBKDERegLOO]

    with open("Simulation5.txt", "w") as f:
        for arr in dataGT:
            for j in range(arr.shape[0]):
                f.write(str(arr[j]) + " ")
            f.write("\n")

    with open("Simulation6.txt", "w") as f:
        for arr in dataLOO:
            for j in range(arr.shape[0]):
                f.write(str(arr[j]) + " ")
            f.write("\n")


    dataGT = [PWConstantGT, PWLinearGT, MAverageGT, GaussKRegGT[~np.isnan(GaussKRegGT)], TricubeKRegGT[~np.isnan(TricubeKRegGT)], EpanechnikovKRegGT[~np.isnan(EpanechnikovKRegGT)], GaussLinearGT[~np.isnan(GaussLinearGT)], TricubeLinearGT[~np.isnan(TricubeLinearGT)], EpanechnikovLinearGT[~np.isnan(EpanechnikovLinearGT)], GaussQuadGT[~np.isnan(GaussQuadGT)], VBKDERegGT[~np.isnan(VBKDERegGT)]]
    dataLOO = [PWConstantLOO, PWLinearLOO, MAverageLOO, GaussKRegLOO[~np.isnan(GaussKRegLOO)], TricubeKRegLOO[~np.isnan(TricubeKRegLOO)], EpanechnikovKRegLOO[~np.isnan(EpanechnikovKRegLOO)], GaussLinearLOO[~np.isnan(GaussLinearLOO)], TricubeLinearLOO[~np.isnan(TricubeLinearLOO)], EpanechnikovLinearLOO[~np.isnan(EpanechnikovLinearLOO)], GaussQuadLOO[~np.isnan(GaussQuadLOO)], VBKDERegLOO[~np.isnan(VBKDERegLOO)]]

    plt.figure(figsize=(14, 5))
    plt.boxplot(dataGT, labels=["PW Const", "PW Linear", "Mov. Ave", "Gauss K Ave", "Tri K Ave", "Ep K Ave", "Gauss Lin", "Tri Lin", "Ep Lin", "Gauss Quad", "VBKDE"])
    plt.title(f"{NUM_TRIALS} Trials Error Calculations, Full Evaluation Points")
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

    print(f"Gaussian Fails: KReg {countgausskernel}, Linear {countgausslinear}, Quad {countgaussquad}")
    print(f"Tricube Fails: KReg {counttrikernel}, Linear {counttrilinear}")
    print(f"Epanech Fails: KReg {countepkernel}, Linear {counteplinear}")
    print(f"VBKDE Fails: {countvbkde}")
    