"""
PlotKSUSD.py

Example of plotting ground truth data using KSUSD.py.

For a fixed numpy random seed, draw 20 random observations from a Uniform[0,1]
random variable, and create a dataset X = sin(4t) + epsilon, where epsilon is 
drawn from a Normal(0, 1/3) random variable.

For each of the methods in KSUSD.py, fit the model to the random data, and plot
the trend estimation for these three cases:
    1. No interpolation between observations
    2. Midpoint interpolation between observations
    3. 5-point interpolation between observations
    
Also produces a plot of the pure data vs. ground truth, and two plots related
to VBKDE Regression.

Author: Eric Jameson
"""

from KSUSD import *
import numpy as np
import matplotlib.pyplot as plt

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

   
    plt.figure()
    plt.title("Simulated Data and True Curve")
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # Piecewise Constant
    PC1 = PiecewiseConstant(test1, FullData)
    PC2 = PiecewiseConstant(test2, FullData)
    PC3 = PiecewiseConstant(test3, FullData)

    plt.figure(figsize=(14,5))
    plt.suptitle("Piecewise Constant Trend Estimation")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(PC1[0,:], PC1[1,:], c="r", label="Piecewise Constant Approximation")
    plt.legend(loc="lower left")
    ax.set_title("20 Evaluation Points")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(PC2[0,:], PC2[1,:], c="r", label="Piecewise Constant Approximation")
    plt.legend(loc="lower left")
    ax.set_title("39 Evaluation Points")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(PC3[0,:], PC3[1,:], c="r", label="Piecewise Constant Approximation")
    plt.legend(loc="lower left")
    ax.set_title("96 Evaluation Points")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Piecewise Linear
    PL1 = PiecewiseLinear(test1, FullData)
    PL2 = PiecewiseLinear(test2, FullData)
    PL3 = PiecewiseLinear(test3, FullData)

    plt.figure(figsize=(14,5))
    plt.suptitle("Piecewise Linear Trend Estimation")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(PL1[0,:], PL1[1,:], c="r", label="Piecewise Linear Approximation")
    plt.legend(loc="lower left")
    ax.set_title("20 Evaluation Points")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(PL2[0,:], PL2[1,:], c="r", label="Piecewise Linear Approximation")
    plt.legend(loc="lower left")
    ax.set_title("39 Evaluation Points")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(PL3[0,:], PL3[1,:], c="r", label="Piecewise Linear Approximation")
    plt.legend(loc="lower left")
    ax.set_title("96 Evaluation Points")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Moving Average
    AverageData1 = MovingAverage(test1, FullData, 0.3)
    AverageData2 = MovingAverage(test2, FullData, 0.3)
    AverageData3 = MovingAverage(test3, FullData, 0.3)
    
    plt.figure(figsize=(14,5))
    plt.suptitle("Moving Average Trend Estimation, K_frac=0.3")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(AverageData1[0,:], AverageData1[1,:], c="r", label="KNN Average")
    plt.legend(loc="lower left")
    ax.set_title("20 Evaluation Points")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(AverageData2[0,:], AverageData2[1,:], c="r", label="KNN Average")
    plt.legend(loc="lower left")
    ax.set_title("39 Evaluation Points")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(AverageData3[0,:], AverageData3[1,:], c="r", label="KNN Average")
    plt.legend(loc="lower left")
    ax.set_title("96 Evaluation Points")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Gaussian Kernel
    GaussianSmoothed1 = KernelSmoothing(test1, FullData, GaussianKernel, 0.2)
    GaussianSmoothed2 = KernelSmoothing(test2, FullData, GaussianKernel, 0.2)
    GaussianSmoothed3 = KernelSmoothing(test3, FullData, GaussianKernel, 0.2)

    plt.figure(figsize=(14,5))
    plt.suptitle("Gaussian Kernel-Weighted Moving Average, lambda=0.2")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianSmoothed1[0,:], GaussianSmoothed1[1,:], c="r", label="Gaussian Kernel Average")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianSmoothed2[0,:], GaussianSmoothed2[1,:], c="r", label="Gaussian Kernel Average")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianSmoothed3[0,:], GaussianSmoothed3[1,:], c="r", label="Gaussian Kernel Average")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Tricube Kernel
    TricubeSmoothed1 = KernelSmoothing(test1, FullData, TricubeKernel, 0.2)
    TricubeSmoothed2 = KernelSmoothing(test2, FullData, TricubeKernel, 0.2)
    TricubeSmoothed3 = KernelSmoothing(test3, FullData, TricubeKernel, 0.2)

    plt.figure(figsize=(14,5))
    plt.suptitle("Tri-cube Kernel-Weighted Moving Average, lambda=0.2")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(TricubeSmoothed1[0,:], TricubeSmoothed1[1,:], c="r", label="Tri-cube Kernel Average")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(TricubeSmoothed2[0,:], TricubeSmoothed2[1,:], c="r", label="Tri-cube Kernel Average")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(TricubeSmoothed3[0,:], TricubeSmoothed3[1,:], c="r", label="Tri-cube Kernel Average")
    plt.legend(loc="lower left")
    ax.set_title("96 Evaluation Points")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Epanechnikov Kernel
    EpanechnikovSmoothed1 = KernelSmoothing(test1, FullData, EpanechnikovKernel, 0.2)
    EpanechnikovSmoothed2 = KernelSmoothing(test2, FullData, EpanechnikovKernel, 0.2)
    EpanechnikovSmoothed3 = KernelSmoothing(test3, FullData, EpanechnikovKernel, 0.2)

    plt.figure(figsize=(14,5))
    plt.suptitle("Epanechnikov Kernel-Weighted Moving Average, lambda=0.2")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(EpanechnikovSmoothed1[0,:], EpanechnikovSmoothed1[1,:], c="r", label="Epanechnikov Kernel Average")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(EpanechnikovSmoothed2[0,:], EpanechnikovSmoothed2[1,:], c="r", label="Epanechnikov Kernel Average")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(EpanechnikovSmoothed3[0,:], EpanechnikovSmoothed3[1,:], c="r", label="Epanechnikov Kernel Average")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Gaussian Linear
    GaussianLinear1 = LocalLinear(test1, FullData, GaussianKernel, 0.2)
    GaussianLinear2 = LocalLinear(test2, FullData, GaussianKernel, 0.2)
    GaussianLinear3 = LocalLinear(test3, FullData, GaussianKernel, 0.2)   

    plt.figure(figsize=(14,5))
    plt.suptitle("Gaussian Kernel Locally Weighted Linear Regression, lambda=0.2")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianLinear1[0,:], GaussianLinear1[1,:], c="r", label="Gaussian Linear Regression")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianLinear2[0,:], GaussianLinear2[1,:], c="r", label="Gaussian Linear Regression")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianLinear3[0,:], GaussianLinear3[1,:], c="r", label="Gaussian Linear Regression")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Tricube Linear
    TricubeLinear1 = LocalLinear(test1, FullData, TricubeKernel, 0.2)
    TricubeLinear2 = LocalLinear(test2, FullData, TricubeKernel, 0.2)
    TricubeLinear3 = LocalLinear(test3, FullData, TricubeKernel, 0.2)
    
    plt.figure(figsize=(14,5))
    plt.suptitle("Tri-cube Kernel Locally Weighted Linear Regression, lambda=0.2")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(TricubeLinear1[0,:], TricubeLinear1[1,:], c="r", label="Tri-cube Linear Regression")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(TricubeLinear2[0,:], TricubeLinear2[1,:], c="r", label="Tri-cube Linear Regression")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(TricubeLinear3[0,:], TricubeLinear3[1,:], c="r", label="Tri-cube Linear Regression")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Epanechnikov Linear
    EpanechnikovLinear1 = LocalLinear(test1, FullData, EpanechnikovKernel, 0.2)
    EpanechnikovLinear2 = LocalLinear(test2, FullData, EpanechnikovKernel, 0.2)
    EpanechnikovLinear3 = LocalLinear(test3, FullData, EpanechnikovKernel, 0.2)

    plt.figure(figsize=(14,5))
    plt.suptitle("Epanechnikov Kernel Locally Weighted Linear Regression, lambda=0.2")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(EpanechnikovLinear1[0,:], EpanechnikovLinear1[1,:], c="r", label="Epanechnikov Linear Regression")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(EpanechnikovLinear2[0,:], EpanechnikovLinear2[1,:], c="r", label="Epanechnikov Linear Regression")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(EpanechnikovLinear3[0,:], EpanechnikovLinear3[1,:], c="r", label="Epanechnikov Linear Regression")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # Gaussian Quadratic
    GaussianQuad1 = LocalQuadratic(test1, FullData, GaussianKernel, 0.3)
    GaussianQuad2 = LocalQuadratic(test2, FullData, GaussianKernel, 0.3)
    GaussianQuad3 = LocalQuadratic(test3, FullData, GaussianKernel, 0.3)   

    plt.figure(figsize=(14,5))
    plt.suptitle("Gaussian Quadratic Smoothing, lambda=0.3")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianQuad1[0,:], GaussianQuad1[1,:], c="r", label="Gaussian Linear")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianQuad2[0,:], GaussianQuad2[1,:], c="r", label="Gaussian Linear")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(GaussianQuad3[0,:], GaussianQuad3[1,:], c="r", label="Gaussian Linear")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()

    # VBKDE Regression
    KDE = VBKDE(test, TrueData, 0.3, GaussianKernel)
    plt.figure()
    plt.title("Variable Bandwidth KDE, K_frac = 0.3")
    plt.plot(KDE[0,:], KDE[1,:], c="tab:orange", label="Kernel Density Estimation")
    plt.scatter(t_sparse, np.zeros(t_sparse.shape[0]), marker=".", label="Simulated time coordinates")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
            
    CDF_vals = IntegrateKDE(KDE, TrueData)
    plt.figure(figsize=(9,5))
    plt.suptitle("Variable Bandwidth Estimated CDF, K_frac = 0.3")
    ax = plt.subplot(121)
    ax.set_title("Observed time points vs. \nEstimated CDF Values")
    plt.scatter(CDF_vals[0,:], CDF_vals[1,:], marker=".", label="Estimated CDF")
    plt.legend(loc="upper left")
      
    sine_vals = IntegrateKDE(KDE, np.vstack((t_true, y_true)))
    ax = plt.subplot(122)
    ax.set_title("Estimated CDF Values vs. \nObserved Time Series Values")
    plt.scatter(CDF_vals[1,:], FullData[1,:], marker=".", label="Estimated CDF Time Series")
    plt.plot(sine_vals[1,:], y_true, c="lime", label="Transformed True Curve")
    plt.subplots_adjust(top=0.834, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.legend(loc="lower left")
    plt.show()
  
    VBKDE1 = VBKDERegression(test1, test, FullData, 0.3, GaussianKernel, LocalLinear, {'Kernel': GaussianKernel, 'K_lambda': 0.1})
    VBKDE2 = VBKDERegression(test2, test, FullData, 0.3, GaussianKernel, LocalLinear, {'Kernel': GaussianKernel, 'K_lambda': 0.1})
    VBKDE3 = VBKDERegression(test3, test, FullData, 0.3, GaussianKernel, LocalLinear, {'Kernel': GaussianKernel, 'K_lambda': 0.1})

    plt.figure(figsize=(14,5))
    plt.suptitle("Variable Bandwidth Gaussian KDE Regression, K_frac=0.3, lambda=0.1")
    ax = plt.subplot(131)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(VBKDE1[0,:], VBKDE1[1,:], c="r", label="VBKDE Regression")
    ax.set_title("20 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(132)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(VBKDE2[0,:], VBKDE2[1,:], c="r", label="VBKDE Regression")
    ax.set_title("39 Evaluation Points")
    plt.legend(loc="lower left")

    ax = plt.subplot(133)
    plt.scatter(t_sparse, y_noise, marker=".", label="Simulated Data")
    plt.plot(t_true, y_true, c="lime", label="True Curve")
    plt.plot(VBKDE3[0,:], VBKDE3[1,:], c="r", label="VBKDE Regression")
    ax.set_title("96 Evaluation Points")
    plt.legend(loc="lower left")
    plt.subplots_adjust(top=0.889, bottom=0.078, left=0.042, right=0.989, hspace=0.2, wspace=0.145)
    plt.show()