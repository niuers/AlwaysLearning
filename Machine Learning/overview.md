* Effective number of parameters of KNN
* 

# Overview
1. Regression and classification can both be viewed as a task in function approximation.
1. A large subset of the most popular techniques in use today are variants of these two simple procedures, i.e. Least Squares and KNN.

# Linear Model
1. The linear model makes huge assumptions about structure and yields stable but possibly inaccurate preditions. It has low variance and potentially high bias.

## Extentions
1. Local regression fits linear models by locally weighted least squares, rather than fitting constants locally.
1. Linear models fit to a basis expansion of the original inputs allow arbitrarily complex models.
1. Projection pursuit and neural network models consist of sums of non-linearly transformed linear models.

# K-nearest neighbors (KNN)
1. The method of k-nearest neighbors make very mild structural assumptions about the underlying data: its predictions are often accurate but can be unstable. Any particular subregion of the decision boundary depends on a handful of input points and their particular positions, and is thus wiggly and unstable. It has high variance and low bias.
1. The error on the training data should be approximately an increasing function of k, and will always be 0 for k = 1.
1. It appears that KNN fits have a single parameter, the number of neighbors k, compared to the p parameters in least-squares fits. However the effective number of parameters of KNN is **N/k** and is generally bigger than p, and decreases with increasing k.
1. It is also clear that we cannot use sum-of-squared errors on the training set as a criterion for picking k, since we would always pick k = 1!

## Extend to Kernel methods
1. Kernel methods use weights that decrease smoothly to zero with distance from the target point, rather than the effective 0/1 weights used by KNN.
1. In high-dimensional spaces the distance kernels are modified to emphasize some variable more than others.


