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
1. The error on the training data should be approximately an increasing function of k, and will always be ```0``` for ```k = 1```.
1. It appears that KNN fits have a single parameter, the number of neighbors k, compared to the ```p``` parameters in least-squares fits. However the effective number of parameters of KNN is **```N/k```** and is generally bigger than ```p```, and decreases with increasing ```k```.
1. It is also clear that we cannot use sum-of-squared errors on the training set as a criterion for picking ```k```, since we would always pick ```k = 1```!

## Extend to Kernel methods
1. Kernel methods use weights that decrease smoothly to zero with distance from the target point, rather than the effective ```0/1``` weights used by KNN.
1. In high-dimensional spaces the distance kernels are modified to emphasize some variable more than others.


## Curse of dimensionality and why KNN doesn't work in high dimension? 
1. **Definition**: As the dimensionality of the features space increases, the number configurations can grow exponentionally, and thus the number of configurations covered by an observation decreases. (Chris Albon)
1. Sparse sampling makes local neighborhoods (which is used to approximate the theoretically optimal conditional expectation by KNN averaging) in high dimension no longer local: For example, to capture 1% or 10% of the data in a unit hypercube, we must cover 63% or 80% of the range of each input variable. Such neighborhoods are no longer local. Reducing the percentage of data dramatically does not help much either, since the fewer observations we average, the higher is the variance of our fit.
1. Most data points are closer to the boundary of the sample space than to any other data point: This makes prediction much more difficult near the edges of the training sample. One must extrapolate from neighboring sample points rather than interpolate between them.
1. In high dimensions all feasible training samples (The samples are assumed to come from the unknown true probability distribution of data, and each sample has many data points) sparsely populate the input space: The [sampling density, i.e. the number of recorded samples per unit distance along a dimension](https://math.stackexchange.com/questions/283006/what-is-a-sampling-density-why-is-the-sampling-density-proportional-to-n1-p) is proportional to N<sup>1/p</sup>, where ```p``` is the dimension of the input space and ```N``` is the sample size. To keep the same density, the sample size has to grow exponentionally, which is infeasible with high dimension.
1. The complexity of functions of many variables (i.e. the truth relationship of data) can grow exponentially with the dimension, and if we wish to be able to estimate such functions with the same accuracy as function in low dimensions, then we need the size of our training set to grow exponentially as well.

#### How to avoid curse of dimension?  
1. By imposing some heavy restrictions (e.g. assume data is linear) on the class of models being fitted, we can avoid the curse of dimensionality. However, if the assumptions are wrong, all bets are off and the 1-nearest neighbor may dominate. There is a whole spectrum of models between the rigid linear models and the extremely flexible 1-nearest-neighbor models, each with their own assumptions and biases, which have been proposed specifically to avoid the exponential growth in complexity of functions in high dimensions by drawing heavily on these assumptions.

