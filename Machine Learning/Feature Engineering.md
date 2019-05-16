* Deal with **class-imbalanced dataset**. Imbalanced datasets are problematic for modeling because the model will expend most of its effort fitting to the larger class. Since we have plenty of data in both classes, a good way to resolve the problem is to downsample the larger class (restaurants) to be roughly the same size as the smaller class (nightlife).
* It’s essential to tune hyperparameters when comparing models or features. The default settings of a software package will always return a model.
* training a linear classifier boils down to finding the best linear combination of features, which are column vectors of the data matrix. The solution space is characterized by the column space and the null space of the data matrix.
The quality of the trained linear classifier directly depends upon the null space and the column space of the data matrix. A large column space means that there is little linear dependency between the features, which is generally good. The null space contains “novel” data points that cannot be formulated as linear combinations of existing data; a large null space could be problematic. 

# Evaluation of Feature Engineering Procedure
One needs to have a metric of machine learning model performance to evaluate the effectiveness of a feature engineering procedure. 
First obtain a baseline performance, and compare performance against it after the feature engineering procedure.

# Understand Data

## The four levels of data
### The nominal level

It has the weakest structure. It is discrete and order-less. It consists of data that are purely described by name. Basic examples include blood type (A, O, AB), species of animal, or names of people. These types of data are all qualitative.
1. Count the number of different values
```
df.value_counts().index[0]
```
2. Plot the bar chart ('bar') or pie chart ('pie')
```
df['col_name'].value_counts().sort_values(ascending=False).head(20).plot(kind='bar')
```

### The ordinal level

The ordinal scale inherits all of the properties of the nominal level, but has important additional properties:
Data at the ordinal level can be naturally ordered
This implies that some data values in the column can be considered better than or greater than others

As with the nominal level, data at the ordinal level is still categorical in nature, even if numbers are used to represent the categories.
1. Median and percentiles
2. Stem-and-leaf plots
3. Box plot
```
df['col'].value_counts().plot(kind='box')
```

### The interval level

At the interval data level, we are working with numerical data that not only has ordering like at the ordinal level, but also has meaningful differences between values. This means that at the interval level, not only may we order and compare values, we may also add and subtract values.
1. Check number of unique values
```
df['col'].nunique()
```
2. Plot histogram, use sharex=True to put all x-axis in one scale
```
df['col'].hist(by=df['val'], sharex=True, sharey=True, figsize=(10, 10), bins=20)
```
3. Plot mean values
```
df.groupby('val')['col'].mean().plot(kind='line')
```
4. Scatter plot of two columns
```
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df['col1'], df['col2'])
plt.show()
```
5. Plot with groupby
```
df.groupby('col1').mean()['col2'].plot()
```
6. Rolling smoothing
```
f.groupby('col1').mean()['col2'].rolling(20).mean().plot()
```

### The ratio level

Now we have a notion of true zero which gives us the ability to multiply and divide values. It allows ratio statement.
1. Bar chart
```
fig = plt.figure(figsize=(15,5))
ax = fig.gca()

df.groupby('col1')[['col2']].mean().sort_values('col2', ascending=False).tail(20).plot.bar(stacked=False, ax=ax, color='darkorange')
```
## Numerical Data
1. The first sanity check for numeric data is whether the magnitude matters. Do we just need to know whether it’s positive or negative? Or perhaps we only need to know the magnitude at a very coarse granularity? This sanity check is particularly important for automatically accrued numbers such as counts—the number of daily visits to a website, the number of reviews garnered by a restaurant, etc.
1. Next, consider the scale of the features. Models that are smooth functions of input features are sensitive to the scale of the input.
Logical functions, on the other hand, are not sensitive to input feature scale. Another example of a logical function is the step function (e.g., is input x greater than 5?). Decision tree models consist of step functions of input features. Hence, models based on space-partitioning trees (decision trees, gradient boosted machines, random forests) are not sensitive to scale. 

The only exception is if the scale of the input grows over time, which is the case if the feature is an accumulated count of some sort—eventually it will grow outside of the range that the tree was trained on. If this might be the case, then it might be necessary to rescale the inputs periodically. Another solution is the bin-counting method

1. It’s also important to consider the distribution of numeric features. 
The distribution of input features matters to some models more than others. For instance, the training process of a linear regression model assumes that prediction errors are distributed like a Gaussian. This is usually fine, except when the prediction target spreads out over several orders of magnitude. In this case, the Gaussian error assumption likely no longer holds. One way to deal with this is to transform the output target in order to tame the magnitude of the growth. (Strictly speaking this would be target engineering, not feature engineering.) Log transforms, which are a type of power transform, take the distribution of the variable closer to Gaussian.

### Feature space vs Data space
Collectively, a collection of data can be visualized in feature space as a point cloud. Conversely, we can visualize features in data space. 

### Dealing with Counts
It is a good idea to check the scale and determine whether to keep the data as raw numbers, convert them into binary values to indicate presence, or bin them into coarser granularity. 

1. Binarization
1. Quantization or Binning. Raw counts that span several orders of magnitude are problematic for many models. In a linear model, the same linear coefficient would have to work for all possible values of the count. Large counts could also wreak havoc in unsupervised learning methods such as k-means clustering, which uses Euclidean distance as a similarity function to measure the similarity between data points. A large count in one element of the data vector would outweigh the similarity in all other elements, which could throw off the entire similarity measurement.
One solution is to contain the scale by quantizing the count. In other words, we group the counts into bins, and get rid of the actual count values. Quantization maps a continuous number to a discrete one. We can think of the discretized numbers as an ordered sequence of bins that represent a measure of intensity.

In order to quantize data, we have to decide how wide each bin should be. The solutions fall into two categories: fixed-width or adaptive. 
   1. FIXED-WIDTH BINNING: With fixed-width binning, each bin contains a specific numeric range. The ranges can be custom designed or automatically segmented, and they can be linearly scaled or exponentially scaled. To map from the count to the bin, we simply divide by the width of the bin and take the integer part. When the numbers span multiple magnitudes, it may be better to group by powers of 10 (or powers of any constant): 0–9, 10–99, 100–999, 1000–9999, etc. The bin widths grow exponentially, going from O(10), to O(100), O(1000), and beyond. To map from the count to the bin, we take the log of the count. Exponential-width binning is very much related to the log transform
   1. QUANTILE BINNING: But if there are large gaps in the counts, then there will be many empty bins with no data. This problem can be solved by adaptively positioning the bins based on the distribution of the data. This can be done using the quantiles of the distribution. 
  
### Log Transformation
The log function compresses the range of large numbers and expands the range of small numbers. The larger x is, the slower log(x) increments. The log transform is a powerful tool for dealing with positive numbers with a heavy-tailed distribution.

### Power Transforms: Generalization of the Log Transform
The log transform is a specific example of a family of transformations known as power transforms. In statistical terms, these are variance-stabilizing transformations. To understand why variance stabilization is good, consider the Poisson distribution. This is a heavy-tailed distribution with a variance that is equal to its mean: hence, the larger its center of mass, the larger its variance, and the heavier the tail. Power transforms change the distribution of the variable so that the variance is no longer dependent on the mean. For example, suppose a random variable X has the Poisson distribution. If we transform X by taking its square root, the variance of X˜=sqrt(X) is roughly constant, instead of being equal to the mean.

A simple generalization of both the square root transform and the log transform is known as the Box-Cox transform
Setting λ to be less than 1 compresses the higher values, and setting λ higher than 1 has the opposite effect.

The Box-Cox formulation only works when the data is positive. For nonpositive data, one could shift the values by adding a fixed constant. When applying the Box-Cox transformation or a more general power transform, we have to determine a value for the parameter λ. This may be done via maximum likelihood (finding the λ that maximizes the Gaussian likelihood of the resulting transformed signal) or Bayesian methods. A full treatment of the usage of Box-Cox and general power transforms is outside the scope of this book. Interested readers may find more information on power transforms in Econometric Methods by Johnston and DiNardo (1997).

* A probability plot, or probplot, is an easy way to visually compare an empirical distribution of data against a theoretical distribution. This is essentially a scatter plot of observed versus theoretical quantiles. 

## Feature Scaling or Normalization/feature normalization

### Min-Max Scaling
### Standardization (Variance Scaling)
### DON’T “CENTER” SPARSE DATA!
Use caution when performing min-max scaling and standardization on sparse features. Both subtract a quantity from the original feature value. For min-max scaling, the shift is the minimum over all values of the current feature; for standardization, it is the mean. If the shift is not zero, then these two transforms can turn a sparse feature vector where most values are zero into a dense one. This in turn could create a huge computational burden for the classifier, depending on how it is implemented (not to mention that it would be horrendous if the representation now included every word that didn’t appear in a document!). Bag-of-words is a sparse representation, and most classification libraries optimize for sparse inputs.
### ℓ^2 Normalization
This technique normalizes (divides) the original feature value by what’s known as the ℓ^2 norm, also known as the Euclidean norm. 

After ℓ2 normalization, the feature column has norm 1. This is also sometimes called ℓ2 scaling.

* No matter the scaling method, feature scaling always divides the feature by a constant (known as the normalization constant). Therefore, it does not change the shape of the single-feature distribution. 

## Interaction Features
* A simple pairwise interaction feature is the product of two features. The analogy is the logical AND.
Decision tree–based models get this for free, but generalized linear models often find interaction features very helpful.

* The training and scoring time of a linear model with pairwise interaction features would go from O(n) to O(n2), where n is the number of singleton features.

There are a few ways around the computational expense of higher-order interaction features. One could perform feature selection on top of all of the interaction features. Alternatively, one could more carefully craft a smaller number of complex features.

## Feature Selection
### Filtering

Filtering techniques preprocess features to remove ones that are unlikely to be useful for the model. For example, one could compute the correlation or mutual information between each feature and the response variable, and filter out the features that fall below a threshold.

Filtering techniques are much cheaper than the wrapper techniques described next, but they do not take into account the model being employed. Hence, they may not be able to select the right features for the model. It is best to do prefiltering conservatively, so as not to inadvertently eliminate useful features before they even make it to the model training step.

### Wrapper methods

The wrapper method treats the model as a black box that provides a quality score of a proposed subset for features. There is a separate method that iteratively refines the subset.

### Embedded methods
These methods perform feature selection as part of the model training process. For example, a decision tree inherently performs feature selection because it selects one feature on which to split the tree at each training step. Another example is the ℓ1 regularizer, which can be added to the training objective of any linear model. The ℓ1 regularizer encourages models that use a few features as opposed to a lot of features, so it’s also known as a sparsity constraint on the model. Embedded methods incorporate feature selection as part of the model training process. 

A full treatment of feature selection is outside the scope of this book. Interested readers may refer to the survey paper by Guyon and Elisseeff (2003).


















## Structured Data Type
### Basic Checks

1. Count the number of rows, missing values, data types etc
```df.info()```
2. Count how many missing values in each column
```df.isnull().sum()```
3. Check descriptive statistics on quantitative columns
```df.describe()```
4. Check the differences of histograms for data in category 1 vs. category 2.
5. Check correlation map
```
# look at the heatmap of the correlation matrix of data
import seaborn as sns
sns.heatmap(df.corr())
```



## Data Cleanup
### Remove rows with missing values in them
Most time this strategy is not very good.
### Impute (fill in) missing values
Use pipeline to fill with mean or median etc.
```
from sklearn.preprocessing import Imputer
```

### Data normalization
1. Z-score standardization
1. Min-max scaling
1. Row normalization: It ensure that each row of data has a unit norm, meaning that each row will be the same vector length.

### A list of some popular learning algorithms that are affected by the scale of data
1. KNN-due to its reliance on the Euclidean Distance
1. K-Means Clustering - same reasoning as KNN
1. Logistic regression, SVM, neural networks - if you are using gradient descent to learn weights
1. Principal component analysis - eigen vectors will be skewed towards larger columns
1. RBF Kernels, and anything that uses the Euclidean distance.

### Encoding Categorical Data
1. Encoding at the nominal level
   1. Transform our categorical data into dummy variables. 
The dummy variable trap is when you have independent variables that are multicollinear, or highly correlated. Simply put, these variables can be predicted from each other. So, in our gender example, the dummy variable trap would be if we include both female as (0|1) and male as (0|1), essentially creating a duplicate category. It can be inferred that a 0 female value indicates a male.

1. Encoding at the ordinal level
To maintain the order, we will use a label encoder. 

### Bucketing Continuous features into categories
```
pandas.cut
```

# Feature Construction
## Polynomial Features
A key method of working with numerical data and creating more features is through scikit-learn's PolynomialFeatures class. In its simplest form, this constructor will create new columns that are products of existing columns to capture feature interactions.

## Text Specific Feature Construction
1. Bag of words representation
   1. Tokenizing
   1. Counting
   1. Normalizing

1. CountVectorizer
   It converts text columns into matrices where columns are tokens and cell values are counts of occurrences of each token in each document. The resulting matrix is referred to as a document-term matrix because each row will represent a document (in this case, a tweet) and each column represents a term (a word).
1. The Tf-idf vectorizer

# Feature Selection
* If your features are mostly categorical, you should start by trying to implement a SelectKBest with a Chi2 ranker or a tree-based model selector.
* If your features are largely quantitative, using linear models as model-based selectors and relying on correlations tends to yield greater results.
* If you are solving a binary classification problem, using a Support Vector Classification model along with a SelectFromModel selector will probably fit nicely, as the SVC tries to find coefficients to optimize for binary classification tasks.
* A little bit of EDA can go a long way in manual feature selection. The importance of having domain knowledge in the domain from which the data originated cannot be understated.

## Statistical-based
Model-based selection relies on a preprocessing step that involves training a secondary machine learning model and using that model's predictive power to select features.
### Pearson correlation

We will assume that the more correlated a feature is to the response, the more useful it will be. Any feature that is not as strongly correlated will not be as useful to us.
It is worth noting that Pearson's correlation generally requires that each column be normally distributed (which we are not assuming). We can also largely ignore this requirement because our dataset is large (over 500 is the threshold).
Correlation coefficients are also used to determine feature interactions and redundancies. A key method of reducing overfitting in machine learning is spotting and removing these redundancies.
### Hypothesis testing

Feature selection via hypothesis testing will attempt to select only the best features from a dataset, these tests rely more on formalized statistical methods and are interpreted through what are known as p-values. 
In the case of feature selection, the hypothesis we wish to test is along the lines of: True or False: This feature has no relevance to the response variable. We want to test this hypothesis for every feature and decide whether the features hold some significance in the prediction of the response. 
Simply put, the lower the p-value, the better the chance that we can reject the null hypothesis. For our purposes, the smaller the p-value, the better the chances that the feature has some relevance to our response variable and we should keep it.
```
# SelectKBest selects features according to the k highest scores of a given scoring function
from sklearn.feature_selection import SelectKBest

# This models a statistical test known as ANOVA
from sklearn.feature_selection import f_classif

# f_classif allows for negative values, not all do
# chi2 is a very common classification criteria but only allows for positive values
# regression has its own statistical tests
```
The big take away from this is that the f_classif function will perform an ANOVA test (a type of hypothesis test) on each feature on its own (hence the name univariate testing) and assign that feature a p-value. The SelectKBestwill rank the features by that p-value (the lower the better) and keep only the best k (a human input) features. Let's try this out in Python.

## Model-based
1. Tree-based model feature importance
1. Linear model's coef_ attribute

# Feature Transformation
It's a suite of algorithms designed to alter the internal structure of data to produce mathematically superior super-columns. The toughest part of feature transformations is the suspension of our belief that the original feature space is the best. We must be open to the fact that there may be other mathematical axes and systems that describe our data just as well with fewer features, or possibly even better. Feature transformation algorithms are able to construct new features by selecting the best of all columns and combining this latent structure with a few brand new columns
## Dimension Deduction
### PCA
1. A scree plot is a simple line graph that shows the percentage of total variance explained in the data by each principal component. To build this plot, we will sort the eigenvalues in order of descending value and plot the cumulative variance explained by each component and all components prior.
1. Centering data doesn't affect the principal components. The reason this is happening is because matrices have the same covariance matrix as their centered counterparts. If two matrices have the same covariance matrix, then they will have the same eignenvalue decomposition.
1. PCA is scale-invariant, meaning that scale affects the components. Note that when we say scaling, we mean centering and dividing by the standard deviation. It's because once we scaled our data, the columns' covariance with one another became more consistent and the variance explained by each principal component was spread out instead of being solidified in a single PC. In practice and production, we generally recommend scaling, but it is a good idea to test your pipeline's performance on both scaled and un-scaled data.
1. PCA 1, our first principal component, should be carrying the majority of the variance within it, which is why the projected data is spread out mostly across the new x axis
1. The assumption that we were making was that the original data took on a shape that could be decomposed and represented by a single linear transformation (the matrix operation).

### Linear Discriminant Analysis

1. The main difference between LDA and PCA is that instead of focusing on the variance of the data as a whole like PCA, LDA optimizes the lower-dimensional space for the best class separability. This means that the new coordinate system is more useful in finding decision boundaries for classification models, which is perfect for us when building classification pipelines.
1. The reason that LDA is extremely useful is that separating based on class separability helps us avoid overfitting in our machine learning pipelines. This is also known as preventing the curse of dimensionality. LDA also reduces computational costs.
1. The way LDA is trying to work is by drawing decision boundaries between our classes. Because we only have three classes in the iris, we may only draw up to two decision boundaries. In general, fitting LDA to a dataset with n classes will only produce up to n-1 components in eigenvalues, i.e. the rest eigen values are close to zero.
1. This is because the goal of scalings_ is not to create a new coordinate system, but just to point in the direction of boundaries in the data that optimizes for class separability.
1. It is sufficient to understand that the main difference between PCA and LDA is that PCA is an unsupervised method that captures the variance of the data as a whole whereas LDA, a supervised method, uses the response variable to capture class separability.
1. It is common to correctly use all three of these algorithms in the same pipelines and perform hyper-parameter tuning to fine-tune the process. This shows us that more often than not, the best production-ready machine learning pipelines are in fact a combination of multiple feature engineering methods.

### PCA and LDA are extremely powerful tools, but have limitations.
1. Both of them are linear transformations, which means that they can only create linear boundaries and capture linear qualities in our data. 
1. They are also static transformations. No matter what data we input into a PCA or LDA, the output is expected and mathematical. If the data we are using isn't a good fit for PCA or LDA (they exhibit non-linear qualities, for example, they are circular), then the two algorithms will not help us, no matter how much we grid search.

# Feature Learning
It focuses on feature learning using non-parametric algorithms (those that do not depend on the shape of the data) to automatically learn new features. They do not make any assumptions about the shape of the incoming data and rely on stochastic learning.
instead of throwing the same equation at the matrix of data every time, they will attempt to figure out the best features to extract by looking at the data points over and over again (in epochs) and converge onto a solution (potentially different ones at runtime).
## No-parametric fallacy
It is important to mention that a model being non-parametric doesn't mean that there are no assumptions at all made by the model during training.
While the algorithms that we will be introducing in this chapter forgo the assumption on the shape of the data, they still may make assumptions on other aspects of the data, for example, the values of the cells.
They all involve learning brand new features from raw data. They then use these new features to enhance the way that they interact with data.

## Restricted Boltzmann Machine
1. A simple deep learning architecture that is set up to learn a set number of new dimensions based on a probabilistic model that data follows.
1. The features that are extracted by RBMs tend to work best when followed by linear models such as linear regression, logistic regression, perceptron's, and so on.
1. The restriction in the RBM is that we do not allow for any intra-layer communication. This lets nodes independently create weights and biases that end up being (hopefully) independent features for our data.

## Word Embedding

Mastering a subject is not just about knowing the definitions and being able to derive the formulas. It is not enough to know how the mechanism works and what it can do - one must also understand why it is designed that way, how it relates to other techniques, and what the pros and cons of each approach are. Mastery is about knowing precisely how something is done, having an intuition for the underlying principles, and integrating it into one's existing web of knowledge. One does not become a master of something by simply reading a book, though a good book can open new doors. It has to involve practice - putting the ideas to use, which is an iterative process. With every iteration, we know the ideas better and become increasingly more adept and creative at applying them. The goal of this book is to facilitate the application of its ideas.


# Text Data: Flattening, Filtering, and Chunking

## Bag of Words
* The ordering of words in the vector is not important, as long as it is consistent for all documents in the dataset. Neither does bag-of-words represent any concept of word hierarchy.

* Sometimes it is also informative to look at feature vectors in data space. A feature vector contains the value of the feature in each data point. The axes denote individual data points, and the points denote feature vectors.
With bag-of-words featurization for text documents, a feature is a word, and a feature vector contains the counts of this word in each document. 
In this way, a word is represented as a “bag-of-documents.”  As we shall see in Chapter 4, these bag-of-documents vectors come from the matrix transpose of the bag-of-words vectors.

* Bag-of-words is not perfect. Breaking down a sentence into single words can destroy the semantic meaning.

## Bag-of-n-Grams
* An n-gram is a sequence of n tokens. A word is essentially a 1-gram, also known as a unigram. After tokenization, the counting mechanism can collate individual tokens into word counts, or count overlapping sequences as n-grams.

* n-grams retain more of the original sequence structure of the text, and therefore the bag-of-n-grams representation can be more informative. However, this comes at a cost. Theoretically, with k unique words, there could be k2 unique 2-grams (also called bigrams). 

* Bag-of-n-grams generates a lot more distinct n-grams. It increases the feature storage cost, as well as the computation cost of the model training and prediction stages. The number of data points remains the same, but the dimension of the feature space is now much larger. Hence, the data is much more sparse. The higher n is, the higher the storage and computation cost, and the sparser the data. For these reasons, longer n-grams do not always lead to improvements in model accuracy (or any other performance measure). People usually stop at n = 2 or 3. Longer n-grams are rarely used.


## Filtering for Cleaner Features

### Stopwords
Stopword lists are a way of weeding out common words that make for vacuous features.

### Frequency-Based Filtering

1. FREQUENT WORDS
Frequency statistics are great for filtering out corpus-specific common words as well as general-purpose stopwords.

1. RARE WORDS
Rare words incur a large computation and storage cost for not much additional gain.Rare words can be easily identified and trimmed based on word count statistics. Alternatively, their counts can be aggregated into a special garbage bin, which can serve as an additional feature.

1. Stemming
Stemming is an NLP task that tries to chop each word down to its basic linguistic word stem form.

## Atoms of Meaning: From Words to n-Grams to Phrases
One way to combat the increase in sparsity and cost in bag-of-n-grams is to filter the n-grams and retain only the most meaningful phrases. 

### Parsing and Tokenization
1. Parsing is necessary when the string contains more than plain text.
1. Tokenization turns the string—a sequence of characters—into a sequence of tokens. Each token can then be counted as a word. 

### Collocation Extraction for Phrase Detection

* In computational natural language processing (NLP), the concept of a useful phrase is called a collocation. In the words of Manning and Schütze (1999: 151), “A collocation is an expression consisting of two or more words that correspond to some conventional way of saying things.” 

* Collocations are more meaningful than the sum of their parts.  Not every collocation is an n-gram. Conversely, not every n-gram is deemed a meaningful collocation.

#### How to discover and extract collocations from text? 
1. FREQUENCY-BASED METHODS
A simple hack is to look at the most frequently occurring n-grams. 

1. HYPOTHESIS TESTING FOR COLLOCATION EXTRACTION
The key idea is to ask whether two words appear together more often than they would by chance. Hypothesis testing is a way to boil noisy data down to “yes” or “no” answers. It involves modeling the data as samples drawn from random distributions. 

### CHUNKING AND PART-OF-SPEECH TAGGING
* To generate longer phrases, there are other methods such as chunking or combining with part-of-speech (PoS) tagging.Chunking is a bit more sophisticated than finding n-grams, in that it forms sequences of tokens based on parts of speech, using rule-based models.

## TF-IDF (term frequency–inverse document frequency)
1. tf-idf makes rare words more prominent and effectively ignores common words. It is closely related to the frequency-based filtering methods. Tf-idf transforms word count features through multiplication with a constant. Hence, it is an example of feature scaling.
```
bow(w, d) = # times word w appears in document d
tf-idf(w, d) = bow(w, d) * log(N / (# documents in which word w appears))
N is the total number of documents in the dataset. 
```

## Impacts of Feature Scaling on Linear Models

1. For linear models like logistic regression, the features are used through the "data matrix". The columns represent all possible words in the vocabulary. The rows represent each document. It contains data points represented as fixed-length flat vectors. With bag-of-words vectors, the data matrix is also known as the "document-term matrix".

1. Feature scaling methods are essentially column operations on the data matrix. In particular, tf-idf and ℓ2 normalization both multiply the entire column (an n-gram feature, for example) by a constant.

1. The null space of the data matrix can be large for a couple of reasons. 
   * First, many datasets contain data points that are very similar to one another. This means the effective row space is small compared to the number of data points in the dataset. 
   * Second, the number of features can be much larger than the number of data points.
   * Moreover, the number of distinct words usually grows with the number of documents in the dataset, so adding more documents would not necessarily decrease the feature-to-data ratio or reduce the null space.

1. With bag-of-words (no feature scaling), the column space is relatively small compared to the number of features. There could be words that appear roughly the same number of times in the same documents. This would lead to the corresponding column vectors being nearly linearly dependent, which leads to the column space being not as full rank as it could be (see Appendix A for the definition of full rank). This is called a **rank deficiency**.

1. Rank-deficient row space and column space lead to the model being overly provisioned for the problem. The linear model outfits a weight parameter for each feature in the dataset. If the row and column spaces were full rank (Strictly speaking, the row space and column space for a rectangular matrix cannot both be full rank. The maximum rank for both subspaces is the smaller of m (the number of rows) and n (the number of columns).), then the model would allow us to generate any target vector in the output space. When they are rank deficient, the model has more degrees of freedom than it needs. This makes it harder to pin down a solution.

### Can feature scaling solve the rank deficiency problem of the data matrix? 
The column space is defined as the linear combination of all column vectors (boldface indicates a vector): ```a1v1 + a2v2 + ... + anvn```. Feature scaling replaces a column vector with a constant multiple, say v˜1=cv1. But we can still generate the original linear combination by just replacing a1 with a˜1=a1/c. It appears that feature scaling does not change the rank of the column space. Similarly, feature scaling does not affect the rank of the null space, because one can counteract the scaled feature column by reverse scaling the corresponding entry in the weight vector.

However, as usual, there is one catch. If the scalar is 0, then there is no way to recover the original linear combination; v1 is gone. If that vector is linearly independent from all the other columns, then we’ve effectively shrunk the column space and enlarged the null space.

If that vector is not correlated with the target output, then this is effectively pruning away noisy signals, which is a good thing. This turns out to be the key difference between tf-idf and ℓ2 normalization. ℓ2 normalization would never compute a norm of zero, unless the vector contains all zeros. If the vector is close to zero, then its norm is also close to zero. Dividing by the small norm would accentuate the vector and make it longer.

Tf-idf, on the other hand, can generate scaling factors that are close to zero, as shown in Figure 4-2. This happens when the word is present in a large number of documents in the training set. Such a word is likely not strongly correlated with the target vector. Pruning it away allows the solver to focus on the other directions in the column space and find better solutions (although the improvement in accuracy will probably not be huge, because there are typically few noisy directions that are prunable in this way).

Where feature scaling—both ℓ2 and tf-idf—does have a telling effect is on the convergence speed of the solver. This is a sign that the data matrix now has a much smaller condition number (the ratio between the largest and smallest singular values—see Appendix A for a full discussion of these terms). In fact, ℓ2 normalization makes the condition number nearly 1. But it’s not the case that the better the condition number, the better the solution. During this experiment, ℓ2 normalization converged much faster than either BoW or tf-idf. But it is also more sensitive to overfitting: it requires much more regularization and is more sensitive to the number of iterations during optimization.

Tf-idf and ℓ2 normalization do not improve the final classifier’s accuracy above plain bag-of-words. After acquiring some statistical modeling and linear algebra chops, we realize why: neither of them changes the column space of the data matrix.

One small difference between the two is that tf-idf can “stretch” the word count as well as “compress” it. In other words, it makes some counts bigger, and others close to zero. Therefore, tf-idf could altogether eliminate uninformative words.

Along the way, we also discovered another effect of feature scaling: it improves the condition number of the data matrix, making linear models much faster to train. Both ℓ2 normalization and tf-idf have this effect.

To summarize, the lesson is: the right feature scaling can be helpful for classification. The right scaling accentuates the informative words and downweights the common words. It can also improve the condition number of the data matrix. The right scaling is not necessarily uniform column scaling.





