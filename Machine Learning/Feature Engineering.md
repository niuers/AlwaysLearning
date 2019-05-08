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
