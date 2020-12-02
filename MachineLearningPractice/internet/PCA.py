'''
Today in a meeting I was asked about Principal Component Analysis (PCA) and realized that it is important 
that I learn it. PCA is an unsupervised, non-parametric statistical technique primarily used
for dimensionality reduction in machine learning.

Below are some tutorials and papers I followed from on the internet and practiced how to apply PCA in python.
'''

# PCA is used to reduce the dimensionality of a dataset. This improves ML training as the training set is
# reduced and the learning rates are boosted and computation costs are reduced by removing redundant features.

# PCA can also be used to filter noisy datasets, such as image compression.

'''
Assumptions

PCA is based on the Pearson correlation coefficient framework and inherits similar assumptions.

1. Sample Size: Minimum of 150 observations and a 5:1 ratio of obervation to features.
2. Correlations: The feature set is correlated, so the reduced feature set effectively represents
                the original data space.
3. Linearity: All variables exhibit a constant multivariate normal relationship, and principal components are a
                linear combination of the original features.
4. Outliers: No significant outliers in the data as these can have a disproportionate influence on the results.
5. Large variance implies more structure: high variance axes are treated as principal components, while low variance axes 
                                            are treated as noise and discarded.

'''

'''
Applied PCA workflow

1. Normalize the Data:
    Normalize because unscaled data with different measurement units can
    distort the relative comparison of variance across features.

2. Create a covariance matrix for Eigen decomposition:
    Calculate the covariance among all the different dimensions
    and put them in a covariance matrix which represents the relationships
    in the data. Understanding the percent of variance is integral to
    reducing the feature set.

3. Select the optimal number of principal components:
    Determined by looking at the cumulative explained variant ratio as
    a function of the number of components.

'''

'''
PCA Limitations

Model performance:
    PCA can reduce model performance on datasets with no or low 
    feature correlation or does not meet the assumptions of linearity.

Classification accuracy:
    Valuable information pertaining to distinguishability between
    classes may be discarded if they are in the low variance range. 

Outliers:
    PCA is affected by outliers, and normalization must be an essential
    component of any workflow.

Interpretability:
    Principal components are combinations of original features and do not
    allow for the individual feature importance to be recognized.

'''

# Only found examples of PCA with sci-kit learn. This raises the questions of whether
# PCA can only be done with sci-kit? Can Sci-Kit work with PyTorch to hand over PCA data?

'''
References:

https://medium.com/apprentice-journal/pca-application-in-machine-learning-4827c07a61db

https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html

https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

'''