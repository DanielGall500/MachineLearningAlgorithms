import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target
names = iris.target_names


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X)

tran_x = pca.transform(X)

print "PCA Precision:", pca.get_precision() 
print "PCA Explained Variance Ratio:", pca.explained_variance_ratio_
print "PCA Score:", pca.score(X)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)

lda.fit(X, Y)

tran_x = lda.transform(X)

print "LDA Slope:", lda.coef_
print "LDA Intercept:", lda.intercept_






