from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

def output_accuracy(name, predictions):
	print name, "- Mislabeled Points: %d out of %d" % \
	(sum(iris.target != predictions), len(iris.data))

	print name,"- Accuracy Score", \
	accuracy_score(iris.target, predictions)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

gaussian_pred = clf.fit(iris.data, iris.target).predict(iris.data)

output_accuracy("GaussianNB", gaussian_pred)


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(iris.data, iris.target)

multinomial_pred = clf.predict(iris.data)

output_accuracy("MultinomialNB", multinomial_pred)



