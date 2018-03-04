from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from MetaClassificator import CrossClassifier

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

cross_class = CrossClassifier([LogisticRegression(), KNeighborsClassifier(n_neighbors=11, weights="distance")])
clf = cross_class.fit(X_train, y_train)
y_ = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_)
precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_test, y_, average="macro")
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F-Beta score: " + str(fbeta_score))
