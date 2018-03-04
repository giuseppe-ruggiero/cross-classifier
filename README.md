# CrossClassifier
[![Build Status](https://travis-ci.org/mikel/mail.svg?branch=master)](https://travis-ci.org/mikel/mail)

# Contents
* [Introduction](#introduction)
* [Simple Demo](#simple-demo)
* [Documentation](#documentation)
* [Note](#note)
* [Contacts](#contacts)

## Introduction
This module tries to find the best learning algorithm for a given dataset in order to solve classification problems in the best way possible. 

CrossClassifier takes a list, called clf_list, of learning algorithms and, in pairs, compared it such as:
- First, the 5-fold cross validation is performed on both algorithms of the pair;
- The t-student test is performed on the basis of the scores obtained;
- The best of the couple is chosen based on the applied statistics.

By default, the list of learning algorithms consists of 6 algorithms:
- GaussianNB()
- LogisticRegression()
- SGDClassifier()
- AdaBoostClassifier()
- RandomForestClassifier()
- KNeighborsClassifier(n_neighbors=11, weights="distance")

However, the user can also pass a new list of algorithms if he does not want to use the default one.

## Simple Demo

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from MetaClassificator import CrossClassifier
from sklearn.metrics import accuracy_score

# First load the dataset:
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)

# Then create an instance of CrossClassifier (in this case, the default list described above is used)
cross_class = CrossClassifier()

# Then perform your tests through the fit method
best_clf = cross_class.fit(X_train, y_train)

# Apply predictions using the best classifier and print accuracy
y_ = clf.predict(X_test)
print("Accuracy %s: " %accuracy_score(y_test, y_))
```
## Documentation
More information can be found on the [Scikit-learn supervised learning](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning) and on the [T-student test.](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_ind.html)

## Note
Choose your learning algorithms carefully because some of them could take a lot of time on certain types of datasets (e.g. SVC on breast_cancer dataset). 

The training set passed into the fit method must already be processed and ready for use.

## Contacts
For more information or for any problem please do not hesitate to contact me at giuseppe.ruggiero95@yahoo.it
