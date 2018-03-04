"""
==================================
Machine learning module for Python
==================================

This module tries to find the best learning algorithm
for a given dataset in order to solve classification
problems in the best way possible.

In pairs, the algorithms in the 'clf_list' are compared:
- First, the 5-fold cross validation is performed on both algorithms of the pair;
- The t-student test is performed on the basis of the scores obtained;
- The best of the couple is chosen based on the applied statistics.

Note : choose your learning algorithms carefully because some of them could take a lot of time on
    certain types of datasets (e.g. SVC on breast_cancer dataset).
"""
# Author: Giuseppe Ruggiero <giuseppe.ruggiero95@yahoo.it>

import warnings

from scipy import stats
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import *
from sklearn.neighbors import *

warnings.filterwarnings("ignore")

class ListEmptyException(Exception):
    pass

class ListLenException(Exception):
    pass

class InvalidListItems(Exception):
    pass

class InvalidParameter(Exception):
    pass

class CrossClassifier:
    """
    Given a list of learning algorithms for solving classification
    problems, performs comparisons on pairs of them through
    the t-student test; only the best of the couple goes
    on until the best of all algorithms is found.

    Parameters
    ----------
    clf_list : array, default None
        List of learning algorithms.
        If it is None, a default list of 6 algorithms is used:
            - GaussianNB()
            - LogisticRegression()
            - SGDClassifier()
            - AdaBoostClassifier()
            - RandomForestClassifier()
            - KNeighborsClassifier(n_neighbors=11, weights="distance")

        See http://scikit-learn.org/stable/supervised_learning.html#supervised-learning for complete documentation.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from MetaClassificator import CrossClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> data = load_digits()
    >>> X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    >>> cross_class = CrossClassifier([LogisticRegression(), KNeighborsClassifier(n_neighbors=11, weights="distance")])
    >>> clf = cross_class.fit(X_train, y_train)
    >>> y_ = clf.predict(X_test)
    >>> print("Accuracy %s: " %accuracy_score(y_test, y_))
    Start comparison between LogisticRegression and KNeighborsClassifier
    Start t-student test
    p-value: 0.008807766386619509 < 0.05 = True
    Null hypothesis rejected, the difference between two algorithms is found significant.

    Best model: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=11, p=2,
           weights='distance')

    Accuracy: 0.987037037037037
    """

    def __init__(self, clf_list=None):
        if clf_list is None:
            clfs = [GaussianNB(), LogisticRegression(), SGDClassifier(), AdaBoostClassifier(), RandomForestClassifier(),
                    KNeighborsClassifier(n_neighbors=11, weights="distance")]
            self.clf_list = clfs
        elif type(clf_list) is not list:
            raise InvalidParameter("clf_list must be a list")
        elif len(clf_list) == 1:
            raise ListLenException("Inconsistent list size. List must contain at least two elements")
        elif clf_list == []:
            raise ListEmptyException("List can not be empty")
        else:
            valid_list = self.__check_fine_list(clf_list)
            if (valid_list):
                self.clf_list = clf_list
            else:
                raise InvalidListItems("The elements of the list must be learning algorithms "
                                       "(linear_model, svm, naive_bayes, neighbors, ensemble, tree, gaussian_process, "
                                       "neural_network, discriminant_analysis)")

    def fit(self, X, y):
        """Find the best learning algorithm and fit it according to X, y.

        Note : X must already be processed and ready for use.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        clf : object
            Returns the best learning algorithm fitted.
        """
        clf = self.__test_model(X, y)
        clf.fit(X, y)
        return clf

    @staticmethod
    def __check_fine_list(clf_list):
        """Private method.

        __init__ support method to check if the list passed by parameter is correct.

        If the list is correct, true is returned. Otherwise, false is
        returned and an exception is raised.

        Parameters
        ----------
        clf_list : array
            List of learning algorithms.

        Returns
        -------
        valid_list : boolean
            Returns true if the list is correct, false otherwise.
        """
        i = 0
        valid_list = True
        while i < len(clf_list) and valid_list:
            if ("sklearn.linear_model" not in str(type(clf_list[i])) and "sklearn.svm" not in str(type(clf_list[i])) and
                    "sklearn.naive_bayes" not in str(type(clf_list[i])) and "sklearn.neighbors" not in str(
                        type(clf_list[i])) and
                    "sklearn.ensemble" not in str(type(clf_list[i])) and "sklearn.tree" not in str(
                        type(clf_list[i])) and
                    "sklearn.neural_network" not in str(
                        type(clf_list[i])) and "sklearn.discriminant_analysis" not in str(type(clf_list[i])) and
                    "sklearn.gaussian_process" not in str(type(clf_list[i]))):
                valid_list = False
            i += 1
        return valid_list

    def __test_model(self, X, y):
        """Private method.

        fit support method to find the best learning algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        best_clf : object
            Returns the best learning algorithm found.
        """
        best_model = 0
        for i in range(1, len(self.clf_list)):
            clf1_name = str(self.clf_list[best_model]).split("(")[0]
            clf2_name = str(self.clf_list[i]).split("(")[0]
            print("Start comparison between %s and %s" % (clf1_name, clf2_name))
            change_best_model = self.__compute_best_model(X, y, self.clf_list[best_model], self.clf_list[i])
            if change_best_model: best_model = i
            print("Best temporary model: %s\n" % str(self.clf_list[best_model]).split("(")[0])
        print("Best model: %s\n" % self.clf_list[best_model])
        return self.clf_list[best_model]

    def __compute_best_model(self, X, y, clf1, clf2):
        """Private method.

        __test_model support method to compute the best learning algorithm.

        This method performs the 5-fold cross-validation on the pair of algorithms
        "clf1" and "clf2" passed as a parameter and then call the __apply_test method
        to perform the t-test. If the learning algorithm is considered suitable for
        data but, for some reason, does not work well on them, it is excluded
        directly from comparison.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        clf1 : object
            First learning algorithm.

        clf2 : object
            Second learning algorithm.

        Returns
        -------
        change : boolean
            Returns true if the best learning algorithm must be changed.
        """
        change = False
        score1 = self.__cross_validation(clf1, X, y)
        if score1 != []:
            score2 = self.__cross_validation(clf2, X, y)
            if score2 != []:
                statistic = self.__apply_test(score1, score2)
                if statistic < 0: change = True
        else:
            change = True
        return change

    def __cross_validation(self, clf, X, y):
        """Private method.

        __compute_best_model support method to perform the 5-fold cross-validation.

        Parameters
        ----------
        clf : object
            Learning algorithm on which to perform the cross validation.

        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        score : array
            Returns:
                - the cross validation score, if everything has been successful;
                - empty list, if the learning algorithm, for some reason,
                    can not work well on the data and therefore must be excluded directly from comparison;
                -raise an exception, if any errors occurred (e.g. the dataset is not suitable for classification).
        """
        try:
            score = cross_val_score(clf, X, y, cv=5)
        except TypeError as e:
            if "A sparse matrix was passed, but dense data is required" in str(e):
                score = []
            else:
                raise
        return score

    def __apply_test(self, scores1, scores2):
        """Private method.

        __compute_best_model support method to perform the t-test. The parameter
        "equal_var" of stats.ttest_ind() is set to false because we don't want to
        assume equal population variance. The p-value is used to accept or reject
        the null hypothesis that the two algorithms have the same performance.

        See https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_ind.html for complete documentation.

        Parameters
        ----------
        scores1 : array
            Scores obtained by the first algorithm from the 5-fold cross-validation.

        scores2 : array
            Scores obtained by the second algorithm from the 5-fold cross-validation.

        Returns
        -------
        statistic : float
            Returns the test statics obtained by applying the t-student test on the two scores.
        """
        print("Start t-student test")
        statistic, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
        print("p-value: %s < 0.05 = %s" % (p_value, str(p_value < 0.05)))
        print("Null hypothesis{0}rejected, the difference between two algorithms is{0}found significant.".format(
            " " if p_value < 0.05 else " not "))
        return statistic
