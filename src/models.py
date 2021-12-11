from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def linReg(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(multi_class = "multinomial", solver="newton-cg")
    logreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
         .format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
         .format(logreg.score(X_test, y_test)))
    return logreg

def decisionTree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))
    return clf

def naiveBayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    print('Accuracy of GNB classifier on training set: {:.2f}'
         .format(gnb.score(X_train, y_train)))
    print('Accuracy of GNB classifier on test set: {:.2f}'
         .format(gnb.score(X_test, y_test)))
    return gnb

def kNearest(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
         .format(knn.score(X_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
         .format(knn.score(X_test, y_test)))
    return knn

def svm(X_train, X_test, y_train, y_test):
    svm = SVC()
    svm.fit(X_train, y_train)
    print('Accuracy of SVM classifier on training set: {:.2f}'
         .format(svm.score(X_train, y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
         .format(svm.score(X_test, y_test)))
    return svm

def lda(X_train, X_test, y_train, y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    print('Accuracy of LDA classifier on training set: {:.2f}'
         .format(lda.score(X_train, y_train)))
    print('Accuracy of LDA classifier on test set: {:.2f}'
         .format(lda.score(X_test, y_test)))
    return lda