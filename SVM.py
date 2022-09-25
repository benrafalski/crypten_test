import sklearn
from sklearn import datasets 
from sklearn import svm 
from sklearn import metrics 
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()


x = cancer.data 
y = cancer.target 

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# params, kernel=linear, C(softmargin) = 1 seems best
clf = svm.SVC(kernel='linear', C=1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)









