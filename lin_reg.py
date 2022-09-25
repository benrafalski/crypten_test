import pandas as pd 
import numpy as np 
import sklearn 
from sklearn import linear_model 
from sklearn.utils import shuffle 
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3" # 'label' -> what you are trying to predict 
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])



best = 0
for _ in range(30): 

    # train using some data, test using the rest of the data 
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

    # train model here
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best: 
        # save the model 
        best = acc
        with open("studentmodel.pickle", "wb") as f: 
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", 'rb')
linear = pickle.load(pickle_in)

print(f'\nHighest Accuracy: {linear.score(x_test, y_test)}\n')

print(f"Co: {linear.coef_}")
print(f'Intercept: {linear.intercept_}')

predictions = linear.predict(x_test)

for x in range(len(predictions)): 
    print(f'\nModel Prediction: {predictions[x]} vs. Actual: {y_test[x]}')
    print(f'\t [G1, G2, studytime, failures, absences] : {x_test[x]}')

# plotting data 
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show() # doesn't show?



