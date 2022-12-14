# KNN is used to clasify information 
import sklearn
from sklearn.utils import shuffle 
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd 
import numpy as np 
from sklearn import linear_model, preprocessing 

data = pd.read_csv('car.data')
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
class_ = le.fit_transform(list(data['class']))


predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(class_)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# setting K=5
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print(f"| Predicted: {names[predicted[x]]} | Actual: {names[y_test[x]]} | Data: {x_test[x]} |")



