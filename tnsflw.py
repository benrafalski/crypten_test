import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib as plt 


print("\n\n\n\n")
data = keras.datasets.fashion_mnist 

class_names = ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# shrinking down data to be easier to use  
train_images = train_images/255.0 
test_images = test_images/255.0 

# input -> dense -> dense_output
# building a 'sequence' of layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model 
model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)

print(class_names[test_labels[0]])

print(class_names[np.argmax(prediction[0])])

    
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Accuracy: {test_acc}')








