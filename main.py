import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(len(x_train[0]), len(x_train[0][0]))))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics="accuracy")

model.fit(x_train,y_train,epochs=4)
model.save('digit_classify')
loss, accuracy = model.evaluate(x_test,y_test)
print(str(loss) + '\n' + str(accuracy))
model = tf.keras.models.load_model('digit_classify')
img_num = 0
num_right = 0
while os.path.isfile("Numbers/" + str(img_num) + ".png"):
    img = cv2.imread("Numbers/" + str(img_num) + ".png",cv2.IMREAD_GRAYSCALE)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    result = np.argmax(prediction)
    if result == img_num:
        num_right+=1

    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()
    img_num+=1
print(num_right/10)



