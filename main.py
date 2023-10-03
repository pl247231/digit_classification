import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense

class Model(tf.keras.Model):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.layer0 = tf.keras.layers.Flatten(input_shape = (28*28,))
        self.layer1 = tf.keras.layers.Dense(512,activation='relu')
        self.layer2 = tf.keras.layers.Dense(10,activation='softmax')

    def call(self, input: np.array) -> tf.Tensor:
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        return input

def one_hot_encoding(input: np.array) -> np.array:
    unique = set(input)
    num_unique = len(set(input))
    #Create a dict to map input category to one hot index
    num_to_pos = {}
    index = 0
    for value in unique:
      num_to_pos[value] = index
      index+=1
    one_hot = []
    #One hot encode each input value
    for value in input:
      arr = np.array([0]*len(unique))
      arr[num_to_pos[value]] = 1
      one_hot.append(arr)
    return np.array(one_hot)


data = mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


loss_func = 'categorical_crossentropy'

optimizer = 'adam'
model = Model()
model.compile(optimizer = optimizer, loss = loss_func, metrics = ["accuracy"])

tf_train_labels = to_categorical(train_labels)
tf_test_labels = to_categorical(test_labels)
model.fit(train_images, tf_train_labels, epochs = 5, batch_size = 128)
test_loss, test_acc = model.evaluate(test_images, tf_test_labels)
print('testAccuracy', test_acc)
