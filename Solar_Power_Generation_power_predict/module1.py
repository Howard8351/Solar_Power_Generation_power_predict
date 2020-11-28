from tensorflow.data  import Dataset
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
import tensorflow as tf
import time
import random


batch_size = 4

# -- Data Setup -- #
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
x_train, x_test = x_train / 255.0, x_test / 255.0
# Create two inputs and two outputs (for demonstration)
x_train1 = x_train2 = tf.convert_to_tensor(x_train)
y_train1 = y_train2 = tf.convert_to_tensor(y_train)

data = tf.concat([y_train1, y_train1, y_train1], 1)
a = tf.shape(data).numpy()

# -- Dataset API -- #
# Create a Dataset for multiple inputs and Dataset for multiple outputs
input_set = tf.data.Dataset.from_tensor_slices(x_train1)
#output_set = tf.data.Dataset.from_tensor_slices((y_train1, y_train2))
output_set = tf.data.Dataset.from_tensor_slices(data)
# Create Dataset pipeline
input_set = input_set.batch(batch_size).repeat()
#output_set = output_set.batch(batch_size).repeat()
def tensor_split(data):
    a = data[0:10]
    b = data[10]
    c = data[20]
    return (a, (b, c))
output_set = output_set.map(tensor_split, num_parallel_calls = tf.data.experimental.AUTOTUNE)
output_set = output_set.batch(batch_size).repeat()

test = iter(output_set)
a = test.get_next()
#b = tensor_split(a)

# Group the input and output dataset
dataset = tf.data.Dataset.zip((input_set, output_set))
#test = iter(dataset)
#a = test.get_next()
# Initialize the iterator to be passed to the model.fit() function
#data_iter = dataset.make_one_shot_iterator()

# -- Model Definition -- #
# Multiple Inputs
input1 = tf.keras.layers.Input(shape=(10))
# Input 1 Pathway
x1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)((input1))
x1 = tf.keras.layers.Dropout(0.2)(x1)
# Multiple Outputs
output1 = tf.keras.layers.Dense(1)(x1)
output2 = tf.keras.layers.Dense(1)(x1)
# Create Model
model = tf.keras.models.Model(inputs=input1, outputs=[output1, output2])

model.summary()
b = model(a[0])
c = a[1]
# Compile
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# -- Train -- #
model.fit(output_set, steps_per_epoch=len(x_train)//batch_size, epochs=5)
