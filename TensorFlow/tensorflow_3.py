
#Classification 分类学习 --版本1.0
import tensorflow.compat.v1 as tf 
import numpy as np 
tf.compat.v1.disable_eager_execution()

from tensorflow.examples.tutorials.mnist import input_data

#准备数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b,)
	return outputs

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs})
	correct_prdiction =  tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prdiction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
	return result

#define placeholder for inputs to network 搭建神经网络
xs = tf.placeholder(tf.float32, [None, 784])		#28x28
ys = tf.placeholder(tf.float32, [None, 10])

#add output layer 添加输出层
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

#important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))


'''
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# load mnist dataset
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data(path='mnist.npz')
# show a picture of number
# number = train_x[0]
# plt.figure()
# plt.title('handwritten numberal')
# plt.imshow(number)
# plt.show()
# define parameters of neural network
ninput = 784  # input units, 28 * 28
nhidden1 = 256  # hidden units
nhidden2 = 128  # hidden units
noutput = 10  # output units
# construct a three-layers neural network, aka. MLP
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(nhidden1, activation='relu'),
    keras.layers.Dense(nhidden2, activation='relu'),
    keras.layers.Dense(noutput)
])
model.compile(optimizer=keras.optimizers.SGD(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10)
test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
print('loss', test_loss, '\naccuracy: ',test_acc)
'''

