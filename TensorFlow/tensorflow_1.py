'''
#tensorflow example--版本1.0

import tensorflow.compat.v1 as tf 
import numpy as np 
tf.compat.v1.disable_eager_execution()

#创建一些数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#开始创建TensorFlow结构
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)		#0.5是学习效率，小于1的数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()			#初始化变量
#解释创建TensorFlow结构

sess = tf.Session()
sess.run(init)			#激活init

for step in range(201):
	sess.run(train)
	if step % 20 ==0:
		print(step, sess.run(Weights), sess.run(biases))
'''


'''
#tensorflow example--版本2.1

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#creat tensorflow structure start
Weights = tf.Variable(tf.random.uniform((1,), -1.0, 1.0))
biases = tf.Variable(tf.zeros((1,)))

loss = lambda: tf.keras.losses.MSE(y_data, Weights*x_data + biases)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

for step in range(201):
	optimizer.minimize(loss, var_list=[Weights, biases])
	if step % 20 == 0:
		print("{} step, weights = {}, biases = {}".format(step, Weights.read_value(), biases.read_value()))
'''



'''
#session 会话控制 --版本1.0
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1, matrix2)

##method_1
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()

#method_2
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)
'''


'''
#Variable 变量 --版本1.0
import tensorflow.compat.v1 as tf  
tf.compat.v1.disable_eager_execution()

state = tf.Variable(0, name='counter')			#创建变量，赋值为0，名字为counter
#print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()		#创建变量， 必须要激活变量

with tf.Session() as sess:
	sess.run(init)
		for _ in range(3):
		sess.run(update)
		print(sess.run(state))
'''


'''
#Variable 变量 --版本2.1
import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)

for _ in range(3):
	state.assign_add(new_value)
	tf.print(state)
'''



'''
#Placeholder 传入值
import tensorflow.compat.v1 as tf  
tf.compat.v1.disable_eager_execution()

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
	print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))
'''



'''
#Example版本1.0 ---创建一个神经网
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf  
tf.compat.v1.disable_eager_execution()


#添加层 def add_layer()
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs
	
#创建神经网
x_data = np.linspace(-1, 1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
#计算预测值prediction和真实值的误差，对二者差的平方和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#画散点图
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

#mini bacth训练方法
for i in range(1000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i % 50:
		#print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass

		prediction_value = sess.run(prediction, feed_dict={xs:x_data})
		lines = ax.plot(x_data, prediction_value, 'r', lw=5)
		plt.pause(0.1)
'''



#Example版本2.1 ----创建一个神经网络
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, Weights, biases, activation_function=None):

	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs, Weights, biases

def loss(predicted_y, target_y):
	return tf.math.reduce_mean(tf.math.square(predicted_y - target_y))

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
x_data = x_data.astype(np.float32)
noise = noise.astype(np.float32)
y_data = y_data.astype(np.float32)

optimizer = tf.optimizers.SGD(learning_rate=0.1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

Wl = tf.Variable(tf.random.normal((1, 10)))
bl = tf.Variable(tf.zeros((1, 10)) + 0.1)
Wp = tf.Variable(tf.random.normal((10, 1)))
bp = tf.Variable(tf.zeros((1, 1)) + 0.1)

for i in range(1000):
	with tf.GradientTape() as tape:
		l1, Wl, bl = add_layer(x_data, Wl, bl, activation_function=tf.nn.relu)
		prediction, Wp, bp = add_layer(l1, Wp, bp, activation_function=None)
		loss_val = loss(prediction, y_data)
		print('loss: {}\n'.format(loss_val))


	grads = tape.gradient(loss_val, [Wl, bl, Wp, bp])
	optimizer.apply_gradients(zip(grads, [Wl, bl, Wp, bp]))
	print('Wl: {},\n bl{},\n Wp: {},\n bp: {}\n\n'.format(Wl.numpy(), bl.numpy(), Wp.numpy(), bp.numpy()))

	if i % 50 ==0:
		try:
			ax.lines.remove(lines[0])
		except Exception:
			lines = ax.plot(x_data, prediction.numpy(), 'r-', lw=5)
			plt.pause(0.1)





