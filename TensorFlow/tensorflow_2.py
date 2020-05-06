'''
#Example版本1.0 ---可视化神经网
import numpy as np
import tensorflow.compat.v1 as tf  
tf.compat.v1.disable_eager_execution()


#添加层 def add_layer()
def add_layer(inputs, in_size, out_size, activation_function=None):
	with tf.name_scope('layer'):
		with tf.name_scope('Weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
		with tf.name_scope('Wx_plus_b'):
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

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
#计算预测值prediction和真实值的误差，对二者差的平方和再取平均
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]), name='loss')
with tf.name_scope('layer'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
'''

'''
#Example版本2.1 ---可视化神经网
import tensorflow as tf
import numpy as np
from datetime import datetime

def add_layer(inputs, Weights, biases, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

def loss(predicted_y, target_y):
    return tf.math.reduce_mean(tf.math.square(predicted_y - target_y))


def train_one_step(optimizer, x_data, y_data):
    with tf.GradientTape() as tape:
        l1 = add_layer(x_data, Wl, bl, n_layer=1, activation_function=tf.nn.relu)
        prediction = add_layer(l1, Wp, bp, n_layer=2, activation_function=None)
        with tf.name_scope('loss'):
            loss_val = loss(prediction, y_data)
    with tf.name_scope('trains'):
        grads = tape.gradient(loss_val, [Wl, bl, Wp, bp])  # 这里只使用Wl,bl也能得到结果
        optimizer.apply_gradients(zip(grads, [Wl, bl, Wp, bp]))
    return loss_val

@tf.function
def train(optimizer, x_data, y_data):
    loss_val = train_one_step(optimizer, x_data, y_data)
    with summary_writer.as_default():
        tf.summary.scalar('loss', loss_val, step=0)
    return loss_val


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)


Wl = tf.Variable(tf.random.normal((1, 10)), name='Wl')
Wp = tf.Variable(tf.random.normal((10, 1)), name='Wp')
bl = tf.Variable(tf.zeros((1, 10)) + 0.1, name='bl')
bp = tf.Variable(tf.zeros((1, 1)) + 0.1, name='bp')

optimizer = tf.optimizers.SGD(learning_rate=0.1)  # 优化器

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/%s' % stamp
summary_writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on()
last_loss = train(optimizer, x_data, y_data)
with summary_writer.as_default():
    tf.summary.trace_export(
        name="logs_2/",
        step=0,
        profiler_outdir=logdir)
'''


#可视化example2-版本1.0
import numpy as np
import tensorflow.compat.v1 as tf  
tf.compat.v1.disable_eager_execution()


#添加层 def add_layer()
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/wieghts', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs', outputs)
        return outputs
    
#创建神经网
x_data = np.linspace(-1, 1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10,n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1,n_layer=2, activation_function=None)
#计算预测值prediction和真实值的误差，对二者差的平方和再取平均
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]), name='loss')
    tf.summary.scalar('loss', loss)
with tf.name_scope('layer'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)














