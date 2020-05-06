
'''
#RNN 循环神经网络 ----回归分析
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()

BATCH_START = 0		#建立batch data 的时候index
TIME_STEPS = 20		#backpropagation through time 的 time_steps
BATCH_SIZE = 50		
INPUT_SIZE = 1		#sin 数据输入 size
OUTPUT_SIZE = 1		#cos 数据输出 size
CELL_SIZE = 10		#RNN 的 hidden unit size
LR = 0.006			#learning rate


def get_batch():
	global BATCH_START, TIME_STEPS
	#xs shape (50 batch, 20 steps)
	xs = np.arange(BATCH_START,BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
	seq = np.sin(xs)
	res = np.cos(xs)
	BATCH_START += TIME_STEPS
	#return seq, res and xs:shape(batch, step, input)
	return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
	def __init__ (self, n_steps, input_size, output_size, cell_size, batch_size):
		self.n_steps = n_steps
		self.input_size = input_size
		self.output_size = output_size
		self.cell_size = cell_size
		self.batch_size = batch_size
		with tf.name_scope('inputs'):
			self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
			self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
		with tf.variable_scope('in_hidden'):
			self.add_input_layer()
		with tf.variable_scope('LSTM_cell'):
			self.add_cell()
		with tf.variable_scope('out_hidden'):
			self.add_output_layer()
		with tf.name_scope('cost'):
			self.compute_cost()
		with tf.name_scope('train'):
			self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)


	def add_input_layer(self):
		l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')		#(batch*n_step, in_size)
		#Ws(in_size, cell_size)
		Ws_in = self._weight_variable([self.input_size, self.cell_size])
		#bs(cell_size, )
		bs_in = self._bias_variable([self.cell_size,])
		#l_in_y = (batch*n_steps, cell_size)
		with tf.name_scope('Wx_plus_b'):
			l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
		#reshape l_in_y ==> (batch, n_steps, cell_size)
		self.l_in_y = tf.reshape(l_in_y, [-1,self.n_steps,self.cell_size], name='2_3D')



	def add_cell(self):
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
		with tf.name_scope('initial_state'):
			self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
		self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)



	def add_output_layer(self):
		# shape = (batch * steps, cell_size)
		l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
		Ws_out = self._weight_variable([self.cell_size, self.output_size])
		bs_out = self._bias_variable([self.output_size, ])
		# shape = (batch * steps, output_size)
		with tf.name_scope('Wx_plus_b'):
			self.pred = tf.matmul(l_out_x, Ws_out) + bs_out



	def compute_cost(self):
		losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
		with tf.name_scope('average_cost'):
			self.cost = tf.div(
				tf.reduce_sum(losses, name='losses_sum'),
				self.batch_size,
				name='average_cost')
			tf.summary.scalar('cost', self.cost)



	@staticmethod
	def ms_error(labels, logits):
		return tf.square(tf.subtract(labels, logits))

	def _weight_variable(self, shape, name='weights'):
		initializer = tf.random_normal_initializer(mean=0., stddev=1., )
		return tf.get_variable(shape=shape, initializer=initializer, name=name)

	def _bias_variable(self, shape, name='biases'):
		initializer = tf.constant_initializer(0.1)
		return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
	# 搭建 LSTMRNN 模型
	model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
	sess = tf.Session()
	# sess.run(tf.initialize_all_variables()) # tf 马上就要废弃这种写法
	# 替换成下面的写法:
	sess.run(tf.global_variables_initializer())
    
	# 训练 200 次
	for i in range(200):
		seq, res, xs = get_batch()  # 提取 batch data
		if i == 0:
		# 初始化 data
			feed_dict={model.xs: seq, model.ys: res,}
		else:
			feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}		# 保持 state 的连续性

		# 训练
		_, cost, state, pred = sess.run(
			[model.train_op, model.cost, model.cell_final_state, model.pred],
			feed_dict=feed_dict)
        
		# 打印 cost 结果
		if i % 20 == 0:
			print('cost: ', round(cost, 4))

'''


import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
TIME_STEP = 10       # rnn time step
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps); y_np = np.cos(steps)    # float32 for converting torch FloatTensor
plt.plot(steps, y_np, 'r-', label='target (cos)'); plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best'); plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape(batch, 5, 1)
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])          # input y

# RNN
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)    # very first hidden state
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])          # reshape back to 3D

loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # initialize var in graph

plt.figure(1, figsize=(12, 5)); plt.ion()       # continuously plot

for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
    y = np.cos(steps)[np.newaxis, :, np.newaxis]
    if 'final_s_' not in globals():                 # first state, no any hidden state
        feed_dict = {tf_x: x, tf_y: y}
    else:                                           # has hidden state, so pass it to rnn
        feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}
    _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)     # train

    # plotting
    plt.plot(steps, y.flatten(), 'r-'); plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2)); plt.draw(); plt.pause(0.05)

plt.ioff(); plt.show()