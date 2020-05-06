'''
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()

ACTIVATION = tf.nn.relu		#每一层都使用relu
N_LAYERS = 7				#一共7层隐藏层
N_HIDDEN_UNITS = 30			#每个隐藏层有30个神经元

#重复观看
def fix_seed(seed=1):
	#reporducible
	np.random.seed(seed)
	tf.set_random_seed(seed)


#打印图片
def plot_his(inputs, inputs_norm):
	#plot histogram for the inputs of every layer
	for j, all_inputs in enumerate([inputs, inputs_norm]):
		for i, input in enumerate(all_inputs):
			plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
			plt.cla()
			if i == 0:
				the_range = (-7,10)
			else:
				the_range = (-1,1)
			plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
			plt.yticks(())
			if j == 1:
				plt.xticks(the_range)
			else:
				plt.xticks(())
			ax = plt.gca()
			ax.spines['right'].set_color('none')
			ax.spines['top'].set_color('none')
		plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
	plt.draw()
	plt.pause(0.01)


#建立神经网络
def built_net(xs, ys, norm):
	def add_layer(inputs, in_size, out_size, activation_function=None, norm=False):
		Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
		biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		Wx_plus_b = tf.matmul(inputs, Weights) + biases
		
		if norm:
			fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0])
			scale = tf.Variable(tf.ones([out_size]))
			shift = tf.Variable(tf.zeros([out_size]))
			epsilon = 0.001

			ema = tf.train.ExponentialMovingAverage(decay=0.5)
			def mean_var_with_update():
				ema_apply_op = ema.apply([fc_mean, fc_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(fc_mean), tf.identity(fc_var)
			mean, var = mean_var_with_update()


			Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)



		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)

		return outputs

	fix_seed(1)

	if norm:
		# BN for the first input
		fc_mean, fc_var = tf.nn.moments(xs, axes=[0],)
		scale = tf.Variable(tf.ones([1]))
		shift = tf.Variable(tf.zeros([1]))
		epsilon = 0.001
		# apply moving average for mean and var when train on batch
		ema = tf.train.ExponentialMovingAverage(decay=0.5)
		def mean_var_with_update():
			ema_apply_op = ema.apply([fc_mean, fc_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(fc_mean), tf.identity(fc_var)
		mean, var = mean_var_with_update()
		xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)


	#recorde inputs for every layer 
	layers_inputs = [xs]

	#build hidden layers
	for l_n in range(N_LAYERS):
		layer_input = layers_inputs[l_n]
		in_size = layers_inputs[l_n].get_shape()[1].value

		output = add_layer(layer_input, in_size, N_HIDDEN_UNITS, ACTIVATION, norm,)
		layers_inputs.append(output)


	prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

	cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
	train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
	return [train_op, cost, layers_inputs]


#make up data
fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise


#plot input data
plt.scatter(x_data, y_data)
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])		#[num_samples, num_features]
ys = tf.placeholder(tf.float32, [None, 1])

train_op, cost, layers_inputs = built_net(xs, ys, norm=False)					#without BN
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True)		#with BN

sess = tf.Session()
init = tf.global_variables_initializer
sess.run(init)


#record cost
cost_his = []
cost_his_norm = []
record_step = 5

plt.ion()
plt.figure(figsize=(7, 3))
for i in range(250):
	if i % 50 == 0:
		all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x_data, ys: y_data})
		plot_his(all_inputs, all_inputs_norm)

	# train on batch
	sess.run([train_op, train_op_norm], feed_dict={xs: x_data[i*10:i*10+10], ys: y_data[i*10:i*10+10]})

	if i % record_step == 0:
		#record cost
		cost_his.append(sess.run(cost, feed_dict={xs:x_data, ys:y_data}))
		cost_his_norm.append(sess.run(cost_norm, feed_dict={xs:x_data, ys:y_data}))



plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his), label='no BN')     # no norm
plt.plot(np.arange(len(cost_his))*record_step, np.array(cost_his_norm), label='BN')   # norm
plt.legend()
plt.show()
'''


import tensorflow.compat.v1 as tf 
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper parameters
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = tf.nn.tanh
B_INIT = tf.constant_initializer(-0.2)      # use a bad bias initialization

# training data
x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
np.random.shuffle(x)
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise
train_data = np.hstack((x, y))

# test data
test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + noise

# plot input data
plt.scatter(x, y, c='#FF9359', s=50, alpha=0.5, label='train')
plt.legend(loc='upper left')

# tensorflow placeholder
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_train = tf.placeholder(tf.bool, None)     # flag for using BN on training or testing


class NN(object):
    def __init__(self, batch_normalization=False):
        self.is_bn = batch_normalization

        self.w_init = tf.random_normal_initializer(0., .1)  # weights initialization
        self.pre_activation = [tf_x]
        if self.is_bn:
            self.layer_input = [tf.layers.batch_normalization(tf_x, training=tf_is_train)]  # for input data
        else:
            self.layer_input = [tf_x]
        for i in range(N_HIDDEN):  # adding hidden layers
            self.layer_input.append(self.add_layer(self.layer_input[-1], 10, ac=ACTIVATION))
        self.out = tf.layers.dense(self.layer_input[-1], 1, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.loss = tf.losses.mean_squared_error(tf_y, self.out)

        # !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
        # pass the update_ops with control_dependencies to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def add_layer(self, x, out_size, ac=None):
        x = tf.layers.dense(x, out_size, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.pre_activation.append(x)
        # the momentum plays important rule. the default 0.99 is too high in this case!
        if self.is_bn: x = tf.layers.batch_normalization(x, momentum=0.4, training=tf_is_train)    # when have BN
        out = x if ac is None else ac(x)
        return out

nets = [NN(batch_normalization=False), NN(batch_normalization=True)]    # two nets, with and without BN

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# plot layer input distribution
f, axs = plt.subplots(4, N_HIDDEN+1, figsize=(10, 5))
plt.ion()   # something about plotting

def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax,  ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0: p_range = (-7, 10); the_range = (-7, 10)
        else: p_range = (-4, 4); the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(()); a.set_xticks(())
        ax_pa_bn.set_xticks(p_range); ax_bn.set_xticks(the_range); axs[2, 0].set_ylabel('Act'); axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)

losses = [[], []]   # record test loss
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    np.random.shuffle(train_data)
    step = 0
    in_epoch = True
    while in_epoch:
        b_s, b_f = (step*BATCH_SIZE) % len(train_data), ((step+1)*BATCH_SIZE) % len(train_data) # batch index
        step += 1
        if b_f < b_s:
            b_f = len(train_data)
            in_epoch = False
        b_x, b_y = train_data[b_s: b_f, 0:1], train_data[b_s: b_f, 1:2]         # batch training data
        sess.run([nets[0].train, nets[1].train], {tf_x: b_x, tf_y: b_y, tf_is_train: True})     # train

        if step == 1:
            l0, l1, l_in, l_in_bn, pa, pa_bn = sess.run(
                [nets[0].loss, nets[1].loss, nets[0].layer_input, nets[1].layer_input,
                 nets[0].pre_activation, nets[1].pre_activation],
                {tf_x: test_x, tf_y: test_y, tf_is_train: False})
            [loss.append(l) for loss, l in zip(losses, [l0, l1])]   # recode test loss
            plot_histogram(l_in, l_in_bn, pa, pa_bn)     # plot histogram

plt.ioff()

# plot test loss
plt.figure(2)
plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
plt.ylabel('test loss'); plt.ylim((0, 2000)); plt.legend(loc='best')

# plot prediction line
pred, pred_bn = sess.run([nets[0].out, nets[1].out], {tf_x: test_x, tf_is_train: False})
plt.figure(3)
plt.plot(test_x, pred, c='#FF9359', lw=4, label='Original')
plt.plot(test_x, pred_bn, c='#74BCFF', lw=4, label='Batch Normalization')
plt.scatter(x[:200], y[:200], c='r', s=50, alpha=0.2, label='train')
plt.legend(loc='best'); plt.show()