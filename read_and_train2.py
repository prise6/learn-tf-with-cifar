import tensorflow as tf
from tensorflow.contrib.data import Iterator
from PIL import Image
import numpy as np


def parser(record):

	keys_to_features = {
		"id": tf.FixedLenFeature((), tf.int64, default_value = tf.zeros([], dtype = tf.int64)),
		"label": tf.FixedLenFeature((), tf.int64, default_value = tf.zeros([], dtype = tf.int64)),
		"input": tf.FixedLenFeature((), tf.string)
	}

	parsed = tf.parse_single_example(record, keys_to_features)

	image = tf.decode_raw(parsed["input"], tf.uint8)
	image = tf.cast(image, tf.float32) / 255.0
	label = tf.cast(parsed["label"], tf.float32)
	label = label - 1.0

	img_id = tf.cast(parsed["id"], tf.int32)

	return image, label, img_id


def training_data():
	filenames = ['../Datas/train_part_1.tfrecords', '../Datas/train_part_2.tfrecords',
	'../Datas/train_part_3.tfrecords', '../Datas/train_part_4.tfrecords',
	'../Datas/train_part_5.tfrecords', '../Datas/train_part_6.tfrecords',
	'../Datas/train_part_7.tfrecords', '../Datas/train_part_8.tfrecords']
	dataset = tf.contrib.data.TFRecordDataset(filenames)
	dataset = dataset.map(parser)
	# dataset = dataset.shuffle(buffer_size = 1000)
	dataset = dataset.repeat()
	dataset = dataset.batch(128)
	

	return dataset

def testing_data():
	filenames = ['../Datas/train_part_9.tfrecords', '../Datas/train_part_10.tfrecords']
	dataset = tf.contrib.data.TFRecordDataset(filenames)
	dataset = dataset.map(parser)
	dataset = dataset.repeat()
	dataset = dataset.batch(128)

	return dataset

def myFilter(x, y, z):

	return True

training_dataset = training_data()
testing_dataset = testing_data()

training_iterator = training_dataset.make_one_shot_iterator()
next_train_batch = training_iterator.get_next()

testing_iterator = testing_dataset.make_one_shot_iterator()
next_test_batch = testing_iterator.get_next()



x = tf.placeholder(tf.float32, [None, 32*32*3], name = "x")
x_image = tf.reshape(x, [-1, 32, 32, 3])

tf.summary.image('input', x_image, 5)

## convolution 1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev = 5e-2))
b_conv1 = tf.Variable(tf.constant(0.1, shape = [64]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv1)

norm1 = tf.nn.lrn(h_conv1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
pool1 = tf.nn.max_pool(norm1, ksize = [1, 3, 3, 1], strides  = [1, 2, 2, 1], padding = 'SAME', name = 'pool1')


## convolution 2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev = 5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv2)

norm2 = tf.nn.lrn(h_conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool2')


## convolution 3
W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev = 5e-2))
b_conv3 = tf.Variable(tf.constant(0.1, shape = [128]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(pool2, W_conv3, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv3)


## convolution 4
W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev = 5e-2))
b_conv4 = tf.Variable(tf.constant(0.1, shape = [128]))
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv4)


## convolution 5
W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev = 5e-2))
b_conv5 = tf.Variable(tf.constant(0.1, shape = [128]))
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv5)

norm5 = tf.nn.lrn(h_conv5, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75, name = 'norm5')
pool5 = tf.nn.max_pool(norm5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'pool5')


## fully connected 1
h_conv5_flat = tf.reshape(pool5, [-1, 4 * 4 * 128])
W_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * 128, 384], stddev = 0.04))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [384]))
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)


## fully connected 2
W_fc2 = tf.Variable(tf.truncated_normal([384, 192], stddev = 0.04))
b_fc2 = tf.Variable(tf.constant(0.1, shape = [192]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)


## output
W_o = tf.Variable(tf.truncated_normal([192, 10], stddev = 1 / 192.0))
b_o = tf.Variable(tf.constant(0.0, shape = [10]))
h_o = tf.matmul(h_fc2, W_o) + b_o


y_ = tf.placeholder(tf.int64, [128])

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = h_o))

correct_prediction = tf.equal(tf.argmax(h_o, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train_step = tf.train.GradientDescentOptimizer(.01).minimize(cross_entropy)
train_step = tf.train.RMSPropOptimizer(learning_rate =  1e-3).minimize(cross_entropy)


merged = tf.summary.merge_all()

with tf.Session() as sess:
	writer = tf.summary.FileWriter("/tmp/cifar/1")
	writer.add_graph(sess.graph)
	sess.run(tf.global_variables_initializer())
	

	for i in range(10000):
		x_image, label, id = sess.run(next_train_batch)
		sess.run(train_step, feed_dict = {x: x_image, y_: label})

		if (i % 10 == 0) :
			train_accuracy = accuracy.eval(feed_dict = {x: x_image, y_: label})
			print("Step %d, training accuracy %g" % (i, train_accuracy))

		if(i % 50 == 0) :

			summary = merged.eval(feed_dict = {x: x_image, y_: label})
			writer.add_summary(summary, i)

			x_image, label, id = sess.run(next_test_batch)
			test_accuracy = accuracy.eval(feed_dict = {x: x_image, y_: label})
			print("Step %d, testing accuracy %g" % (i, test_accuracy))




