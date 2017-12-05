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
	# parsed = tf.parse_example(record, keys_to_features)

	image = tf.decode_raw(parsed["input"], tf.uint8)
	# image = tf.image.decode_image(image)
	# image = tf.reshape(image, [-1, 32, 32, 3])
	# label = tf.one_hot(tf.cast(parsed["label"], tf.int32), depth = 10, axis = 0, on_value = 1.0, off_value = 0.0)
	# label = tf.one_hot([1], depth = 10)
	# label = tf.reshape(tf.cast(parsed["label"], tf.int32), [-1, 1])
	label = tf.cast(parsed["label"], tf.int64)
	label = label - 1

	img_id = tf.cast(parsed["id"], tf.int32)

	return image, label, img_id


def training_data():
	filenames = ['../Datas/train_part_3.tfrecords', '../Datas/train_part_4.tfrecords']
	dataset = tf.contrib.data.TFRecordDataset(filenames)

	dataset = dataset.map(parser)
	# dataset = dataset.filter(myFilter)
	# dataset = dataset.filter(lambda a, b, c: tf.bool(tf.float32(1)))
	
	# print(dataset.output_shapes)
	dataset = dataset.shuffle(buffer_size = 1000)
	dataset = dataset.batch(50)
	dataset = dataset.repeat()

	# iterator = dataset.make_one_shot_iterator()
	# next_element = iterator.get_next()
	# return next_element

	return dataset

def testing_data():
	filenames = ['../Datas/train_part_10.tfrecords']
	# filenames = ['../Datas/train_part_1.tfrecords']
	dataset = tf.contrib.data.TFRecordDataset(filenames)

	dataset = dataset.map(parser)
	# dataset = dataset.filter(myFilter)
	dataset = dataset.batch(50)

	return dataset

def myFilter(x, y, z):
	# return tf.bool(tf.equal(tf.argmax(y, 1), 1))
	# print(y)
	# tmp = ((y[1]>0) | (y[4]>0))
	# tmp = (y[1]>0)
	return True

training_dataset = training_data()
testing_dataset = testing_data()


iterator = Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
testing_init_op = iterator.make_initializer(testing_dataset)


x = tf.placeholder(tf.float32, [None, 32*32*3], name = "x")
x_image = tf.reshape(x, [-1, 32, 32, 3])

x_image = tf.image.per_image_standardization(x_image)

tf.summary.image('input', x_image, 5)


W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev = 0))
b_conv1 = tf.Variable(tf.constant(0.1, shape = [64]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
norm1 = tf.nn.lrn(h_pool1, 4, bias = 1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev = 0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(norm1, W_conv2, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv2)
# h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
norm2 = tf.nn.lrn(h_conv2, 4, bias = 1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
h_pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


# # W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 30, 90], stddev = 0.1))
# # b_conv3 = tf.Variable(tf.constant(0.1, shape = [90]))
# # h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv3)
# # h_pool3 = tf.nn.max_pool(h_conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 384], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [384]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = tf.Variable(tf.truncated_normal([384, 192], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape = [192]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)


W_fc3 = tf.Variable(tf.truncated_normal([192, 10], stddev=0.1))
b_fc3 = tf.Variable(tf.constant(0.1, shape = [10]))
h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3


y_conv = tf.nn.softmax(h_fc3)

# y_ = tf.placeholder(tf.float32, [None, 10], name = "labels")
y_ = tf.placeholder(tf.int64,  [50], name = "labels")

# tmp = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
# tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
# tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = h_fc3)
# cross_entropy = tf.reduce_mean(tmp)

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))


train_step = tf.train.AdamOptimizer(.01).minimize(cross_entropy)

# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.summary.scalar('test', cross_entropy)
merged = tf.summary.merge_all()

with tf.Session() as sess:
	writer = tf.summary.FileWriter("/tmp/cifar/1")
	writer.add_graph(sess.graph)
	sess.run(tf.global_variables_initializer())
	sess.run(training_init_op)
	for i in range(1):
		print(i)
		x_image, label, id = sess.run(next_batch)


		sess.run(train_step, feed_dict = {x: x_image, y_: label})
		# print(sess.run(x_image, feed_dict = {x: x_image, y_: label}))


		if (i % 10 == 0) :
		# if (True) :
			summary = merged.eval(feed_dict = {x: x_image, y_: label})
			writer.add_summary(summary, i)
			train_accuracy = accuracy.eval(feed_dict = {x: x_image, y_: label})
			print("Step %d, training accuracy %g" % (i, train_accuracy))

	# sess.run(testing_init_op)
	# for i in range(1):
	# 	x_image, label, id = sess.run(next_batch)
	# 	train_accuracy = accuracy.eval(feed_dict = {x: x_image, y_: label})
	# 		# print(type(train_accuracy))
	# 	print("Step %d, testing accuracy %g" % (i, train_accuracy))




