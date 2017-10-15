import tensorflow as tf
from tensorflow.contrib.data import Iterator
from PIL import Image


def parser(record):

	keys_to_features = {
		"id": tf.FixedLenFeature((), tf.int64, default_value = tf.zeros([], dtype = tf.int64)),
		"input": tf.FixedLenFeature((), tf.string),
		"label": tf.FixedLenFeature((), tf.int64, default_value = tf.zeros([], dtype = tf.int64)),
	}

	parsed = tf.parse_single_example(record, keys_to_features)
	

	image = tf.decode_raw(parsed["input"], tf.uint8)
	# image = tf.image.decode_image(image)
	# image = tf.reshape(image, [-1, 32, 32, 3])
	label = tf.one_hot(tf.cast(parsed["label"], tf.int32), depth = 10)
	img_id = tf.cast(parsed["id"], tf.int32)

	# return image



	# return {"image_data": image, "label": label, "img_id": img_id}
	return image, label, img_id


def training_data():
	filenames = ['../Datas/train_part_1.tfrecords', '../Datas/train_part_2.tfrecords', '../Datas/train_part_3.tfrecords', '../Datas/train_part_4.tfrecords']
	# filenames = ['../Datas/train_part_1.tfrecords']
	dataset = tf.contrib.data.TFRecordDataset(filenames)

	dataset = dataset.map(parser)
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
	dataset = dataset.batch(3000)

	# iterator = dataset.make_one_shot_iterator()
	# next_element = iterator.get_next()
	# return next_element
	return dataset

training_dataset = training_data()
testing_dataset = testing_data()


iterator = Iterator.from_structure(training_dataset.output_types,
                                   training_dataset.output_shapes)
next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
testing_init_op = iterator.make_initializer(testing_dataset)





x = tf.placeholder(tf.float32, [None, 1024*3], name = "x")
x_image = tf.reshape(x, [-1, 32, 32, 3])
# tf.summary.image('input', x_image, 5)

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 10], stddev = 0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape = [10]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 10, 30], stddev = 0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape = [30]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides = [1, 1, 1, 1], padding  = 'SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 30, 1000], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [1000]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1000, 10], stddev = 0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]))
h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(h_fc2)

y_ = tf.placeholder(tf.float32, [None, 10], name = "labels")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))

train_step = tf.train.AdamOptimizer(.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(training_init_op)
	for i in range(1000):
		input, label, id = sess.run(next_batch)
		# print(label)
		# print(id)
		sess.run(train_step, feed_dict = {x: input, y_: label, keep_prob: 0.5)}

		# print(sess.run([tf.reduce_mean(tf.cast(correct_prediction, tf.float32))], feed_dict = {x: input, y_: label, keep_prob: 0.5}))

		if (i % 10 == 0) :
			train_accuracy = accuracy.eval(feed_dict = {x: input, y_: label, keep_prob: 1})
			# print(type(train_accuracy))
			print("Step %d, training accuracy %g" % (i, train_accuracy))

		# img = Image.fromarray(input, 'RGB')
		# img.save('test.png')

	sess.run(testing_init_op)
	for i in range(1):
		input, label, id = sess.run(next_batch)
		train_accuracy = accuracy.eval(feed_dict = {x: input, y_: label, keep_prob: 1})
			# print(type(train_accuracy))
		print("Step %d, testing accuracy %g" % (i, train_accuracy))




