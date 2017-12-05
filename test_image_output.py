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
	image = tf.reshape(image, [32, 32, 3])
	label = tf.one_hot(tf.cast(parsed["label"], tf.int32) - 1, depth = 10)
	img_id = tf.cast(parsed["id"], tf.int32)
	label_or = tf.cast(parsed["label"], tf.int32)
	# return image



	# return {"image_data": image, "label": label, "img_id": img_id}
	return image, label, img_id, label_or


def training_data():
	filenames = ['../Datas/train_part_4.tfrecords']
	# filenames = ['../Datas/train_part_1.tfrecords']
	dataset = tf.contrib.data.TFRecordDataset(filenames)

	dataset = dataset.map(parser)
	dataset = dataset.shuffle(buffer_size = 1000)
	# dataset = dataset.batch(1)
	# dataset = dataset.repeat()

	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	return next_element

next_element = training_data()

with tf.Session() as sess:
	for i in range(1):
		input, label, id, label_or = sess.run(next_element)
		print(label)
		print(label_or)
		# print(input)
		# image = tf.reshape(input, [-1, 32, 32, 3])
		img = Image.fromarray(input, 'RGB')
		img.save('test.png')