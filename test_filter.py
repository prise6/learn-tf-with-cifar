import tensorflow as tf
from tensorflow.contrib.data import Iterator
from PIL import Image


def parser(record):

	keys_to_features = {
		"id": tf.FixedLenFeature((), tf.int64, default_value = tf.zeros([], dtype = tf.int64)),
		"label": tf.FixedLenFeature((), tf.int64, default_value = tf.zeros([], dtype = tf.int64)),
		"input": tf.FixedLenFeature((), tf.string)
	}

	parsed = tf.parse_single_example(record, keys_to_features)

	image = tf.decode_raw(parsed["input"], tf.uint8)
	label = tf.one_hot(tf.cast(parsed["label"], tf.int32), depth = 10, axis = 0, on_value = 1.0, off_value = 0.0)
	img_id = tf.cast(parsed["id"], tf.int32)

	return image, label, img_id


filenames = ['../Datas/train_part_10.tfrecords']

dataset = tf.contrib.data.TFRecordDataset(filenames)

dataset = dataset.map(parser)
dataset = dataset.filter( lambda x, y, z: y[1]>0)
dataset = dataset.batch(10)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer)
x, label, id = sess.run(next_element)
print(label)
