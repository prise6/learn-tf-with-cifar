from os import listdir
from os.path import isfile, join, basename
import tensorflow as tf
import time
import pandas as pd
from random import randint


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

labels = pd.read_csv('../Datas/trainLabels.csv')

correspondance = pd.DataFrame(data = {
	'label'    : ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane'],
	'id_label' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

labels = labels.merge(correspondance, on = 'label')
labels['file'] = [randint(1,10) for _ in range(len(labels))]



path = '../Datas/train'
img_to_read = [join(path, img) for img in listdir(path) if isfile(join(path, img))]


def readMyImg(path):
	
	queue = tf.train.string_input_producer(path)

	reader = tf.WholeFileReader()
	label, value = reader.read(queue)
	image = tf.image.decode_png(value) 

	return label, image

label, image = readMyImg(img_to_read)


files_out = list(range(1,11))
files_out_writers = list(map(lambda f : tf.python_io.TFRecordWriter('../Datas/train_part_{}.tfrecords'.format(f)), files_out))



startTime = time.time()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(len(img_to_read)):
		label_s, image_s = sess.run([label, image])
		print(i)

		idx = int(basename(label_s).split(b'.')[0])
		label_s = labels[labels['id'] == idx].id_label
		file = labels[labels['id'] == idx].file

		example = tf.train.Example(features=tf.train.Features(feature = {
			'id': _int64_feature(int(idx)),
			'label' : _int64_feature(int(label_s)),
			'input' : _bytes_feature(image_s.tostring())
		}))

		files_out_writers[int(file)-1].write(example.SerializeToString())

	coord.request_stop()
	coord.join(threads)

for w in files_out_writers:
	w.close()


print("Time taken: %f" % (time.time() - startTime))