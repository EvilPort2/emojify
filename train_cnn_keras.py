import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import load_model
from time import time
K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
	img = cv2.imread('dataset/0/1.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(os.listdir('dataset/'))

image_x, image_y = get_image_size()

def cnn_model():
	num_of_classes = get_num_of_classes()
	model = Sequential()
	model.add(Conv2D(32, (5,5), input_shape=(image_x, image_y, 1), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(10, 10), strides=(10, 10), padding='same'))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	model.add(Dense(num_of_classes, activation='softmax'))
	sgd = optimizers.SGD(lr=1e-2)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	filepath="cnn_model_keras.h5"
	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint1]
	from keras.utils import plot_model
	plot_model(model, to_file='model.png', show_shapes=True)
	return model, callbacks_list

def train():
	with open("train_images", "rb") as f:
		train_images = np.array(pickle.load(f))
	with open("train_labels", "rb") as f:
		train_labels = np.array(pickle.load(f), dtype=np.uint8)

	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.uint8)

	with open("val_images", "rb") as f:
		val_images = np.array(pickle.load(f))
	with open("val_labels", "rb") as f:
		val_labels = np.array(pickle.load(f), dtype=np.uint8)

	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))

	train_labels = np_utils.to_categorical(train_labels)
	test_labels = np_utils.to_categorical(test_labels)
	val_labels = np_utils.to_categorical(val_labels)

	model, callbacks_list = cnn_model()
	tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
	callbacks_list.append(tensorboard)
	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=15, batch_size=100, callbacks=callbacks_list)
	model = load_model('cnn_model_keras.h5')
	scores = model.evaluate(val_images, val_labels, verbose=1)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))

train()