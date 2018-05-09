from keras.models import load_model
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import pickle
from time import time

def train(pre_model_name, new_model_name, learning_rate, epochs, batch_size):
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

	train_images = np.reshape(train_images, (train_images.shape[0], 100, 100, 1))
	test_images = np.reshape(test_images, (test_images.shape[0], 100, 100, 1))
	val_images = np.reshape(val_images, (val_images.shape[0], 100, 100, 1))

	train_labels = np_utils.to_categorical(train_labels)
	test_labels = np_utils.to_categorical(test_labels)
	val_labels = np_utils.to_categorical(val_labels)

	checkpoint = ModelCheckpoint(new_model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))
	callbacks_list = [checkpoint, tensorboard]
	model = load_model(pre_model_name)
	sgd = optimizers.SGD(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
	model = load_model(new_model_name)
	scores = model.evaluate(val_images, val_labels, verbose=1)
	print("CNN Error: %.2f%%" % (100-scores[1]*100))


while True:
	pre_model_name = input('Enter pre trained model file name: ')
	if pre_model_name != '':
		break

new_model_name = input('Enter new model file name (default is same as pre trained model file name): ')
if new_model_name == '':
	new_model_name = pre_model_name

while True:
	learning_rate = input('Enter learning rate (default 0.01): ')
	if learning_rate == '':
		learning_rate = 0.01
		break
	try:
		learning_rate = float(learning_rate)
		break
	except:
		continue

while True:
	epochs = input('Enter epochs (default 100): ')
	if epochs == '':
		epochs = 100
		break
	try:
		epochs = int(epochs)
		break
	except:
		continue

while True:
	batch_size = input('Enter batch size (default 50): ')
	if batch_size == '':
		batch_size = 50
		break
	try:
		batch_size = int(batch_size)
		break
	except:
		continue


train(pre_model_name, new_model_name, learning_rate, epochs, batch_size)