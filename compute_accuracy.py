from keras.models import load_model
import numpy as np
import pickle
from keras.utils import np_utils

with open("train_images", "rb") as f:
	train_images = np.array(pickle.load(f))
with open("train_labels", "rb") as f:
	train_labels = np.array(pickle.load(f), dtype=np.uint8)

train_images = np.reshape(train_images, (train_images.shape[0], 100, 100, 1))
train_labels = np_utils.to_categorical(train_labels)

model_name = input('Enter model name: ')
model = load_model(model_name)
scores = model.evaluate(train_images, train_labels, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))