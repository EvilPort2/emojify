# Emojify
Turn your facial expression into an emoji.

## Note
This project is still under development. Expect errors in recognition of emoji or other errors.

## What I did here
1. Decided what emojis to use.
2. Stored 250 faces for each facial expression in a special way. The special way is that I took only the eyebrows, eyes, nose and mouth of the face. Rest of the face is removed. (Only my face was used)
3. Trained a CNN on these images.
4. As of today there are 11 facial expressions. (Discussed later).

## Outcome
Watch it <a href="https://youtu.be/izUO2rl0Ur8">here</a>.

## Requirements 
0. Python 3.x
1. <a href="https://tensorflow.org">Tensorflow 1.5</a>
2. <a href="https://keras.io">Keras</a>
3. OpenCV 3.4
4. h5py
5. A good grasp over the above 4 topics along with neural networks. Refer to the internet if you have problems with those. I myself am just a begineer in those.
6. A good CPU (preferably with a GPU).
7. Patience.... A lot of it.
8. Tensorboard (for visualizing training)

## Facial expressions used
0 - Neutral<br>
1 - Smile/Happy<br>
2 - Sad<br>
3 - Wink<br>
4 - Kiss<br>
5 - Surprised<br>
6 - Angry<br>
7 - Monkey face<br>
8 - Wink with tongue out<br>
9 - Scared/Terrified<br>
10 - Disgusted<br>

## How to use this repo
This project is done by me and me only. No one else helped me. The model is trained with my face only. So the model might not detect your expressions correctly. Here is what you can do. If you are a newbie to neural networks or machine learning please learn them. This guide contains a lot of technical stuff which you might find hard to understand. 

### Create your facial expression dataset
1. Start this file 
		
			python create_dataset_webcam.py
2. It will ask for label i.e. facial expression id (more about it later), number of pictures you want to take, and starting image number. You can take as many pictures for each expression as you want but make sure you do it using different lighting conditions, facial poisitions etc. Also make sure you take same number of images for each gesture or else you might introduce a bias. I usually keep it to 250.
3. For the starting image number, make sure you check the images in the dataset/ folder. If the file name of the last file is 249.jpg then you should enter 250 i.e. (last image number + 1)
4. The images will be stored in the new_dataset/ folder.

### Retraining with the new_dataset (HARDER way)
You will see why this method is a bit hard.

#### Load the images of the new_dataset/
1. Start the load_images.py file

			python load_images.py
2. Here you will be asked for which dataset folder to use. Enter 'new_dataset/'
3. The images will be stored as pickle file.
4. You will get 6 pickle files viz train_images, train_labels, test_images, test_labels, val_images, and val_labels

#### Retrain the model 
1. Start the retrain_cnn_keras.py file.
		
			python retrain_cnn_keras.py
2. Here you will be asked for the trained model file name, new model file name, learning rate, epochs, and batch number.
3. For the trained model file name enter cnn_model_keras.h5
4. For the new model file name you can enter anything. A warning though, if you keep it blank or enter cnn_model_keras.h5, it will replace the original model when training if the validation accuracy increases from the previous step.
5. What I usually do is I enter cnn_model_keras1.h5 or something like that so that I do not mess it up.
6. For the rest of the hyper parameters, they will depend on how large the new_dataset/ is.
7. If the number of images for each expression is >= 250, I usually keep the default learning rate, 10-20 epochs and a batch size of 100. 
8. After the training you will see the accuracy of the model. 

#### Check the model's accuracy against the dataset/ folder 
1. Start the load_images.py file

			python load_images.py
2. Here you will be asked for which dataset folder to use. Enter 'dataset/'
3. Start the compute_accuracy.py file
			
			python compute_accuracy.py
4. You will be asked to enter the model name. Enter the new model name. Usually in my case it is cnn_model_keras1.h5.
5. If the accuracy is small i.e the CNN error is greater than 4-5% try retraining the model again. That is, do the 2nd step 'Retrain the model'. Only now you need to use cnn_model_keras1.h5 as the model file name and you can keep the new model file name as blank or anything you want.
6. <b>The lower accuracy happens perhaps due to the incorrect settings of the hyperparameters. Please let me know if I am doing it wrong. Need some expert advice here.</b>

### Training the model from the beginning (EASIER way)
1. After you have stored your images in the new_dataset/ folder, move the contents of the folder to the dataset folder.
2. Start the train_cnn_keras.py file
			
			python train_cnn_keras.py

### Start emojify
1. If you are satisfied with the accuracy of the model then you can start the emoji.py file.

			python emojify.py

## How to contribute
If you want to contribute to the dataset then please make a pull request. I will be more than happy to merge the present dataset with your dataset. Just make sure you have moved the contents of the new_dataset/ folder to the dataset/ folder before making a pull request.<br>
If you can optimize the code then feel free to inform me.

## How to cite
