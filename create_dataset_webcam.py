import cv2
import numpy as np
import dlib
import pickle
import os
from imutils import face_utils
from imutils.face_utils import FaceAligner
from random import shuffle, randint
from preprocess_img import create_mask, get_bounding_rect

SHAPE_PREDICTOR_68 = "shape_predictor_68_face_landmarks.dat"

shape_predictor_68 = dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(1)
if cam.read()[0]==False:
	cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=250)


dataset = 'new_dataset/'
label = int(input('Enter label: '))
num_of_images = int(input('Enter number of images that you want to be taken: '))
starting_num = int(input('Enter starting image number: '))
count_images = starting_num
is_capturing = False
if not os.path.exists(dataset+str(label)):
	os.mkdir(dataset+str(label))
while True:
	img = cam.read()[1]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
	rand = randint(0, 10)
	if len(faces) > 0:
		face = faces[0]
		shape_68 = shape_predictor_68(img, face)
		shape = face_utils.shape_to_np(shape_68)
		mask = create_mask(shape, img)
		masked = cv2.bitwise_and(gray, mask)
		maskAligned = fa.align(mask, gray, face)
		faceAligned = fa.align(masked, gray, face)
		(x0, y0, x1, y1) = get_bounding_rect(maskAligned)
		faceAligned = faceAligned[y0:y1, x0:x1]
		faceAligned = cv2.resize(faceAligned, (100, 100))
		(x, y, w, h) = face_utils.rect_to_bb(face)
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
		if count_images-starting_num < int(num_of_images):
			if is_capturing:
				cv2.putText(img, str(count_images-starting_num), (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0) )
				if rand%2 == 0:
					faceAligned = cv2.flip(faceAligned, 1)
				cv2.imwrite(dataset+str(label)+'/'+str(count_images)+'.jpg', faceAligned)
				count_images += 1
		else:
			break
		cv2.imshow('faceAligned', faceAligned)
	cv2.imshow('img', img)
	keypress = cv2.waitKey(1)
	if keypress == ord('q'):
		break
	elif keypress == ord('c'):
		if is_capturing:
			is_capturing = False
		else:
			is_capturing = True
