import cv2
import numpy as np
import dlib
import pickle
import os, csv
from imutils import face_utils
from imutils.face_utils import FaceAligner
from preprocess_img import create_mask, get_bounding_rect
from random import shuffle

SHAPE_PREDICTOR_68 = "shape_predictor_68_face_landmarks.dat"

shape_predictor_68 = dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector = dlib.get_frontal_face_detector()

fa = FaceAligner(shape_predictor_68, desiredFaceWidth=250)

DATASET = 'new_dataset/'
FACES = 'faces/'
folders = os.listdir(FACES)
for folder in folders:
	folder = folder + '/'
	c = 0
	for image in os.listdir(FACES+folder):
		path = FACES+folder+image
		img = cv2.imread(path)
		print(image)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
		if len(faces) > 0:
			for face in faces:
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
				cv2.imshow('faceAligned', faceAligned)
				cv2.imwrite(DATASET+folder+image, faceAligned)
				c+=1
		cv2.imshow('img', img)
		if cv2.waitKey(1) == ord('q'):
			break