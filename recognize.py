import cv2
import numpy as np
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from keras.models import load_model
from preprocess_img import create_mask

CNN_MODEL = 'cnn_model_keras1.h5'
SHAPE_PREDICTOR_68 = "shape_predictor_68_face_landmarks.dat"

cnn_model = load_model(CNN_MODEL)
shape_predictor_68 = dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(1)
if cam.read()[0]==False:
	cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=100)

def get_image_size():
	img = cv2.imread('dataset/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred = model.predict(processed)
	print(pred[0]*100)
	pred_probab = pred[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def recognize():
	disp_probab, disp_class = 0, 0
	while True:
		img = cam.read()[1]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
		if len(faces) > 0:
			face = faces[0]
			shape_68 = shape_predictor_68(img, face)
			shape = face_utils.shape_to_np(shape_68)
			mask = create_mask(shape, img)
			masked = cv2.bitwise_and(gray, mask)
			(x, y, w, h) = face_utils.rect_to_bb(face)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
			faceAligned = fa.align(masked, gray, face)
			cv2.imshow('faceAligned', faceAligned)
			pred_probab, pred_class = keras_predict(cnn_model, faceAligned)
			if pred_probab > 0.5:
				disp_probab = pred_probab
				disp_class = pred_class
			cv2.putText(img, str(disp_probab), (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0))
			cv2.putText(img, str(disp_class), (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0))
		cv2.imshow('img', img)
		if cv2.waitKey(1) == ord('q'):
			break

keras_predict(cnn_model, np.zeros((150, 150, 1), dtype=np.uint8))
recognize()