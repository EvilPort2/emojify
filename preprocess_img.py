import cv2
import numpy as np
from imutils import contours

def euclidean_distance(a, b):
	dist = 0
	if len(a) == len(b):
		for i in range(len(a)):
			dist += (a[i]-b[i])**2
		dist = np.sqrt(dist)
	return int(dist)

def highest_euclidean_distance(shape, fixed_point, *other_points):
	num = len(other_points)
	largest_distance = 0
	for point in other_points:
		dist = euclidean_distance(fixed_point, shape[point])
		if dist > largest_distance:
			largest_distance = dist
	return largest_distance

def centroid(shape, *points):
	num_of_points = len(points)
	x, y = 0, 0
	for point in points:
		x += shape[point][0]
		y += shape[point][1]
	centroid = (int(x/num_of_points), int(y/num_of_points))
	return centroid

def get_points(face_part, shape):
	points = [shape[point] for point in face_part]
	return np.array(points)

def create_mask(shape, img):
	height, width, channels = img.shape
	mask = np.zeros((height, width), dtype=np.uint8)
	right_eye = (36, 37, 38, 39, 40, 41)
	left_eye = (42, 43, 44, 45, 46, 47)
	mouth = (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)
	nose = (27, 31, 33, 35)
	left_eyebrow = (17, 18, 19, 20, 21)
	right_eyebrow = (22, 24, 25, 26, 26)
	
	middle_right_eye = centroid(shape, *right_eye)
	middle_left_eye = centroid(shape, *left_eye)
	radius_left_eye = highest_euclidean_distance(shape, middle_left_eye, *left_eye) 
	radius_right_eye = highest_euclidean_distance(shape, middle_right_eye, *right_eye)
	mask = cv2.circle(mask, middle_right_eye, radius_right_eye, 255, -1)
	mask = cv2.circle(mask, middle_left_eye, radius_left_eye, 255, -1)

	middle_mouth = centroid(shape, *mouth)
	radius_mouth = highest_euclidean_distance(shape, middle_mouth, *mouth)
	mask = cv2.circle(mask, middle_mouth, radius_mouth, 255, -1)

	mask = cv2.fillPoly(mask, [get_points(left_eyebrow, shape)], 255)
	mask = cv2.fillPoly(mask, [get_points(right_eyebrow, shape)], 255)
	mask = cv2.fillPoly(mask, [get_points(nose, shape)], 255)
	return mask

def get_bounding_rect(img):
	cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
	boundingBoxes = contours.sort_contours(cnts, method='left-to-right')[1]
	
	x0, y0 = boundingBoxes[0][0], boundingBoxes[0][1]
	x1, y1 = 0, 0
	for i, (x,y,w,h) in enumerate(boundingBoxes):
		if i == 0:
			continue
		if x+w > x1:
			x1 = x+w
		if y+h > y1:
			y1 = y+h
	return x0,y0,x1,y1