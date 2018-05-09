import cv2
import numpy as np

def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image    
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def blend(image, emoji, position):
	x, y, w, h = position
	emoji = cv2.resize(emoji, (w, h))
	try:
		image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
	except:
		pass
	return image

'''
img = cv2.imread('Surprised_Emoji_Icon.png', -1)
img = cv2.resize(img, (100, 100))
black = np.zeros((480, 640, 3), dtype=np.uint8)
white = np.ones((480, 640, 3), dtype=np.uint8)*127

y, x, c = img.shape
white[0:240, 0:320] = [255, 0, 0]
white[240:, 320:] = [255, 255, 0]tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(img)
rgba = [b,g,r, alpha]
blend = blend_transparent(white[210:210+y, 330:330+x], img)
white[210:210+y, 330:330+x] = blend
cv2.imshow('black', white)
cv2.waitKey(0)
'''