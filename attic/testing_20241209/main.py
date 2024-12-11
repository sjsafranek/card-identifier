import os.path
from PIL import Image
import numpy as np
import cv2
import pytesseract

from src.models import Rectangle

#filename = os.path.join('images', 'a_spades.jpg')
#filename = os.path.join('images', '4_spades.jpg')

filename = os.path.join('images', 'knight.jpg')
#filename = os.path.join('images', 'comet.jpg')
#filename = os.path.join('images', 'euryale.jpg')


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'



def load_image(filename):
    img = Image.open(filename)
    return np.array(img)


def clean_image(img):
    #--- convert the image to HSV color space ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #--- find Otsu threshold on hue and saturation channel ---
    ret, thresh_H = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh_S = cv2.threshold(hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #--- some morphology operation to clear unwanted spots ---
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh_H + thresh_S, kernel, iterations = 1)
    #--- Apply a gaussian blur for further cleanup ---
    blur = cv2.GaussianBlur(dilation, (5,5), 0)
    #--- Return cleaned image ---
    return blur


def find_card_contour(cleaned):
    #--- find contours on the result above ---
    #contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #--- find the second largest contour ---
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]), reverse=True)
    return contours[index_sort[1]]


def deskew_image(img):
	gray = img
	# check number of channels
	if 3 == len(img.shape):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turn to gray
	imOTSU = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)[1] # get threshold with positive pixels as text
	coords = np.column_stack(np.where(imOTSU > 0)) # get coordinates of positive pixels (text)
	angle = cv2.minAreaRect(coords)[-1] # get a minAreaRect angle
	if angle < -45: # adjust angle
	    angle = -(90 + angle)
	else:
	    angle = -angle
	# get width and center for RotationMatrix2D
	h = gray.shape[0] # get width and height of image
	w = gray.shape[1] # get width and height of image
	center = (w // 2, h // 2) # get the center of the image
	M = cv2.getRotationMatrix2D(center, angle, 1.0) # define the matrix
	rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # apply it
	return rotated


def clip_image(image, rect):
    return img[rect.minY: rect.maxY, rect.minX: rect.maxX]


def calculateMeanSquaredError(img1, img2):
    print(img1.shape)
    print(img2.shape)
    if (img1.shape[0] > img2.shape[0]):
    	
    else if (img1.shape[0] < img2.shape[0]):

    if (img1.shape[1] > img2.shape[1]):

    else if (img1.shape[1] < img2.shape[1]):

    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse


def normalize(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    zeros = np.zeros((grey.shape[0], grey.shape[1]))
    normalized = cv2.normalize(grey, zeros, 0, 255, cv2.NORM_MINMAX)
    threshold = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)[1]
    return threshold


def get_text_from_image(img):
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)
    #img = cv2.GaussianBlur(img, (5, 5), 0)
    return pytesseract.image_to_string(img)



img = load_image(filename)
cleaned = clean_image(img)
cv2.imwrite('temp/cleaned.jpg', img)
contour = find_card_contour(cleaned)
bbox = Rectangle(contour=contour)
#cv2.rectangle(img, bbox.topLeft, bbox.bottomRight, (0,0,255), 10)
#cv2.imwrite('temp/bbox.jpg', img)
clipped = clip_image(img, bbox)
cv2.imwrite('temp/clipped.jpg', clipped)
#deskewed = deskew_image(clipped)
#cv2.imwrite('temp/deskewed.jpg', deskewed)

target = normalize(clipped)
cv2.imwrite('temp/target.jpg', clipped)


filename = os.path.join('images', 'knight2.jpg')
img = load_image(filename)
cleaned = clean_image(img)
cv2.imwrite('temp/cleaned2.jpg', img)
contour = find_card_contour(cleaned)
bbox = Rectangle(contour=contour)
clipped = clip_image(img, bbox)
cv2.imwrite('temp/clipped2.jpg', clipped)
#deskewed = deskew_image(clipped)
#cv2.imwrite('temp/deskewed2.jpg', deskewed)

source = normalize(clipped)
cv2.imwrite('temp/source.jpg', clipped)



mse = calculateMeanSquaredError(target, source)
print(mse)



# img = cv2.imread(filename)

# #--- convert the image to HSV color space ---
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# #--- find Otsu threshold on hue and saturation channel ---
# ret, thresh_H = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, thresh_S = cv2.threshold(hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# #--- some morphology operation to clear unwanted spots ---
# kernel = np.ones((5, 5), np.uint8)
# dilation = cv2.dilate(thresh_H + thresh_S, kernel, iterations = 1)

# blur = cv2.GaussianBlur(dilation, (5,5), 0)

# #--- find contours on the result above ---
# #contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # #--- since there were few small contours found, retain those above a certain area ---
# img2 = img.copy()
# for i in range(len(contours)):
#      contour = contours[i]
#      cv2.drawContours(img2, [contour], -1, (0, 255, 0), 3)


# # Display second largest contour
# index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]), reverse=True)
# cv2.drawContours(img2, [contours[index_sort[0]]], -1, (0, 255, 0), 10)

# # Get bounding box of second largest contour
# contour = contours[index_sort[1]]
# x, y, w, h = cv2.boundingRect(contour)
# cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255), 7)


# cv2.imwrite('test_opencv.jpg', img2)







