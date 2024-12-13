from pathlib import Path
import os.path
import glob
import yaml
import cv2


configFile = os.path.join("v2", "datasets", "tarot", "tarot.yaml")


config = {}
with open(configFile) as stream:
    config = yaml.safe_load(stream)

print(config)


def normalizeX(x, maxX):
	return (x - 0) / (maxX - 0)


def normalizeY(y, maxY):
	return (y - 0) / (maxY - 0)


def image_resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=interpolation)

    # return the resized image
    return resized


def draw_bounding_box(image):
	# Display image and wait for user to select a region
	showCrosshair = False
	fromCenter = False
	bbox = cv2.selectROI("Select ROI", image, fromCenter, showCrosshair)
	if bbox != (0, 0, 0, 0):
		x, y, w, h = bbox
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	return (x, y, x + w, y + h)


# # Create window
# cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE);
# cv2.resizeWindow("Image", 800, 800);


def getClassId(val):
	for key, value in config['names'].items():
		if str(key) == val:
			return key
		elif value == val:
			return key
	return None


def getClassIdFromFilename(filename):
	val = Path(filename).stem
	return getClassId(val)

def getClassIdFromParentDirectory(filename):
	directory = os.path.dirname(filename)
	return getClassIdFromFilename(directory)

def getClassIdFromUser():
	class_id = None
	while class_id is None:
		val = input("Enter class name or id: ")
		class_id = getClassId(val)
	return class_id


def getCoordinatesFromImage(image_file):
	# Load image
	image = cv2.imread(image_file)
	resized = image_resize(image, height=800)

	# Draw bounding box
	bbox = draw_bounding_box(resized)

	# Fix scaling of bounding box
	size1 = image.shape[:2]
	size2 = resized.shape[:2]
	if size1 != size2:
		bbox = list(bbox)
		ratio = size2[0]/size1[0]
		bbox[0] = bbox[0] / ratio
		bbox[1] = bbox[1] / ratio
		bbox[2] = bbox[2] / ratio
		bbox[3] = bbox[3] / ratio
		bbox = tuple(bbox)

	# Return normalized coordinates
	return (
		normalizeX(bbox[0], size1[1]),
		normalizeY(bbox[1], size1[0]),
		normalizeX(bbox[2], size1[1]),
		normalizeY(bbox[3], size1[0])
	)


# for image_file in glob.glob(os.path.join(trainImageDirectory, "*.jpg")):
# 	print()
# 	label_file = image_file.replace("images", "labels").replace(".jpg", ".txt")
# 	if os.path.exists(label_file):
# 		continue
# 	print(image_file)
# 	class_id = getClassIdFromParentDirectory(image_file) or getClassIdFromFilename(image_file) or getClassIdFromUser()
# 	coordinates = getCoordinatesFromImage(image_file)
# 	with open(label_file, 'w') as fh:
# 		fh.write(f'{class_id} {coordinates[0]} {coordinates[1]} {coordinates[2]} {coordinates[3]}')



for directory in glob.glob("data/images/*"):
	for image_file in glob.glob(os.path.join(directory, '*.jpg')):
		label_file = image_file.replace(".jpg", ".txt")
		if os.path.exists(label_file):
			continue
		print(f'File: {image_file}')
		class_id = getClassIdFromParentDirectory(image_file)
		if class_id is None:
			class_id = getClassIdFromFilename(image_file)
		if class_id is None:
			class_id = getClassIdFromUser()
		coordinates = getCoordinatesFromImage(image_file)
		with open(label_file, 'w') as fh:
			fh.write(f'{class_id} {coordinates[0]} {coordinates[1]} {coordinates[2]} {coordinates[3]}')







