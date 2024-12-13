import sys
import cv2


filepath = sys.argv[1]


def image_resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
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
	cv2.imshow("Image", image)
	cv2.waitKey(0)
	return bbox


# Create window
cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE);
cv2.resizeWindow("Image", 800, 800);

# Load image
image = cv2.imread(filepath)
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


def normalizeX(x):
	return (x - 0) / (size1[0] - 0)

def normalizeY(y):
	return (y - 0) / (size1[1] - 0)


print(normalizeX(bbox[0]), normalizeY(bbox[1]), normalizeX(bbox[2]), normalizeY(bbox[3]))