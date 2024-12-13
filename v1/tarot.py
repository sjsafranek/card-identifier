import os
import glob
import json
import numpy
import pandas
from PIL import Image
import cv2
import clip
import torch
from sklearn.metrics.pairwise import cosine_similarity


ALLOWED_FILE_TYPES = (".jpg", ".png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
KERNEL_SIZE = (5, 5)


class ImageModel(object):

    def __init__(self, embeddings=[]):
        self.embeddings = embeddings

    def train(self, dataset_path):
        print(f'Training on {dataset_path}')
        for directory_name in os.listdir(dataset_path):
            directory_path = os.path.join(dataset_path, directory_name)
            for filename in os.listdir(directory_path):
                if filename.endswith(ALLOWED_FILE_TYPES):
                    image_path = os.path.join(directory_path, filename)
                    print(f'Processing {image_path}')
                    embedding = processAndEncodeImage(file_path=image_path).cpu().numpy().tolist()
                    self.embeddings.append({'name': directory_name, 'embedding': embedding})

    def export(self, file_path):
        with open(file_path, "w") as fh:
            json.dump(self.embeddings, fh)

    def identify(self, image):
        source_embedding = image
        if str == type(image):
            source_embedding = processAndEncodeImage(file_path=image)
        
        highest_similarity = -1
        best_match = None

        # Reshape the image embedding to be 2D
        source_embedding = source_embedding.cpu().numpy().reshape(1, -1)

        # Compare against each precomputed embedding
        for item in self.embeddings:
            name = item['name']
            target_embedding = item['embedding']    

            # Convert the card embedding to a numpy array
            target_embedding = torch.tensor(target_embedding).reshape(1, -1).numpy()

            # Compute cosine similarity
            similarity = cosine_similarity(source_embedding, target_embedding)[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = name
            if similarity >= 1:
                break

        return pandas.DataFrame([{'name': best_match, 'score': highest_similarity}])

    def detect(self, image_path, min_score = 0.75):
        # Lets be lazy and cheat...
        df = self.identify(image_path)
        if df['score'].max() > 0.95:
            return df
        #
        img = openImage(image_path)
        cropped_cards = detect_and_crop_cards(img)
        matches = []
        # TODO :: Use multiprocess
        for i, card in enumerate(cropped_cards):
            image_embedding = processAndEncodeImage(image=card['image'])
            match = self.identify(image_embedding)
            matches.append({
                'name': match.iloc[0]['name'],
                'score': match.iloc[0]['score'],
                'rect': card['rect']
            })
        df = pandas.DataFrame(matches)
        df = df[df['score'] > min_score]
        indices = df.groupby('name')['score'].idxmax()
        return df.loc[indices]

    @staticmethod
    def load(self, file_path):
        with open(file_path, "r") as f:
            return ImageModel(embeddings=json.load(f))



def load(file_path):
    with open(file_path, "r") as f:
        return ImageModel(embeddings=json.load(f))


def openImage(file_path):
    return Image.open(file_path)


def processImage(image=None, file_path=None):
    if file_path:
        image = openImage(file_path)
    return PREPROCESS(image).unsqueeze(0).to(DEVICE)


def encodeImage(image):
    with torch.no_grad():
        return MODEL.encode_image(image)


def processAndEncodeImage(image=None, file_path=None):
    return encodeImage(processImage(image=image, file_path=file_path))






# Detect and crop cards from the spread
def detect_and_crop_cards(uploaded_image):
    # Convert PIL image to OpenCV format
    image = numpy.array(uploaded_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, KERNEL_SIZE, 0)

    # Adaptive thresholding for binary segmentation
    thresh = cv2.adaptiveThreshold(
        # blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Dilate to close gaps
    kernel = numpy.ones(KERNEL_SIZE, numpy.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    min_area = 20000  # Adjust based on card size
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]

    # Debug: Draw filtered contours
    debug_image_filtered = image.copy()
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(debug_image_filtered, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f'tmp/boxes_filtered.jpg', debug_image_filtered)

    # Merge nearby bounding boxes
    merged_boxes = merge_nearby_boxes(bounding_boxes, threshold=50)  # Adjust threshold as needed
    
    # Crop cards
    cropped_cards = []
    for x, y, w, h in merged_boxes:
        # Stefan - checking the aspect ratio seems to give false negatives...
        aspect_ratio = w / h
        if 0.5 < aspect_ratio < 2.0:  # Ensure only plausible card shapes are detected
            padding = 10
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, image.shape[1] - x)
            h = min(h + 2 * padding, image.shape[0] - y)
            card = image[y : y + h, x : x + w]
            cropped_cards.append({
                'rect': {
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                },
                'image': Image.fromarray(card)
            })

    # Debug: Draw merged boxes
    image_with_boxes = image.copy()
    for x, y, w, h in merged_boxes:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(f'tmp/boxes_merged.jpg', image_with_boxes)

    return cropped_cards


def merge_nearby_boxes(boxes, threshold=20):  # Reduce threshold
    merged_boxes = []
    for box in boxes:
        x, y, w, h = box
        new_box = True
        for (mx, my, mw, mh) in merged_boxes:
            # Check if the boxes overlap or are close to each other
            if (
                abs(x - mx) < threshold and abs(y - my) < threshold
                and abs((x + w) - (mx + mw)) < threshold
                and abs((y + h) - (my + mh)) < threshold
            ):
                # Merge boxes
                merged_boxes[i] = merge_boxes(box, (mx, my, mw, mh))
                new_box = False
                break
            # elif contains(box, (mx, my, mw, mh)):
            #     new_box = False
            # elif contains((mx, my, mw, mh), box):
            #     new_box = False
        if new_box:
            merged_boxes.append((x, y, w, h))
    return merged_boxes


def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (
        min(x1, x2),
        min(y1, y2),
        max(x1 + w1, x2 + w2) - min(x1, x2),
        max(y1 + h1, y2 + h2) - min(y1, y2),
    )


# # Separating Axis Theorem
# def intersects(rect1, rect2):
#     return not (rect1.top_right.x < rect2.bottom_left.x
#                 or rect1.bottom_left.x > rect2.top_right.x
#                 or rect1.top_right.y < rect2.bottom_left.y
#                 or rect1.bottom_left.y > rect2.top_right.y)


def getPoints(rect):
    x1, y1, w, h = rect
    x2 = x1 + w
    y2 = y1 + h
    return [
        (x1, y1),
        (x1, y2),
        (x2, y2),
        (x2, y1)
    ]

def contains(rect, point):
    if 4 == len(point):
        for pt in getPoints(point):
            if not contains(rect, pt):
                return False
        return True
    points = getPoints(rect)
    bl = points[0]
    tr = points[2]
    return (point[0] > bl[0] and point[0] < tr[0] and point[1] > bl[1] and point[1] < tr[1]) 

