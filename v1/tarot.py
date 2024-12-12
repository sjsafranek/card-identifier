import os
import glob
import json
import numpy
import statistics
from PIL import Image
import clip
import torch
import cv2
from sklearn.metrics.pairwise import cosine_similarity


ALLOWED_FILE_TYPES = (".jpg", ".png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)


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
                    name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the card name
                    print(f'Processing {image_path}')
                    embedding = processAndEncodeImage(file_path=image_path).cpu().numpy().tolist()
                    self.embeddings.append({'name': name, 'embedding': embedding})

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

        return best_match, highest_similarity

    def detect(self, image_path):
        img = openImage(image_path)
        cropped_cards = detect_and_crop_cards(img)
        print(f'Detected {len(cropped_cards)} cards')
        for i, card_image in enumerate(cropped_cards):
            card_image.save(f'crop_{1}.jpg')
            image_embedding = processAndEncodeImage(image=card_image)
            yield self.identify(image_embedding)


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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding for binary segmentation
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to close gaps
    kernel = numpy.ones((5, 5), numpy.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    min_area = 20000  # Adjust based on card size
    # areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c)]
    print(min(areas))
    print(max(areas))
    print(statistics.mean(areas))
    print(statistics.stdev(areas))
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]

    # Debug: Draw filtered contours
    debug_image_filtered = image.copy()
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(debug_image_filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Merge nearby bounding boxes
    merged_boxes = merge_nearby_boxes(bounding_boxes, threshold=50)  # Adjust threshold as needed
    
    # Crop cards
    cropped_cards = []
    for x, y, w, h in merged_boxes:
        # Stefan - checking the aspect ratio seems to give false negatives...
        # aspect_ratio = w / h
        # if 0.5 < aspect_ratio < 2.0:  # Ensure only plausible card shapes are detected
        padding = 10
        x = max(x - padding, 0)
        y = max(y - padding, 0)
        w = min(w + 2 * padding, image.shape[1] - x)
        h = min(h + 2 * padding, image.shape[0] - y)
        card = image[y : y + h, x : x + w]
        cropped_cards.append(Image.fromarray(card))

    # Debug: Draw merged boxes
    debug_image_merged = image.copy()
    for x, y, w, h in merged_boxes:
        cv2.rectangle(debug_image_merged, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return cropped_cards





def merge_nearby_boxes(boxes, threshold=20):  # Reduce threshold
    merged_boxes = []
    for box in boxes:
        x, y, w, h = box
        new_box = True
        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            # Check if the boxes overlap or are close to each other
            if (
                abs(x - mx) < threshold and abs(y - my) < threshold
                and abs((x + w) - (mx + mw)) < threshold
                and abs((y + h) - (my + mh)) < threshold
            ):
                # Merge boxes
                merged_boxes[i] = (
                    min(x, mx),
                    min(y, my),
                    max(x + w, mx + mw) - min(x, mx),
                    max(y + h, my + mh) - min(y, my),
                )
                new_box = False
                break
        if new_box:
            merged_boxes.append((x, y, w, h))
    return merged_boxes

