import json
import numpy
from PIL import Image
import clip
import torch
import cv2
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_EMBEDDINGS_FILE = "tarot_embeddings.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS = {}
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)


def train(images_path, file_path=DEFAULT_EMBEDDINGS_FILE):
    global EMBEDDINGS

    # Path to the tarot card images
    EMBEDDINGS = {}

    # Process each tarot card image
    for filename in os.listdir(images_path):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(images_path, filename)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = MODEL.encode_image(image).cpu().numpy().tolist()
            card_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the card name
            EMBEDDINGS[card_name] = embedding
            print(f"Processed {card_name}")

    save_embeddings(EMBEDDINGS, filename=filename)
    print(f"All tarot embeddings have been saved to '{file_path}'")
    return EMBEDDINGS


def open_image_from_file(file_path):
    return Image.open(file_path)


def process_image(image=None, file_path=None):
    if file_path:
        image = open_image_from_file(file_path)
    return PREPROCESS(image).unsqueeze(0).to(DEVICE)


def encode_image(image):
    with torch.no_grad():
        return MODEL.encode_image(image)


def process_and_encode_image(image=None, file_path=None):
    img = process_image(image=image, file_path=file_path)
    return encode_image(img)


def save_embeddings(embeddings, file_path=DEFAULT_EMBEDDINGS_FILE):
    with open("tarot_embeddings.json", "w") as fh:
        json.dump(tarot_embeddings, fh)


def load_embeddings(file_path=DEFAULT_EMBEDDINGS_FILE):
    global EMBEDDINGS
    with open(file_path, "r") as f:
        EMBEDDINGS = json.load(f)
    return EMBEDDINGS


# Step 4: Compare the uploaded image to precomputed embeddings
def identify_card(image_embedding, embeddings=None):
    embeddings = embeddings or EMBEDDINGS
    highest_similarity = -1
    best_match = None

    # Reshape the image embedding to be 2D
    image_embedding = image_embedding.cpu().numpy().reshape(1, -1)

    # Compare against each precomputed embedding
    for card_name, card_embedding in embeddings.items():
        # Convert the card embedding to a numpy array
        card_embedding = torch.tensor(card_embedding).reshape(1, -1).numpy()

        # Compute cosine similarity
        similarity = cosine_similarity(image_embedding, card_embedding)[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = card_name

    return best_match, highest_similarity




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

    # Display binary mask
    # st.image(dilated, caption="Binary Mask (Debug)", use_container_width=True)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size
    min_area = 20000  # Adjust based on card size
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]

    # Debug: Draw filtered contours
    debug_image_filtered = image.copy()
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(debug_image_filtered, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # st.image(debug_image_filtered, caption="Filtered Contours (Debug)", channels="BGR", use_container_width=True)

    # Merge nearby bounding boxes
    merged_boxes = merge_nearby_boxes(bounding_boxes, threshold=50)  # Adjust threshold as needed
    
    # Crop cards
    cropped_cards = []
    for x, y, w, h in merged_boxes:
        aspect_ratio = w / h
        if 0.5 < aspect_ratio < 2.0:  # Ensure only plausible card shapes are detected
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
    # st.image(debug_image_merged, caption="Merged Bounding Boxes (Debug)", channels="BGR", use_container_width=True)

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


def init():
    load_embeddings()