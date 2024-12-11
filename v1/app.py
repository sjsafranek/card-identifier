import cv2
import numpy as np
from PIL import Image
import streamlit as st
import clip
import torch
import json
# from sklearn.metrics.pairwise import cosine_similarity

import tarot


# Load tarot embeddings and CLIP model
@st.cache_resource
def load_resources():
    with open("tarot_embeddings.json", "r") as f:
        tarot_embeddings = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return tarot_embeddings, model, preprocess, device

# # Process the uploaded image
# def process_image(image, preprocess, device):
#     image = preprocess(image).unsqueeze(0).to(device)
#     return image

# # Identify the tarot card
# def identify_card(image_embedding, tarot_embeddings):
#     highest_similarity = -1
#     best_match = None
#     image_embedding = image_embedding.cpu().numpy().reshape(1, -1)

#     for card_name, card_embedding in tarot_embeddings.items():
#         card_embedding = torch.tensor(card_embedding).reshape(1, -1).numpy()
#         similarity = cosine_similarity(image_embedding, card_embedding)[0][0]
#         if similarity > highest_similarity:
#             highest_similarity = similarity
#             best_match = card_name

#     return best_match, highest_similarity

# Function to merge overlapping or nearby contours
def merge_contours(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = []

    for box in bounding_boxes:
        x, y, w, h = box
        overlap = False

        # Check if the box overlaps with an existing box
        for i, merged in enumerate(merged_boxes):
            mx, my, mw, mh = merged
            if (x < mx + mw and x + w > mx and y < my + mh and y + h > my):
                # Merge the boxes
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged_boxes[i] = (nx, ny, nw, nh)
                overlap = True
                break

        if not overlap:
            merged_boxes.append((x, y, w, h))

    return merged_boxes

# Detect and crop cards from the spread
def detect_and_crop_cards(uploaded_image):
    # Convert PIL image to OpenCV format
    image = np.array(uploaded_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding for binary segmentation
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilate to close gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Display binary mask
    st.image(dilated, caption="Binary Mask (Debug)", use_container_width=True)

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
    st.image(debug_image_filtered, caption="Filtered Contours (Debug)", channels="BGR", use_container_width=True)

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
    st.image(debug_image_merged, caption="Merged Bounding Boxes (Debug)", channels="BGR", use_container_width=True)

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

# Streamlit UI
st.title("Tarot Card Spread Identifier")
st.write("Upload an image of a tarot card spread to identify individual cards.")

# Upload image
uploaded_file = st.file_uploader("Choose a tarot card spread image", type=["jpg", "png"])

if uploaded_file is not None:
    # Load resources
    tarot_embeddings, model, preprocess, device = load_resources()

    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Spread", use_container_width=True)

    # Detect and crop cards
    st.write("Detecting cards in the spread...")
    cropped_cards = detect_and_crop_cards(uploaded_image)

    if not cropped_cards:
        st.write("No cards detected. Please try another image.")
    else:
        # Display and process each cropped card
        for i, card_image in enumerate(cropped_cards):
            st.image(card_image, caption=f"Cropped Card {i + 1}", use_container_width=True)

            # Process the card with CLIP
            card_tensor = tarot.process_image(card_image, preprocess, device)
            with torch.no_grad():
                image_embedding = model.encode_image(card_tensor)

            card_name, similarity_score = tarot.identify_card(image_embedding, tarot_embeddings)
            st.write(f"**Card {i + 1}: {card_name}** (Similarity Score: {similarity_score:.2f})")
