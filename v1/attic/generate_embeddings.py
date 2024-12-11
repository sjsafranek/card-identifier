import clip
import torch
from PIL import Image
import os
import json

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to the tarot card images
tarot_images_path = "./tarot_images"  # Change this to the folder where your images are stored
tarot_embeddings = {}

# Process each tarot card image
for filename in os.listdir(tarot_images_path):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(tarot_images_path, filename)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().numpy().tolist()
        card_name = os.path.splitext(filename)[0]  # Use the filename (without extension) as the card name
        tarot_embeddings[card_name] = embedding
        print(f"Processed {card_name}")

# Save the embeddings to a JSON file
with open("tarot_embeddings.json", "w") as f:
    json.dump(tarot_embeddings, f)

print("All tarot embeddings have been saved to 'tarot_embeddings.json'")
