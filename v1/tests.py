
import tarot


if __name__ == "__main__":
    
    # File paths
    test_image_path = "../data/images/2_of_cups.jpg"  # Replace with your test image file name
    #test_image_path = "../data/tests/20241130_135912.jpg"

    tarot.init()

    # Process the test image
    image_embedding = tarot.process_and_encode_image(file_path=test_image_path)

    # Identify the card
    card_name, similarity_score = tarot.identify_card(image_embedding)

    # Print the results
    print(f"Identified Card: {card_name}")
    print(f"Similarity Score: {similarity_score}")


    # Detect cards
    img = tarot.open_image_from_file(test_image_path)
    cropped_cards = tarot.detect_and_crop_cards(img)

    for i, card_image in enumerate(cropped_cards):
        print(f"Cropped Card {i + 1}")

        # Process the test image
        image_embedding = tarot.process_and_encode_image(image=card_image)

        # Identify the card
        card_name, similarity_score = tarot.identify_card(image_embedding)

        # Print the results
        print(f"Identified Card: {card_name}")
        print(f"Similarity Score: {similarity_score}")
