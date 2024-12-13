from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# # Train the model
train_results = model.train(
    data="tarot.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
#results = model("datasets/tarot/images/train/10_of_coins.jpg")
# results = model("datasets/tarot/images/train/2_of_cups.jpg")
#results = model("datasets/tarot/images/train/3_of_wands.jpg")

#results = model("attic/testing_20241209/images/knight.jpg")
#results = model("attic/testing_20241209/images/knight2.jpg")
#results = model("attic/testing_20241209/images/comet.jpg")
#results = model("attic/testing_20241209/images/euryale.jpg")

# results[0].show()

# for result in results:
#     print(dir(result))
#     # print(result.boxes)  # Boxes object for bounding box outputs
#     # print(result.masks)  # Masks object for segmentation masks outputs
#     # print(result.keypoints)  # Keypoints object for pose outputs
#     # print(result.probs)  # Probs object for classification outputs
#     # print(result.obb)  # Oriented boxes object for OBB outputs	
#     # print(result.to_json())
#     print(result.to_df())

## Export the model to ONNX format
#path = model.export(format="onnx")  # return path to exported model
