from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="tarot.yaml",  # path to dataset YAML    
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("datasets/images/val/2055e471bde4c527c0c3ede51853c9f2.jpg")

results[0].show()

for result in results:
    print(result.to_df())


## Export the model to ONNX format
#path = model.export(format="onnx")  # return path to exported model
