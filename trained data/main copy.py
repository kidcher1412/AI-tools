import torch
import cv2
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import display, Image as IPImage

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Define class labels (replace with your own class labels)
class_labels = ['cat', 'chicken', 'cow', 'dog', 'fox',
                'goat', 'horse', 'person', 'racoon', 'skunk']



def detect_objects(image_path):
    image = Image.open(image_path)
    results = model(image)

    # Process and display the results
    pred = results.pred[0]
    for det in pred:
        class_idx, confidence, bbox = det[5], det[4], det[:4]
        class_label = class_labels[int(class_idx)]
        bbox = [round(float(coord), 2) for coord in bbox]
        print(f"Detected {class_label} with confidence {confidence} at {bbox}")

        # Draw bounding box on image
        bbox = [int(coord * image.width) if i % 2 == 0 else int(coord *
                                                                image.height) for i, coord in enumerate(bbox)]
        cv2.rectangle(image, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, f"{class_label}: {confidence:.2f}",
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    image.show()


if __name__ == "__main__":
    image_path = 'testbest.png'  # Path to the input image
    detect_objects(image_path)
