# Object Detection using Pretrained YOLO Model
# Easy Python Code

# Install first:
# pip install ultralytics opencv-python matplotlib

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")   # lightweight pretrained model
image_path = "trafficdl.png"   # replace with your image
results = model(image_path)

print("\nDetected Objects:\n")

for r in results:
    boxes = r.boxes

    for box in boxes:

        # Confidence score
        confidence = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Object name
        object_name = model.names[class_id]

        print(f"Object: {object_name}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 30)


annotated_frame = results[0].plot()
plt.imshow(annotated_frame)
plt.axis("off")
plt.title("Object Detection Result")
plt.show()