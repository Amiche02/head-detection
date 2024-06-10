import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections, BoundingBoxAnnotator, LabelAnnotator, plot_image
from PIL import Image

# model
save_dir = "model_save"

os.makedirs(save_dir, exit_ok=True)

model_path = hf_hub_download(repo_id="konthee/YOLOv9-Head-Detection", filename="yolov9c_best.pt", cash_dir=save_dir)

model = YOLO(model_path)

image_path = "../Data/test_data/test4.jpg"
image = Image.open(image_path)
output = image(image)
results = Detections.from_ultralytics(output[0])

bounding_box_annotator = BondingBoxAnnotator()
label_annotator = LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image, detections=results)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=results)

# display annotated image
plot_image(annotated_image)