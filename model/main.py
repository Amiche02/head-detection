from inference import get_model
import supervision as sv
import cv2

image_file = "../Data/test_data/test4.jpg"
image = cv2.imread(image_file)

model = get_model("facial-head-detection/5")
results = model.infer(image)

detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

# Create supervision annotator
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Annotate image
annotated_image = bounding_box_annotator.annotate(scene = image, detections=detections)
annotated_image = label_annotator.annotate(scene = annotated_image, detections=detections)

sv.plot_image(annotated_image)
