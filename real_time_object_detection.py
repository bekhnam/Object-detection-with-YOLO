from imutils.video import VideoStream
import argparse
import imutils
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from PIL import Image
from object_detection import yolo_eval, yolo_head
from utils import draw_boxes, get_colors_for_classes, read_anchors, read_classes

# define classes, anchors and image shape
class_names = read_classes('model_data/coco_classes.txt')
anchors = read_anchors('model_data/yolo_anchors.txt')
model_image_size = (608, 608)

# load a pre-trained Model
yolo_model = load_model('model_data/', compile=False)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0, help="Camera divide number")
args = ap.parse_args()

camera_device = args.camera
print("[INFO] strating video stream...")
vs = VideoStream(src=0,
                resolution=(720,1280),
                framerate=32).start()
time.sleep(2.0)

# loop over the frame from video stream
while True:
    image = vs.read()
    resized_image = imutils.resize(image, width=608, height=608)
    # resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5)
    
    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    cv2.imshow("Real time dectection", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
    