from imutils.video import VideoStream
import argparse
import imutils
import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from PIL import ImageFont, ImageDraw
from object_detection import yolo_eval, yolo_head
from utils import draw_boxes, get_colors_for_classes, read_anchors, read_classes

def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.shape[1] + 0.5).astype('int32')
    )
    thickness = (image.shape[0] + image.shape[1]) // 300
    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]

        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)
        
        top, left, bottom, right = box
        top = max(0, np.floor(top+0.5).astype('int32'))
        left = max(0, np.floor(left+0.5).astype('int32'))
        bottom = min(image.shape[1], np.floor(bottom+0.5).astype('int32'))
        right = min(image.shape[0], np.floor(right+0.5).astype('int32'))
        print(label, (left,top), (right, bottom))

        cv2.rectangle(image, (left, top), (right, bottom), colors[c], 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, label, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c], 2)
    return np.array(image)

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
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# loop over the frame from video stream
while True:
    _, image = vs.read()
    resized_image = cv2.resize(image, (608, 608))
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.shape[0], image.shape[1]], 10, 0.3, 0.5)
    
    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    cv2.imshow("Real time dectection", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()