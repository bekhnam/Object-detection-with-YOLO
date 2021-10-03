import argparse
import os 
import scipy.io
import scipy.misc

from PIL import Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model
from utils import *

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    box_scores = box_confidence * box_class_probs
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)

    # create filtering mask
    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

# def iou(box1, box2):
#     (box1_x1, box1_y1, box1_x2, box1_y2) = box1
#     (box2_x1, box2_y1, box2_x2, box2_y2) = box2

#     # calculate coordinates of the intersection
#     xi1 = max(box1_x1, box2_x1)
#     yi1 = max(box1_y1, box2_y1)
#     xi2 = min(box1_x2, box2_x2)
#     yi2 = min(box1_y2, box2_y2)

#     inter_width = xi2 - xi1
#     inter_height = yi2 - yi1
#     inter_area = max(inter_width, 0) * max(inter_height,0)

#     # calculate the union area
#     box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
#     box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

#     union_area = box1_area + box2_area - inter_area

#     iou = inter_area / union_area

#     return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners"""
    box_mins = box_xy - (box_wh / 2.0)
    box_maxes = box_xy + (box_wh / 2.0)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2], #y_min
        box_mins[..., 0:1], #x_min
        box_maxes[..., 1:2], #y_max
        box_maxes[..., 0:1]
    ])

def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=score_threshold)
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

def yolo_head(feats, anchors, num_classes):
    """Convert final layer features to bounding box parameters"""
    num_anchors = len(anchors)
    # reshape to batch, height, width, num_anchors, box_params
    anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
    # dynamic implementation of conv dims for fully convolutional model
    conv_dims = K.shape(feats)[1:3]
    # in YOLO the height index is the inner most iteration
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
    conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes+5])
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.softmax(feats[..., 5:])

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs

# define classes, anchors and image shape
class_names = read_classes('model_data/coco_classes.txt')
anchors = read_anchors('model_data/yolo_anchors.txt')
model_image_size = (608, 608) # same as yolo_model input layer size

# load a pre-trained Model
yolo_model = load_model('model_data/', compile=False)
yolo_model.summary()

# run the YOLO on an image
def predict(image_file):
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608,608))

    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))

    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5)

    print('Found {} boxes for {}'.format(len(out_boxes), 'images/' + image_file))

    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    image.save(os.path.join('out', image_file), quality=100)
    output_image = Image.open(os.path.join('out', image_file))
    output_image.show()

    return out_scores, out_boxes, out_classes

# construct the argument parser 
ap = argparse.ArgumentParser()
ap.add_argument("--image", type=str,
    default='test.jpg',
    help='path to input image')
args = ap.parse_args()

out_scores, out_boxes, out_classes = predict(args.image)
