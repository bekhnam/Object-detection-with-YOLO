import imghdr
import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import backend as K

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def scale_boxes(boxes, image_shape):
    """Scales the predicted boxes in order to be drawable on the image"""
    height = float(image_shape[0])
    width = float(image_shape[1])
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def get_colors_for_classes(num_classes):
    if (hasattr(get_colors_for_classes, "colors") and
        len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors
    
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    get_colors_for_classes.colors = colors
    return colors

def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')
    )
    thickness = (image.size[0] + image.size[1]) // 300
    colors = get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]

        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)
        
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top+0.5).astype('int32'))
        left = max(0, np.floor(left+0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom+0.5).astype('int32'))
        right = min(image.size[0], np.floor(right+0.5).astype('int32'))
        print(label, (left,top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left+i, top+i, right-i, bottom-i], outline=colors[c])
        
        draw.rectangle([tuple(text_origin), tuple(text_origin+label_size)], outline=colors[c])
        draw.text(text_origin, label, fill=(0,0,0), font=font)
        del draw
    return np.array(image)

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) # add batch dimension
    return image, image_data