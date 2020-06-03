# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on images.
"""

import colorsys
import os
import random
import numpy as np
from keras import backend as K
from keras.models import load_model
from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image


class YOLO(object):
    
    _defaults = {
        "model_path": 'model_data/yolov3_pes.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/pes_classes.txt',
        "score": 0.5,
        "iou": 0.5,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()       
        
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(",")]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(".h5"), "Keras model must be a .h5 file."

        self.yolo_model = load_model(model_path, compile=False)
        print("{} model, anchors, and classes loaded.".format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        """
        Perform pedestrian detection in image

        Parameters:
        -----------
            image: input image to perform detection

        Return:
        -------
            return_boxs: output bounding boxes in tlwh (top, left, width, height) format
            return_scores: scores associated with bounding boxes

        """

        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return_boxs = []
        return_scores = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != "person":
                continue
            box = out_boxes[i]
            x = int(box[1])
            y = int(box[0])  
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0 
            return_boxs.append([x, y, w, h])
            return_scores.append(out_scores[i])

        return return_boxs, return_scores

    def close_session(self):
        self.sess.close()


if __name__ == "__main__":
    pass