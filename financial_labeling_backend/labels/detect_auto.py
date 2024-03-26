import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
import onnxruntime
from PIL import Image
import io
import os
import natsort
import requests
import imghdr
from tkinter import filedialog
from xml.etree import ElementTree as ET

from torch import sigmoid
from ultralytics import YOLO

import hanlp

# HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH,
#                    output_key='ner',input_key='tok')
HanLP = hanlp.pipeline() \
    .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
    # HanLP = hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH')


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    # def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
    #     return draw_detections(image, self.boxes, self.scores,
    #                            self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


def get_mask(row, box, img_width, img_height):
    mask = row.reshape(160, 160)
    x1, y1, x2, y2 = box
    # // box坐标是相对于原始图像大小，需转换到相对于160*160的大小
    mask_x1 = round(x1 / img_width * 160)
    mask_y1 = round(y1 / img_height * 160)
    mask_x2 = round(x2 / img_width * 160)
    mask_y2 = round(y2 / img_height * 160)
    mask = mask[mask_y1:mask_y2, mask_x1:mask_x2]
    mask = sigmoid(mask)
    # // 把mask的尺寸调整到相对于原始图像大小
    mask = cv2.resize(mask, (round(x2 - x1), round(y2 - y1)))
    mask = (mask > 0.5).astype("uint8") * 255
    return mask


def run_detect(model_path, image_file_path, conf_thres, iou_thres):
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    image_file = requests.get('http://localhost:8000/media/' + image_file_path)
    imgtype = imghdr.what(None, image_file.content)
    if imgtype in ('jpg', 'png', 'jpeg', 'bmp'):
        img = np.array(Image.open(io.BytesIO(image_file.content)))
        res = []
        boxes, scores, class_ids = yolov8_detector(img)
        for i in range(len(class_ids)):
            res.append([class_ids[i], scores[i], boxes[i]])
    return res


def run_segment(model_path, image_file_path, conf_thres):
    yolov8_segment = YOLO(model_path)
    image_file = requests.get('http://localhost:8000/media/' + image_file_path)
    imgtype = imghdr.what(None, image_file.content)
    if imgtype in ('jpg', 'png', 'jpeg', 'bmp'):
        img = Image.open(io.BytesIO(image_file.content))
        results = yolov8_segment.predict(img, conf=conf_thres)

    points_list_all = []
    for result in results:
        points_list = []
        for res in result:
            for masks, box in zip(res.masks.xy, res.boxes):
                merge_mask = masks[0].tolist()
                for i in range(1, len(masks)):
                    merge_mask.extend(masks[i].tolist())
                res_class = int(box.cls[0])
                points_list.append([merge_mask, res_class])
        points_list_all.append(points_list)
    return points_list_all


def run_span(text):

    replacements = {
        '\n': '#',
        "(": '#',
        ")": '#',
        "（": '#',
        "）": '#',
        "《": '#',
        "》": '#',
        " ": '#',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    tok_list = HanLP(text)["tok"]
    ner_list = HanLP(text)["ner"]
    start = 0
    end = len(tok_list[0])
    new_tok_list = [{"word": tok_list[0], "start": start, "end": end, "id": 0}]
    i = 0
    for tok in tok_list[1:]:
        start = end
        end = start + len(tok)
        i += 1
        new_tok_list.append({"word": tok, "start": start, "end": end, "id": i})
        # print(f"Word: {tok}, Start: {start}, End: {end}")
    # print(new_tok_list)
    res_list = []
    for ner in ner_list:
        start = new_tok_list[ner[2]]["start"]
        end = new_tok_list[ner[3] - 1]["end"]
        res_list.append([ner[0], ner[1], start, end])
    return res_list


if __name__ == '__main__':
    model_path = "./auto_models/models/yolov8n-seg.onnx"
    image_dir_path = "./auto_models/coco_2017/"

    res = run_segment(model_path, image_dir_path, 0.3)
    print(res)
