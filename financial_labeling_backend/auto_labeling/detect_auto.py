import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
import onnxruntime
from PIL import Image
# from tkinter import *
# import PIL.Image
import io
import os
import natsort
import requests
import imghdr
# from tkinter import filedialog
# from xml.etree import ElementTree as ET
# from ultralytics import YOLO


class Detector:
    def __init__(self):
        # self.weigth_path=weight_path
        # current_script_path = os.path.dirname(os.path.abspath(__file__))
        # # 构建权重文件的完整路径
        # self.full_weight_path = os.path.join(current_script_path, weight_path)
        self.data = r""  # 选择你的配置文件(一般为.txt)
        self.weights = r""  # 可去官方下载v5 6.0的预训练权重模型

        self.img = None
        cudnn.benchmark = True
        self.track = True

        self.objectList = []
        self.h = None
        self.w = None
        self.predefined_classes = []
        self.imgdir = r""  # 你需要标注的图片文件夹
        self.outdir = r""  # 你需要保存的xml文件夹
        self.detect_class = r""  # 你需要自动标注的类型
        self.root_window = None
        self.flag = False

    def init_model(self):
        print(self.weights)
        self.model = YOLO(self.weights)

    @torch.no_grad()
    def run(self, image_path):
        results = []
        results_yolo = self.model(image_path)
        for r in results_yolo:
            tmp_r = r.boxes
            for xyxy, conf, cls in zip(tmp_r.xyxy, tmp_r.conf, tmp_r.cls):
                # 将Tensor转换为标准Python类型
                xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(conf)
                class_pred = int(cls)

                results.append([xmin, ymin, xmax, ymax, class_pred])

        # self.draw_boxes(image_path, results)
        return results

        # infor_list=[]
        #         info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls)]
        #         info_list.append(info)
        #     # x1,y1=int(info_list[0][0]),int(info_list[0][1])
        #     # x2,y2=int(info_list[0][2]),int(info_list[0][3])
        #     return info_list
        # else:
        #     return None

    def create_annotation(self, xn):
        global annotation
        tree = ET.ElementTree()
        tree.parse(xn)
        annotation = tree.getroot()

    # 遍历xml里面每个object的值如果相同就不插入
    def traverse_object(self, AnotPath):
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object')
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            x1 = int(BndBox.find('xmin').text)
            y1 = int(BndBox.find('ymin').text)
            x2 = int(BndBox.find('xmax').text)
            y2 = int(BndBox.find('ymax').text)
            self.objectList.append([x1, y1, x2, y2, ObjName])

    # 定义一个创建一级分支object的函数
    def create_object(self, root, objl):  # 参数依次，树根，xmin，ymin，xmax，ymax
        # 创建一级分支object
        _object = ET.SubElement(root, 'object')
        # 创建二级分支
        name = ET.SubElement(_object, 'name')
        # print(obj_name)
        name.text = str(objl[4])
        pose = ET.SubElement(_object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(_object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(_object, 'difficult')
        difficult.text = '0'
        # 创建bndbox
        bndbox = ET.SubElement(_object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = '%s' % objl[0]
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = '%s' % objl[1]
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = '%s' % objl[2]
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = '%s' % objl[3]

    # 创建xml文件的函数
    def create_tree(self, image_name, h, w, imgdir):
        global annotation
        # 创建树根annotation
        annotation = ET.Element('annotation')
        # 创建一级分支folder
        folder = ET.SubElement(annotation, 'folder')
        # 添加folder标签内容
        folder.text = (imgdir)

        # 创建一级分支filename
        filename = ET.SubElement(annotation, 'filename')
        filename.text = image_name

        # 创建一级分支path
        path = ET.SubElement(annotation, 'path')

        # path.text = getcwd() + '\{}\{}'.format(imgdir, image_name)  # 用于返回当前工作目录
        path.text = '{}/{}'.format(imgdir, image_name)  # 用于返回当前工作目录

        # 创建一级分支source
        source = ET.SubElement(annotation, 'source')
        # 创建source下的二级分支database
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'

        # 创建一级分支size
        size = ET.SubElement(annotation, 'size')
        # 创建size下的二级分支图像的宽、高及depth
        width = ET.SubElement(size, 'width')
        width.text = str(w)
        height = ET.SubElement(size, 'height')
        height.text = str(h)
        depth = ET.SubElement(size, 'depth')
        depth.text = '3'

        # 创建一级分支segmented
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = '0'

    def pretty_xml(self, element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

    def work(self):
        with open(self.detect_class, "r") as f:  # 打开文件
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                self.predefined_classes.append(line)

        IMAGES_LIST = os.listdir(self.imgdir)
        for image_name in natsort.natsorted(IMAGES_LIST):
            # print(image_name)
            # 判断后缀只处理图片文件
            if image_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                image = cv2.imread(os.path.join(self.imgdir, image_name))
                if image is None:
                    print(image_name + "图像为空请删除")
                    continue
                file_tail = os.path.splitext(image_name)[1]
                # weight_path="best.pt"
                # self.model=attempt_load(weights, map_location="cpu")
                coordinates_list = self.run(os.path.join(self.imgdir, image_name))
                (h, w) = image.shape[:2]
                # xml_name = ('.\{}\{}.xml'.format(outdir, image_name.strip('.jpg')))
                # 分割文件名和后缀
                image_basename, file_extension = image_name.rsplit('.', 1)
                xml_name = os.path.join(self.outdir, image_basename + '.xml')

                if (os.path.exists(xml_name)):
                    self.create_annotation(xml_name)
                    self.traverse_object(xml_name)
                else:
                    self.create_tree(image_name, h, w, self.imgdir)
                if coordinates_list:
                    print(image_name + "已标注完成")
                    for coordinate in coordinates_list:
                        label_id = coordinate[4]
                        if (self.predefined_classes.count(self.predefined_classes[label_id]) > 0):
                            object_information = [int(coordinate[0]), int(coordinate[1]), int(coordinate[2]),
                                                  int(coordinate[3]), self.predefined_classes[label_id]]
                            if (self.objectList.count(object_information) == 0):
                                self.create_object(annotation, object_information)
                    self.objectList = []
                    # 将树模型写入xml文件
                    tree = ET.ElementTree(annotation)
                    root = tree.getroot()
                    self.pretty_xml(root, '\t', '\n')
                    # tree.write('.\{}\{}.xml'.format(outdir, image_name.strip('.jpg')), encoding='utf-8')
                    tree.write('{}\{}.xml'.format(self.outdir, image_name.strip(file_tail)), encoding='utf-8')
                else:
                    print(image_name)

    # 客户端
    def client(self):
        def creatWindow():
            self.root_window.destroy()
            window()

        def judge(str):
            if (str):
                text = "你已选择" + str
            else:
                text = "你还未选择文件夹，请选择"
            return text

        def test01():
            self.imgdir = r""
            self.imgdir += filedialog.askdirectory()
            creatWindow()

        def test02():
            self.outdir = r""
            self.outdir += filedialog.askdirectory()
            creatWindow()

        def test03():
            self.data = r""
            self.data += filedialog.askopenfilename()
            creatWindow()

        def test04():
            self.weights = r""
            self.weights += filedialog.askopenfilename()
            creatWindow()

        def test05():
            self.detect_class = r""
            self.detect_class += filedialog.askopenfilename()
            creatWindow()

        def tes06():
            self.init_model()
            self.work()
            self.flag = True
            creatWindow()

        def window():
            self.root_window = Tk()
            self.root_window.title("")
            screenWidth = self.root_window.winfo_screenwidth()  # 获取显示区域的宽度
            screenHeight = self.root_window.winfo_screenheight()  # 获取显示区域的高度
            tk_width = 500  # 设定窗口宽度
            tk_height = 400  # 设定窗口高度
            tk_left = int((screenWidth - tk_width) / 2)
            tk_top = int((screenHeight - tk_width) / 2)
            self.root_window.geometry('%dx%d+%d+%d' % (tk_width, tk_height, tk_left, tk_top))
            self.root_window.minsize(tk_width, tk_height)  # 最小尺寸
            self.root_window.maxsize(tk_width, tk_height)  # 最大尺寸
            self.root_window.resizable(width=False, height=False)
            btn_1 = Button(self.root_window, text='请选择你要标注的图片文件夹', command=test01,
                           height=0)
            btn_1.place(x=169, y=40, anchor='w')

            text = judge(self.imgdir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=70, anchor='w')
            btn_2 = Button(self.root_window, text='请选择你要保存的xml文件夹(.xml)', command=test02,
                           height=0)
            btn_2.place(x=169, y=100, anchor='w')
            text = judge(self.outdir)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=130, anchor='w')
            # btn_3 = Button(self.root_window, text='请选择你的配置文件(.txt)', command=test03,
            #                height=0)
            # btn_3.place(x=169, y=160, anchor='w')
            # text = judge(self.data)
            # text_label = Label(self.root_window, text=text)
            # text_label.place(x=160, y=190, anchor='w')

            # if(self.outdir and self.imgdir and self.data):
            btn_4 = Button(self.root_window, text='请选择使用的模型(.pt)', command=test04,
                           height=0)
            btn_4.place(x=169, y=220, anchor='w')
            text = judge(self.weights)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=250, anchor='w')

            btn_5 = Button(self.root_window, text='请选择需要自动标注的类别文件(.txt)', command=test05,
                           height=0)
            btn_5.place(x=169, y=280, anchor='w')
            text = judge(self.detect_class)
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=310, anchor='w')

            btn_6 = Button(self.root_window, text='开始自动标注', command=tes06,
                           height=0)
            btn_6.place(x=169, y=340, anchor='w')
            if (self.flag):
                text = "标注完成"
            else:
                text = "等待标注"
            text_label = Label(self.root_window, text=text)
            text_label.place(x=160, y=370, anchor='w')
            self.root_window.mainloop()

        window()


class_names = ['none',
               'text',
               'title',
               'figure',
               'figure_caption',
               'table',
               'table_caption',
               'header',
               'footer',
               'reference',
               'equation']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


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

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
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


def run_detect(model_path, image_dir_path, conf_thres, iou_thres):
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    IMAGES_LIST = os.listdir(image_dir_path)
    imgs = []
    for image_dir in natsort.natsorted(IMAGES_LIST):
        if os.path.isdir(image_dir_path + image_dir):
            image_name = os.listdir(image_dir_path + image_dir)
            if len(image_name):
                image_file = requests.get('http://localhost:8000/media/' + os.path.join(image_dir,image_name[0]))
                imgtype = imghdr.what(None, image_file.content)
                if imgtype in ('jpg', 'png', 'jpeg', 'bmp'):
                    imgs.append(np.array(Image.open(io.BytesIO(image_file.content))))
    res_all = []
    for img in imgs:
        res = []
        boxes, scores, class_ids = yolov8_detector(img)
        for i in range(len(class_ids)):
            res.append([class_ids[i], scores[i], boxes[i]])
        res_all.append(res)
    return res_all

if __name__ == '__main__':
    # detector = Detector()
    # detector.client()
    model_path = "auto_models/models/yanbao_paper30_CDLA-best.onnx"
    image_dir_path = "../media/"

    res = run_detect(model_path,image_dir_path, 0.3, 0.5)
    print(res)
    # img = np.array(Image.open("auto_models/test_image/"))()

    # # Detect Objects
    # res = []
    # boxes, scores, class_ids = yolov8_detector(img)
    # for i in range(len(class_ids)):
    #     res.append([class_ids[i], scores[i], boxes[i]])
    # print("res: {}".format(res))
    # print("boxes: {}".format(boxes))
    # print("scores: {}".format(scores))
    # print("class_ids: {}".format(class_ids))

