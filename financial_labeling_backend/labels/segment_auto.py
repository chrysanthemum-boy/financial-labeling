import argparse
import json

import cv2
import math
import numpy as np
from numpy import array
import onnxruntime as rt
import os
import time
from tqdm import tqdm


class LetterBox:
    """
    前处理步骤:resize图像大小和边界填充
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not self.scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(
                dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'],
                                   (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


class non_max_suppression:
    """
    后处理步骤:非极大值抑制
    """

    def __init__(self, conf_thres=0.25, iou_thres=0.45, nc=0,  # number of classes (optional)
                 agnostic=False, multi_label=False, max_det=300,
                 max_time_img=0.05, max_nms=30000, max_wh=7680):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.nc = nc
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.max_time_img = max_time_img
        self.max_nms = max_nms
        self.max_wh = max_wh

    def __call__(self, prediction):
        # assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        # assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'
        # if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        #     prediction = prediction[0]  # select only inference output
        self.prediction = prediction
        bs = self.prediction.shape[0]  # batch size
        nc = self.nc or (self.prediction.shape[1] - 4)  # number of classes
        nm = self.prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = self.prediction[:, 4:mi].max(1) > self.conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + self.max_time_img * bs  # seconds to quit after
        self.multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        output = [np.zeros((0, 6 + nm), dtype=np.uint8)] * bs
        for xi, x in enumerate(self.prediction):  # image index, image inference
            x = x.transpose(1, 0)[xc[xi]]  # confidence

            box, cls, mask = x[:, :4], x[:, 4:nc+4], x[:, -nm:]
            # center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(box)
            if self.multi_label:
                i, j = (cls > self.conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate(
                    (box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1).reshape(cls.shape[0], 1), cls.argmax(
                    1).reshape(cls.shape[0], 1)
                x = np.concatenate((box, conf, j.astype(np.float64), mask), 1)[
                    conf.reshape(-1) > self.conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[(-x[:, 4]).argsort()[:self.max_nms]]

            # Batched NMS
            c = x[:, 5:6] * (0 if self.agnostic else self.max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self.numpy_nms(boxes, scores, self.iou_thres)  # NMS
            i = i[:self.max_det]  # limit detections

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded
        return output

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    def box_area(self, boxes: array):
        """
        :param boxes: [N, 4]
        :return: [N]
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, box1: array, box2: array):
        """
        :param box1: [N, 4]
        :param box2: [M, 4]
        :return: [N, M]
        """
        area1 = self.box_area(box1)  # N
        area2 = self.box_area(box2)  # M
        # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    def numpy_nms(self, boxes: array, scores: array, iou_threshold: float):
        idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
        keep = []
        while idxs.size > 0:  # 统计数组中元素的个数
            max_score_index = idxs[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)

            if idxs.size == 1:
                break
            idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
            other_boxes = boxes[idxs]  # [?, 4]
            ious = self.box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
            idxs = idxs[ious[0] <= iou_threshold]

        keep = np.array(keep)
        return keep


class process_mask:
    """
    后处理步骤:上采样还原mask大小
    """

    def __init__(self, shape, upsample=False) -> None:
        self.shape = shape
        self.upsample = upsample

    def __call__(self, protos, masks_in, bboxes, *args, **kwds):
        c, mh, mw = protos.shape  # CHW
        ih, iw = self.shape

        sigmoid_masks = []
        for i in range(masks_in.shape[0]):
            sigmoid_masks.append(self.sigmoid_function(
                (masks_in @ protos.astype(np.float64).reshape(c, -1))[i]))

        masks = np.array(sigmoid_masks).reshape(-1, mh, mw)  # CHW

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = self.crop_mask(masks, downsampled_bboxes)  # CHW
        if self.upsample:
            masks = (masks*255).astype(np.uint8)
            masks = masks.transpose(1, 2, 0)
            masks = cv2.resize(masks, self.shape)
            masks[masks <= (255*0.5)] = 0.0
            masks[masks > (255*0.5)] = 1.0
        return masks

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def crop_mask(self, masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.array_split(
            boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.array(range(w), dtype=np.float64).reshape(
            1, 1, -1)  # rows shape(1,w,1)
        c = np.array(range(h), dtype=np.float64).reshape(
            1, -1, 1)  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class scale_boxes:
    def __init__(self, img1_shape, ratio_pad=None):
        """
        后处理步骤：缩放矩形框
        """
        self.img1_shape = img1_shape
        self.ratio_pad = ratio_pad

    def __call__(self, img0_shape, boxes, *args, **kwds):
        self.img0_shape = img0_shape
        if self.ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(self.img1_shape[0] / self.img0_shape[0],
                       self.img1_shape[1] / self.img0_shape[1])
            pad = (self.img1_shape[1] - self.img0_shape[1] * gain) / \
                2, (self.img1_shape[0] - self.img0_shape[0]
                    * gain) / 2  # wh padding
        else:
            gain = self.ratio_pad[0][0]
            pad = self.ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain

        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(
            0, self.img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(
            0, self.img0_shape[0])  # y1, y2
        return boxes


class Colors:
    """
    保存结果时矩形框和masks的颜色,每种类别一个颜色
    """

    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class save_img:
    def __init__(self, save_masks=False, save_box=False, cls_name=None):
        self.save_masks = save_masks
        self.save_box = save_box
        self.name = cls_name

    def __call__(self, save_path, masks, boxes, scores, cls,
                 im_gpu, im_shape, im, colors):
        self.masks,self.im_gpu,self.im_shape,self.im=masks,im_gpu, im_shape, im
        self.boxes, self.scores, self.cls, self.colors = boxes, scores, cls, colors
        if self.save_masks and self.save_box:
            masks_write = self.masks_write(
                masks=masks, im_gpu=im_gpu, im_shape=im_shape, im=im.copy())
            # ,self.box_list,self.scores_list))
            cv2.imwrite(save_path, self.box_write(masks_write.copy()))
        elif self.save_masks:  # 只保存掩码
            cv2.imwrite(save_path, self.masks_write(
                masks=masks, im_gpu=im_gpu, im_shape=im_shape, im=im.copy()))
        elif self.save_box:  # 只保存矩形框
            # ,self.box_list,self.scores_list))
            cv2.imwrite(save_path, self.box_write(self.im.copy()))
        else:
            cv2.imwrite(save_path, self.im.copy())

    def box_write(self, image):  # ,box_list,scores_list
        """
        输出结果中保存矩形框
        """
        box_write = image.copy()
        for box, scores, cls, colors in zip(self.boxes, self.scores, self.cls, self.colors):
            cx = int(box[0])  # np.mean([int(box[0]), int(box[2])])
            cy = int(box[1])  # np.mean([int(box[1]), int(box[3])])
            box_write = cv2.rectangle(box_write, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), color=colors, thickness=2)

            mess = str(self.name[int(cls)])+':%.2f' % scores
            h, w = image.shape[:2]
            retval, _ = cv2.getTextSize(mess, 0, 1e-3 * h, 1)  # 计算文本的宽和高
            box_write = cv2.rectangle(box_write, (int(box[0]), max(int(box[1]-retval[1]), 0)), (min(int(
                box[0]+retval[0]), w), int(box[1])), color=colors, thickness=-1)  # 画文本的背景填充框
            cv2.putText(box_write, mess, (int(cx), int(cy)),  # 画文本内容（类别：置信度）
                        0, 1e-3 * h, (255, 255, 255), 1)
        return box_write

    def masks_write(self, masks, im_gpu, im_shape, im, alpha=0.5, retina_masks=False):
        """
        输出结果中保存masks
        """
        colors = np.array(self.colors, dtype=np.float32)/255.0*alpha
        if len(masks.shape) == 3:
            masks = masks.transpose(2, 0, 1)[:, :, :, None]  # shape(n,h,w,1)
        else:
            masks = masks[None, :, :, None]
        masks_color = [masks[i] * colors[i]
                       for i in range(masks.shape[0])]  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(0) * \
            2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu[::-1, :, :]
        im_gpu = im_gpu.transpose(1, 2, 0)  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255)
        im_mask_np = im_mask
        im[:] = im_mask_np if retina_masks else self.scale_image(
            im_gpu.shape, im_mask_np, im_shape)
        # 画置信度
        for box, scores, cls, colors in zip(self.boxes, self.scores, self.cls, self.colors):
            cx = int(box[0])  # np.mean([int(box[0]), int(box[2])])
            cy = int(box[1])  # np.mean([int(box[1]), int(box[3])])
            mess = str(self.name[int(cls)])+':%.2f' % scores
            h, w = im.shape[:2]
            retval, _ = cv2.getTextSize(mess, 0, 1e-3 * h, 1)  # 计算文本的宽和高
            im = cv2.rectangle(im, (int(box[0]), max(int(box[1]-retval[1]), 0)), (min(int(
                box[0]+retval[0]), w), int(box[1])), color=colors, thickness=-1)  # 画文本的背景填充框
            cv2.putText(im, mess, (int(cx), int(cy)), 0,  # 画文本内容（类别：置信度）
                        1e-3 * h, (255, 255, 255), 1)
        return im

    def scale_image(self, im1_shape, masks, im0_shape, ratio_pad=None):
        """
        输出结果中保存masks时缩放图像
        """
        # Rescale coordinates (xyxy) from im1_shape to im0_shape
        if ratio_pad is None:  # calculate from im0_shape
            # gain  = old / new
            gain = min(im1_shape[0] / im0_shape[0],
                       im1_shape[1] / im0_shape[1])
            pad = (im1_shape[1] - im0_shape[1] * gain) / \
                2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks


class Segmentation_inference:
    def __init__(self, model_path, device, conf_thres=0.25, iou_thres=0.7, imgsz=640,
                 save_masks=False, save_box=False, show_time=False) -> None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else [
            'CPUExecutionProvider']
        self.sess = rt.InferenceSession(
            model_path, providers=providers)  # 实例化推理类
        self.input_name = self.sess.get_inputs()[0].name  # ，模型输入
        self.out_name = [
            output.name for output in self.sess.get_outputs()]  # 模型输出
        self.name = eval(self.sess.get_modelmeta(
        ).custom_metadata_map['names'])  # 类别名，种类名
        self.LetterBox = LetterBox(new_shape=(
            imgsz, imgsz))  # 前处理，resize和填充矩形框
        self.NMS = non_max_suppression(
            conf_thres=conf_thres, iou_thres=iou_thres, nc=len(self.name))  # 后处理，极大值抑制
        self.Process_Mask = process_mask(
            shape=(imgsz, imgsz), upsample=True)  # 后处理，上采样还原掩码大小
        self.Scale_Boxes = scale_boxes(img1_shape=(imgsz, imgsz))  # 后处理，缩放矩形框
        self.save_img = save_img(
            save_masks=save_masks, save_box=save_box, cls_name=self.name)  # 保存输出结果
        self.show_time = show_time  # 是否打印运行时间

    def __call__(self, *args, **kwds):
        self.im = cv2.imdecode(np.fromfile(kwds['path'], dtype=np.int8), 1)
        # 前处理
        time1 = time.time()
        self.im0 = self.img_pretreatment(im=self.im)
        time2 = time.time()

        # 推理
        time3 = time.time()
        self.preds = self.sess.run(
            self.out_name, {self.input_name: [self.im0]})
        time4 = time.time()

        # 后处理
        time5 = time.time()
        self.masks, self.box_list, self.scores_list, self.cls = self.img_reprocessing(
            preds=self.preds)
        time6 = time.time()

        # 打印处理时间及保存结果
        if self.show_time:  # 打印运行时间
            print(f'\npretreatment——time:{(time2-time1)*1000}ms')
            print(f'inference——time:{(time4-time3)*1000}ms')
            print(f'reprocessing——time:{(time6-time5)*1000}ms')
            print('-----------------------------')

        if type(self.masks) == type(None):  # 如果没有目标，则保存原图
            cv2.imwrite(kwds['save_path'], self.im)
            return None

        colors = Colors()
        self.colors = [colors(x, True) for x in self.cls]  # 每一种类别一个颜色
        self.save_img(save_path=kwds['save_path'], masks=np.array(  # 保存结果
            self.masks), boxes=self.box_list, scores=self.scores_list, cls=self.cls,
            im_gpu=self.im0, im_shape=self.im.shape, im=self.im.copy(),
            colors=self.colors)
        return True

    def img_pretreatment(self, im):
        """
        前处理
        """
        im1 = self.LetterBox(image=im)
        im2 = im1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im3 = np.ascontiguousarray(im2)  # contiguous
        im4 = im3.astype(np.float32)/255.0
        return im4

    def img_reprocessing(self, preds):
        """
        后处理
        """
        p = self.NMS(prediction=preds[0])
        if len(p[0]) == 0:
            return (None, None, None, None)
        proto = self.preds[1][-1] if len(self.preds[1]) == 3 else self.preds[1]
        for i, pred in enumerate(p):
            masks = self.Process_Mask(
                protos=proto[i], masks_in=pred[:, 6:], bboxes=pred[:, :4])
            pred[:, :4] = self.Scale_Boxes(
                img0_shape=self.im.shape, boxes=pred[:, :4])
        return masks, pred[:, :4], pred[:, 4], pred[:, 5]


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default='test_img.jpg', help='Image Path or Image dir path')  # 预测路径
    parser.add_argument('--save_path', type=str,
                        default='result_img.jpg', help='result Image save Path or dir path')  # 保存路径
    parser.add_argument('--weight', type=str,
                        default='test.onnx', help='weights path')  # 模型路径
    parser.add_argument('--imgsz', type=int,
                        default=640, help='Input Image Size')  # 模型输入图片大小
    parser.add_argument('--device', type=str, default='cpu',
                        help='Hardware devices')  # 使用cpu还是GPU
    parser.add_argument('--save_masks', action='store_true',
                        default=False, help='save the mask?')  # 结果是否保存masks
    parser.add_argument('--save_box', action='store_true',
                        default=False, help='save the box?')  # 结果是否保存box
    parser.add_argument('--show_time', action='store_true',
                        default=False, help='Output processing time')  # 是否显示预处理和推理时间
    parser.add_argument('--conf_thres', type=float,
                        default=0.25, help='Confidence level threshold')  # 置信度阈值
    parser.add_argument('--iou_thres', type=float,
                        default=0.7, help='IOU threshold')  # IOU阈值
    return parser.parse_args()


def main(opt):
    calculate = Segmentation_inference(opt.weight, opt.device,  # 模型路径和是否使用GPU
                                       conf_thres=opt.conf_thres,  # 置信度阈值,小于阈值的预测结果会被删除
                                       iou_thres=opt.iou_thres,  #  IOU阈值，大于阈值的预测结果会被删除
                                       imgsz=opt.imgsz,  # 输入图片的大小
                                       save_masks=opt.save_masks,  # 是否保存masks
                                       save_box=opt.save_box,  # 是否保存box
                                       show_time=opt.show_time,  # 是否打印预处理和推理时间
                                       )

    inference_images_Path = []  # 需要预测的图片路径列表
    save_images_path = []  # 保存的图片路径列表
    if not os.path.isdir(opt.path):  # 检测单张图片
        assert not os.path.isdir(
            opt.save_path), '预测路径为图片名，则保存路径也应该指定为文件名，而不是目录(加上后缀!)'
        inference_images_Path.append(opt.path)
        save_images_path.append(opt.save_path.split('.')[0]+'.jpg')
    else:  # 检测一个文件夹中的图片
        assert os.path.isdir(
            opt.save_path), '预测路径为文件夹目录，则保存路径也应该指定为已存在的某个文件夹目录(目录路径一定要存在!)'
        for img_path in os.listdir(opt.path):
            inference_images_Path.append(os.path.join(opt.path, img_path))
            save_images_path.append(os.path.join(opt.save_path, img_path))
    for path, save_path in zip(tqdm(inference_images_Path), save_images_path):
        try:
            assert type(calculate(path=path,  # 预测路径
                        save_path=save_path,  # 保存路径
                                  )) != type(None), f'\n{path}:没有检测到目标,请适当调低阈值'
        except BaseException as e:
            print(e)


if '__main__' == __name__:
    # opt = parse_opt()
    # main(opt)
    # 原始的文本列表
    text_list = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # 颜色列表
    hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    import random

    # 生成80个不同的随机颜色
    random_colors = []
    for _ in range(80):
        # 生成随机的RGB颜色值
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)

        # 将RGB颜色值转换为十六进制字符串
        hex_color = '{:02x}{:02x}{:02x}'.format(red, green, blue)

        random_colors.append(hex_color)

    # 输出随机颜色
    for color in random_colors:
        print("#" + color)
    # 构建新的列表
    new_list = []
    for i, text in enumerate(text_list):
        new_item = {
            "text": text,
            "suffix_key": "c",
            "background_color": "#" + random_colors[i],
            "text_color": "#ffffff"
        }
        new_list.append(new_item)

    # 输出新列表
    for item in new_list:
        print(item)

    # 将列表转换为 JSON 格式
    json_data = json.dumps(new_list, indent=4)

    # 将 JSON 数据写入文件
    with open('auto_models/classes/output.json', 'w') as f:
        f.write(json_data)
