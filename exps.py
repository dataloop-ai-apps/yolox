import numpy as np
import torch
import torch.nn as nn
import os

from yolox.exp import Exp as MyExp
from yolox.data.datasets import COCODataset
from yolox.evaluators import COCOEvaluator
from yolox.utils import (gather, is_main_process, postprocess, synchronize, time_synchronized)
from yolox.data import TrainTransform

import itertools
from collections import ChainMap, defaultdict
from tqdm import tqdm
import time


###########################
# EXPERIMENTS TO INHERIT #
###########################

class Yolov3Exp(MyExp):
    def __init__(self):
        super(Yolov3Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = "Yolov3-Exp"

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOFPN, YOLOXHead
            backbone = YOLOFPN()
            head = YOLOXHead(self.num_classes, self.width, in_channels=[128, 256, 512], act="lrelu")
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model


class TinyExp(MyExp):
    def __init__(self):
        super(TinyExp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = "Tiny-yolox-Exp"
        self.enable_mixup = False


class NanoExp(MyExp):
    def __init__(self):
        super(NanoExp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = "Nano-yolox-Exp"

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model


class SmallExp(MyExp):
    def __init__(self):
        super(SmallExp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "Small-yolox-Exp"


class MediumExp(MyExp):
    def __init__(self):
        super(MediumExp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = "Medium-yolox-Exp"


class LargeExp(MyExp):
    def __init__(self):
        super(LargeExp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = "Large-yolox-Exp"


class XLargeExp(MyExp):
    def __init__(self):
        super(XLargeExp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = "XLarge-yolox-Exp"


#############################
# DYNAMIC CLASS FOR INHERIT #
#############################

class DynamicClassCreator:
    def __init__(self, class_name):
        # Ensure a class name is provided
        if not class_name:
            raise ValueError("Invalid class name")

        # Retrieve the actual base class based on its name
        base_class = globals().get(class_name)

        if not base_class:
            raise ValueError(f"Base class '{class_name}' not found.")

        # Define an empty dictionary to store attributes and methods
        class_dict = {}

        # Iterate over attributes and methods of the base class
        for name, value in vars(base_class).items():
            # Exclude special methods and attributes
            if not name.startswith('__'):
                class_dict[name] = value

        # Create the dynamic class using the type() function
        dynamic_class = type(class_name, (base_class,), class_dict)

        # Assign the dynamic class to an attribute of the creator instance
        setattr(self, class_name, dynamic_class)


#############################
# CLASSES FOR DtlExp TO USE #
#############################
class DtlEvaluator(COCOEvaluator):
    def evaluate(
            self, model, distributed=False, half=False, trt_file=None,
            decoder=None, test_size=None, return_outputs=False
    ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
                progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            # different process/device might have different speed,
            # to make sure the process will not be stucked, sync func is used here.
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()

        if return_outputs:
            return eval_results, output_data
        return eval_results


class DtlDataset(COCODataset):
    def load_anno_from_ids(self, id_):
        img = self.coco.loadImgs(id_)[0]
        width = img["width"]
        height = img["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            img["file_name"]
            if "file_name" in img
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)


if __name__ == '__main__':
    exp_name = "XLargeExp"

    # Create an instance of DynamicClassCreator
    dynamic_creator = DynamicClassCreator(exp_name)

    # Access the dynamically created class
    DynamicClassObject = getattr(dynamic_creator, exp_name)

    # Instantiate the dynamically created class
    instance = DynamicClassObject()
