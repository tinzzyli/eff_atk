from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2
import torch.nn as nn
import torchvision

import sys
from pathlib import Path

YOLOV5_FILE = Path(f"../model/yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add ROOT to PATH
from models.common import DetectMultiBackend
from utils.general import Profile, non_max_suppression

from PIL import Image
import logging


def create_logger(module, filename, level):
    # Create a formatter for the logger, setting the format of the log with time, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a logger named 'logger_{module}'
    logger = logging.getLogger(f'logger_{module}')
    logger.setLevel(level)     # Set the log level for the logger
    
    # Create a file handler, setting the file to write the logs, level, and formatter
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(level)         # Set the log level for the file handler
    fh.setFormatter(formatter) # Set the formatter for the file handler
    
    # Add the file handler to the logger
    logger.addHandler(fh)
    
    return logger


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def run_attack(outputs,bx, strategy, max_tracker_num, adam_opt):
    outputs = outputs[0][0] # lifang535 add
    
    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    scores = outputs[:,5] * outputs[:,4]
    height = outputs[:,2]
    width = outputs[:,3]

    sel_scores_b = scores[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_scores_m = scores[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_scores_s = scores[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]
    sel_height_b = height[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_height_m = height[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_height_s = height[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]
    sel_width_b = width[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_width_m = width[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_width_s = width[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]


    sel_dets = torch.cat((sel_scores_b, sel_scores_m, sel_scores_s), dim=0)
    sel_height = torch.cat((sel_height_b, sel_height_m, sel_height_s), dim=0)
    sel_width = torch.cat((sel_width_b, sel_width_m, sel_width_s), dim=0)
    sel_aaa = (sel_width/640) * (sel_height/640)
    loss4 = 100*torch.sum(sel_aaa)
    targets = torch.ones_like(sel_dets)
    loss1 = 10*(F.mse_loss(sel_dets, targets, reduction='sum'))
    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores) # lifang535 remove
    # targets = torch.zeros_like(scores) # lifang535 add
    loss3 = 1.0*(F.mse_loss(scores, targets, reduction='sum'))
    # loss = loss1+loss4#+loss3#+loss2 # lifang535 remove
    loss = loss1+loss4+loss3+loss2 # lifang535 add
    
    loss.requires_grad_(True)
    adam_opt.zero_grad()
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    count = (scores > 0.25).sum()
    print('loss',loss.item(),'loss_1',loss1.item(),'loss_2',loss2.item(),'loss_3',loss4.item(),'count:',count.item())
    return bx



class SingleAttack:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, image_list, image_name_list, img_size): # args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.image_list = image_list
        self.image_name_list = image_name_list
        self.img_size = img_size
        
        self.dataloader = None
        self.confthre = 0.25
        self.nmsthre = 0.45
        self.num_classes = None
        self.args = None

    def evaluate(
        self,
        imgs,
        image_name,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        global model, names, device
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        frame_id = 0
        total_l1 = 0
        total_l2 = 0
        strategy = 1
        max_tracker_num = int(8)
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        # for cur_iter, (imgs,path) in enumerate(
        #     progress_bar(self.dataloader)
        #     ):

        print('strategy:',strategy)
        # print(path)
        
        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        adam_opt = Adam([bx], lr=learning_rate, amsgrad=True)
        imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)
        #(1,23625,6)
        
        for iter in tqdm(range(epochs)):
            added_imgs = imgs+bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            added_imgs.clamp_(min=0, max=1)
            input_imgs = (added_imgs - rgb_means)/std
            if half:
                input_imgs = input_imgs.half()
            # outputs = model(input_imgs) # lifang535 remove
            outputs = model(added_imgs) # lifang535 add
            bx = run_attack(outputs,bx, strategy, max_tracker_num, adam_opt)
        if strategy == max_tracker_num-1:
            strategy = 0
        else:
            strategy += 1
        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        
        
        input_path = f"{input_dir}/{image_name}"
        output_path = f"{output_dir}/{image_name}"
        cv2.imwrite(output_path, added_blob) # lifang535: 这个 attack 效果似乎不受小数位损失影响
        
        print(f"saved image to {output_path}")
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms = infer(input_path)
        _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _car_num_after_nms = infer(output_path)
        
        logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, car_num_after_nms: {car_num_after_nms} -> _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _car_num_after_nms: {_car_num_after_nms}")

        
        # save_dir = path[0].replace("ori_img.jpg", "rao_img_3.png")
        # result_dir = os.path.dirname(save_dir)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        #     print(save_dir)
        # cv2.imwrite(save_dir, added_blob)
        # save_dir = path[0].replace("ori_img.jpg", "rao_3.png")
        # bxaaa = torch.clamp(bx*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # print(np.max(bxaaa))
        # print(bxaaa.shape)
        # bxaaa = bxaaa[..., ::-1]
        # cv2.imwrite(save_dir, bxaaa)
        # outputs = outputs.unsqueeze(0)
        # outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        # #scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
        # outputs = outputs[0].detach().cpu().numpy()
        # detections = outputs[:, :7]
        # #detections[:, :4] /= scale
        # aaa_img = torch.clamp(added_imgs*255,0,255)
        # print(len(detections))
        # hua_img = cv2.imread('./tutu/kuang_2.png')
        # for det in detections:
        #     left, top, right, bottom, score = det[:5]
        #     # 将坐标转换为整数
        #     left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        #     # 在图像上绘制矩形
        #     cv2.rectangle(hua_img, (left, top), (right, bottom), (0, 255, 0), 2)
        # result_file_path = path[0].replace("ori_img.jpg", "kuang_3.png")

        # hua_2_img = np.ascontiguousarray(added_blob)
        # cv2.imwrite(result_file_path, hua_img)
        # for det in detections:
        #     left, top, right, bottom, score = det[:5]
        #     # 将坐标转换为整数
        #     left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        #     # 在图像上绘制矩形
        #     cv2.rectangle(hua_2_img, (left, top), (right, bottom), (0, 0, 255), 2)
        # result_file_path = path[0].replace("ori_img.jpg", "kuang_4.png")
        # cv2.imwrite(result_file_path, hua_2_img)
        
        print(l1_norm.item(),l2_norm.item())
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        print(mean_l1.item(),mean_l2.item())
        del bx
        del adam_opt
        del outputs
        del imgs

        return mean_l1,mean_l2

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xywh2xyxy(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def run(self):
        """
        Run the evaluation.
        """
        for image, image_name in zip(self.image_list, self.image_name_list):
            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(device).float()
            image /= 255.0

            if len(image.shape) == 3:
                image = image[None]

            # print(f"image.shape = {image.shape}")
            
            mean_l1, mean_l2 = self.evaluate(image, image_name)


def infer(image_path):
    image = cv2.imread(image_path)
    # print(f"image.shape = {image.shape}") # (608, 1088, 3)
    # print(f"image = {image}")

    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device).float()
    image /= 255.0
    
    if len(image.shape) == 3:
        image = image[None]
    
    # tensor_type = torch.cuda.FloatTensor
    # image_tensor = image.type(tensor_type)
    # image_tensor = image.to(device)
    
    image_tensor = image
    
    # print(f"image_tensor = {image_tensor}")
    
    outputs = model(image_tensor)
    
    # print(f"outputs = {outputs}")
    
    outputs = outputs[0].unsqueeze(0)
    
    # scores = outputs[..., index] * outputs[..., 4]
    # scores = scores[scores > 0.25]
    # print(f"len(scores) = {len(scores)}")
    # objects_num_before_nms = len(scores) # 实际上是 {attack_object} number before NMS
    
    conf_thres = 0.25 # 0.25  # confidence threshold
    iou_thres = 0.45  # 0.45  # NMS IOU threshold
    max_det = 100000    # maximum detections per image
    
    xc = outputs[..., 4] > 0
    x = outputs[0][xc[0]]
    x[:, 5:] *= x[:, 4:5]
    max_scores = x[:, 5:].max(dim=-1).values
    objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
    objects_num_after_nms = 0
    person_num_after_nms = 0
    car_num_after_nms = 0
    
    outputs = non_max_suppression(outputs, conf_thres, iou_thres, max_det=max_det)
    
    for i, det in enumerate(outputs): # detections per image
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence}" # f"{confidence:.2f}"
                box = [round(float(i), 2) for i in xyxy]
                # print(f"Detected {label} with confidence {confidence_str} at location {box}")
                if label == "person":
                    person_num_after_nms += 1
                elif label == "car":
                    car_num_after_nms += 1
            objects_num_after_nms = len(det)
        # print(f"There are {len(det)} objects detected in this image.")
    
    # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
    print(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, car_num_after_nms = {car_num_after_nms}")
    return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms


def dir_process(dir_path):
    image_list = []
    image_name_list = os.listdir(dir_path)
    image_name_list.sort()
    # print(f"image_name_list = {image_name_list}")
    for image_name in image_name_list:
        if image_name.endswith(".png"):
            image_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_path)
            # print(f"image.shape = {image.shape}") # (608, 1088, 3)
            image_list.append(image)

    return image_list, image_name_list


if __name__ == "__main__":
    # image_name = "000001.png"
    # input_path = f"original_image/{image_name}"
    # output_path = f"single_attack_image/person_epochs_200/{image_name}"
    # weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    # device = torch.device('cuda:1')
    # model = DetectMultiBackend(weights=weights, device=device)
    # names = model.names
    # infer(output_path)
    # time.sleep(10000000)

    weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    device = torch.device('cuda:2')
    model = DetectMultiBackend(weights=weights, device=device)
    names = model.names
    print(f"names = {names}")
    
    attack_object_key = 0 # 0: person, 2: car
    attack_object = names[attack_object_key]
    index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

    epochs = 200
    learning_rate = 0.01 #0.07
    
    logger_path = f"log/single_attack/single_attack_{attack_object}_epochs_{epochs}.log"
    logger = create_logger(f"single_attack_{attack_object}_epochs_{epochs}", logger_path, logging.INFO)
    
    # logger = create_logger(f"phantom_attack_{attack_object}_epochs_{epochs}", f"single_attack_{attack_object}_epochs_{epochs}.log", logging.INFO)
    
    input_dir = "original_image"
    output_dir = f"single_attack_image/{attack_object}_epochs_{epochs}"
    
    # start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_size = (608, 1088)
    image_list, image_name_list = dir_process(input_dir)
    
    sa = SingleAttack(
        image_list=image_list,
        image_name_list=image_name_list,
        img_size=img_size,
    )
    
    sa.run()
    
    
    # single_attack(image_list, image_name_list)
    
    # TODO: 测一下哪步时延长
