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
# hit = 2
def allocation_strategy(frameRate):
    K = 1000
    next_reactivate = {}  # 下一次需要重新激活的时间
    max_active = 0  # 最大可用跟踪器数量
    strategy = []
    tracker_id = 0
    t = 0
    while(t<K):
        if t == 0:
            strategy.append(tracker_id)
            next_reactivate[tracker_id] = t+frameRate+1
            tracker_id += 1
            t+=1
        else:
            reactivate_time = min(next_reactivate.values())
            reactivate_tracker, reactivate_time = min(next_reactivate.items(), key=lambda x: x[1])
            if reactivate_time - t <= 1:
                strategy.append(reactivate_tracker)
                next_reactivate[reactivate_tracker] = t+frameRate+1
                t+=1
            else:
                strategy.append(tracker_id)
                strategy.append(tracker_id)
                next_reactivate[tracker_id] = t+frameRate+2
                tracker_id += 1
                t+=2
    return strategy, tracker_id

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

def run_attack(outputs,outputs_2,bx, strategy, max_tracker_num, adam_opt):
    outputs = outputs[0][0]
    outputs_2 = outputs_2[0][0]

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    scores = outputs[:,5] * outputs[:,4]
    scores_2 = outputs_2[:,5] * outputs_2[:,4]
    sel_scores_b = scores[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_scores_m = scores[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_scores_s = scores[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]
    sel_scores_b_2 = scores_2[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_scores_m_2 = scores_2[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_scores_s_2 = scores_2[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]

    sel_dets = torch.cat((sel_scores_b, sel_scores_m, sel_scores_s), dim=0)
    sel_dets_2 = torch.cat((sel_scores_b_2, sel_scores_m_2, sel_scores_s_2), dim=0)
    targets = torch.ones_like(sel_dets) # lifang535 remove
    # targets = torch.zeros_like(sel_dets) # lifang535 add
    loss1 = 10*(F.mse_loss(sel_dets, targets, reduction='sum')+F.mse_loss(sel_dets_2, targets, reduction='sum')) # lifang535 remove
    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores) # lifang535 remove
    # targets = torch.zeros_like(scores) # lifang535 add
    loss3 = 1.0*(F.mse_loss(scores, targets, reduction='sum')+F.mse_loss(scores_2, targets, reduction='sum'))
    # loss = loss1+loss3#+loss2 # lifang535 remove
    loss = loss1+loss2+loss3 # lifang535 add
    
    loss.requires_grad_(True)
    adam_opt.zero_grad()
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    count = (scores > 0.25).sum()
    print('loss',loss.item(),'loss_1',loss1.item(),'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item()) # lifang535 remove
    # print('loss',loss.item(),'loss_1',0,'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item()) # lifang535 add
    return bx



class StraAttack:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        image_list,
        image_name_list,
        img_size):
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
        
        self.cur_iter = 0

    def evaluate(
        self,
        imgs,
        image_name,
        imgs_2,
        image_name_2,
        
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
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
        strategy = 0
        max_tracker_num = int(6)
        strategy, max_tracker_num = allocation_strategy(max_tracker_num)
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        # for cur_iter, (imgs,path,imgs_2,path_2) in enumerate(
        #     progress_bar(self.dataloader)
        #     ):

        print('strategy:',strategy[self.cur_iter])
        # print(path,path_2)
        
        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        adam_opt = Adam([bx], lr=learning_rate, amsgrad=True)
        imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)
        imgs_2 = imgs_2.type(tensor_type)
        imgs_2 = imgs_2.to(device)
        #(1,23625,6)
        
        for iter in tqdm(range(epochs)):
            added_imgs = imgs+bx
            added_imgs_2 = imgs_2+bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            added_imgs.clamp_(min=0, max=1)
            added_imgs_2.clamp_(min=0, max=1)
            input_imgs = (added_imgs - rgb_means)/std # lifang535: 为什么要减去 rgb_means，改
            input_imgs_2 = (added_imgs_2 - rgb_means)/std
            if half:
                input_imgs = input_imgs.half()
                input_imgs_2 = input_imgs_2.half()
            # outputs = model(input_imgs)[0] # lifang535 remove
            # outputs_2 = model(input_imgs_2)[0] # lifang535 remove
            outputs = model(added_imgs) # lifang535 add
            outputs_2 = model(added_imgs_2) # lifang535 add
            bx = run_attack(outputs,outputs_2,bx, strategy[self.cur_iter], max_tracker_num, adam_opt)

        # if strategy == max_tracker_num-1:
        #     strategy = 0
        # else:
        #     strategy += 1
        self.cur_iter += 1
        
        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        added_blob_2 = torch.clamp(added_imgs_2*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob_2 = added_blob_2[..., ::-1]
        
        input_path = f"{input_dir}/{image_name}"
        output_path = f"{output_dir}/{image_name}"
        cv2.imwrite(output_path, added_blob) # lifang535: 这个 attack 效果似乎不受小数位损失影响
        print(f"saved image to {output_path}")
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms = infer(input_path)
        _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _car_num_after_nms = infer(output_path)
        logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, car_num_after_nms: {car_num_after_nms} -> _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _car_num_after_nms: {_car_num_after_nms}")
        
        input_path_2 = f"{input_dir}/{image_name_2}"
        output_path_2 = f"{output_dir}/{image_name_2}"
        cv2.imwrite(output_path_2, added_blob_2) # lifang535: 这个 attack 效果似乎不受小数位损失影响
        print(f"saved image to {output_path_2}")
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms = infer(input_path_2)
        _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _car_num_after_nms = infer(output_path_2)
        logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, car_num_after_nms: {car_num_after_nms} -> _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _car_num_after_nms: {_car_num_after_nms}")
        
        
        # save_dir = path[0].replace("dataset", "botsort_stra")
        # save_dir_2 = path_2[0].replace("dataset", "botsort_stra")
        # result_dir = os.path.dirname(save_dir)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        #     print(save_dir)
        # cv2.imwrite(save_dir, added_blob)
        # cv2.imwrite(save_dir_2, added_blob_2)
        print(l1_norm.item(),l2_norm.item())
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        print(mean_l1.item(),mean_l2.item())
        del bx
        del adam_opt
        del outputs
        del outputs_2
        del imgs
        del imgs_2

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
        # for image, image_name in zip(self.image_list, self.image_name_list):
        # 每次处理两张图片
        for i in range(0, len(self.image_list), 2):
            image = self.image_list[i]
            image_name = self.image_name_list[i]

            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(device).float()
            image /= 255.0
            
            if len(image.shape) == 3:
                image = image[None]
            
            image_2 = self.image_list[i+1]
            image_name_2 = self.image_name_list[i+1]
            
            image_2 = image_2.transpose((2, 0, 1))[::-1]
            image_2 = np.ascontiguousarray(image_2)
            image_2 = torch.from_numpy(image_2).to(device).float()
            image_2 /= 255.0
                
            if len(image_2.shape) == 3:
                image_2 = image_2[None]

            # print(f"image.shape = {image.shape}")
            
            mean_l1, mean_l2 = self.evaluate(image, image_name, image_2, image_name_2)
            

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
    max_det = 10000    # maximum detections per image
    
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
    # output_path = f"stra_attack_image/person_epochs_200/{image_name}"
    # weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    # device = torch.device('cuda:1')
    # model = DetectMultiBackend(weights=weights, device=device)
    # names = model.names
    # infer(output_path)
    # time.sleep(10000000)

    weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    device = torch.device('cuda:3')
    model = DetectMultiBackend(weights=weights, device=device)
    names = model.names
    print(f"names = {names}")
    
    attack_object_key = 0 # 0: person, 2: car
    attack_object = names[attack_object_key]
    index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

    epochs = 200
    learning_rate = 0.01 #0.07
    
    logger_path = f"log/stra_attack/stra_attack_{attack_object}_epochs_{epochs}.log"
    logger = create_logger(f"stra_attack_{attack_object}_epochs_{epochs}", logger_path, logging.INFO)
    
    # logger = create_logger(f"stra_attack_{attack_object}_epochs_{epochs}", f"stra_attack_{attack_object}_epochs_{epochs}.log", logging.INFO)
        
    input_dir = "original_image"
    output_dir = f"stra_attack_image/{attack_object}_epochs_{epochs}"
    
    # start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_size = (608, 1088)
    image_list, image_name_list = dir_process(input_dir)
    
    sa = StraAttack(
        image_list=image_list,
        image_name_list=image_name_list,
        img_size=img_size,
    )
    
    sa.run()
    
    
    # stra_attack(image_list, image_name_list)
    
    # TODO: 测一下哪步时延长
