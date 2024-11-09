from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image
import requests


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
import CONSTANTS
learning_rate = 0.02 #0.07
epochs = 150

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

logger = create_logger("attack_data", "attack_data.log", logging.INFO)

def generate_mask(outputs,result, x_shape, y_shape):

    mask_x = 4
    mask_y = 2
    # mask = torch.ones(mask_y,mask_x)  # 初始mask为3*3
    mask = torch.ones(y_shape,x_shape)
    
    boxes = result["boxes"]

    x_len = int(x_shape / mask_x)
    y_len = int(y_shape / mask_y)
    if boxes is not None:
        for i in range(len(boxes)):
            detection = boxes[i]
            center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2

            # Based on the position of the center of the detection box, determine which region it is in
            region_x = int(center_x / x_len)
            region_y = int(center_y / y_len)
            
            mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05
    
    
    return mask

def run_attack(outputs,result,bx, strategy, max_tracker_num, mask):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    # scores = outputs[:,5] * outputs[:,4] # remove
    
    scores = result["scores"] # add

    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    loss = loss3#+loss2
    
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -3.5 * mask * bx.grad+ bx.data
    count = (scores > 0.9).sum()
    print('loss',loss.item(),'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item())
    return bx

def attack(model, image_processor, inputs, strategy=0, max_tracker_num=15, epochs=200, device=None):
  outputs = None
  imgs = inputs["pixel_values"]

  
  bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
  bx = bx.astype(np.float32)
  bx = torch.from_numpy(bx).to(device).unsqueeze(0)
  bx = bx.data.requires_grad_(True)
  # imgs = imgs.type(tensor_type)
  imgs = imgs.to(device)
  
  
  for iter in tqdm(range(epochs)):
    
    added_imgs = imgs+bx
    
    l2_norm = torch.sqrt(torch.mean(bx ** 2))
    l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
    
    outputs = None
    
    outputs = model(added_imgs)
    
    target_size = [imgs.shape[2:] for _ in range(1)]
    result = image_processor.post_process_object_detection(outputs, 
                                                            threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                            target_sizes = target_size)[0]

    
    if iter == 0:
      mask = generate_mask(outputs, result, added_imgs.shape[3], added_imgs.shape[2]).to(device) # The mask is generated only once
    bx = run_attack(outputs,result,bx, strategy, max_tracker_num, mask)
    
  return



if __name__=="__main__":
  url = "http://images.cocodataset.org/val2017/000000039769.jpg"
  image = Image.open(requests.get(url, stream=True).raw)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
  inputs = image_processor(images=image, return_tensors="pt")
  
  attack(model, image_processor, inputs, device=device)