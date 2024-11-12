import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image, ImageDraw
import inspect
import json
import pdb
import CONSTANTS

def print_function_params(func):
    # 获取函数的签名
    signature = inspect.signature(func)
    
    # 打印参数列表
    print(f"Function '{func.__name__}' parameters:")
    for param in signature.parameters.values():
        print(param)

def visualize_predictions(image, pred_boxes, pred_labels, gt_boxes, gt_labels, image_id):
  """
  Visualizes the predicted and ground truth bounding boxes on an image.

  Parameters:
  - image (PIL.Image or np.array): The image to draw boxes on.
  - pred_boxes (torch.Tensor or np.array): Predicted bounding boxes in format [x_min, y_min, width, height].
  - pred_labels (torch.Tensor or list): Predicted labels for each box.
  - gt_boxes (list or np.array): Ground truth bounding boxes in format [x_min, y_min, width, height].
  - gt_labels (list): Ground truth labels for each box.
  """
  # Convert torch.Tensor to numpy array if needed
  if isinstance(pred_boxes, torch.Tensor):
      pred_boxes = pred_boxes.cpu().numpy()
  if isinstance(pred_labels, torch.Tensor):
      pred_labels = pred_labels.cpu().numpy()
  
  # Draw predicted boxes in red and ground truth boxes in green
  draw = ImageDraw.Draw(image)

  # Draw predicted bounding boxes in red
  for box, label in zip(pred_boxes, pred_labels):
      x_min, y_min, width, height = box
      x_max, y_max = x_min + width, y_min + height
      draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
      draw.text((x_min, y_min), f"Pred: {label}", fill="red")

  # Draw ground truth bounding boxes in green
  for box, label in zip(gt_boxes, gt_labels):
      x_min, y_min, width, height = box
      x_max, y_max = x_min + width, y_min + height
      draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
      draw.text((x_min, y_min), f"GT: {label}", fill="green")

    # Save the image to the specified path
  plt.figure(figsize=(8, 8))
  plt.imshow(image)
  plt.axis("off")
  plt.savefig("../test_img/"+str(image_id)+"-pred-gt.jpg", bbox_inches="tight")
  plt.close()

def parse_example(example):
  
  image_id = example["image_id"]
  image = example["image"]
  width = example["width"]
  height = example["height"]
  bbox_id = example["objects"]["bbox_id"]
  category = example["objects"]["category"]
  bbox = example["objects"]["bbox"]
  area = example["objects"]["area"]
  
  return image_id, image, width, height, bbox_id, category, bbox, area
  
def parse_prediction(results):
  scores = results["scores"]
  labels = results["labels"]
  boxes = results["boxes"]
  return scores, labels, boxes

def get_label_name(idx):
    return CONSTANTS.DETR_CLASSES[idx]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    return iou


def evaluate_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()  # Tracks matched ground truth boxes

    # 获取按分数降序排列的预测框索引

    sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)

    # 按分数从高到低处理每个预测框
    for idx in sorted_indices:
        pred_box = pred_boxes[idx]  # 获取当前预测框的坐标
        max_iou = 0
        matched_gt_idx = -1

        # 找到与当前预测框 IoU 最高的真实框

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                matched_gt_idx = gt_idx

        # 检查 IoU 是否满足阈值，且该真实框尚未匹配
        if max_iou >= iou_threshold and matched_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(matched_gt_idx)
        else:
            fp += 1

    # 计算未匹配的真实框作为假阴性
    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn
  
def calculate_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall
  
  
def calculate_mAP(pred_boxes, gt_boxes, iou_thresholds=np.array([0.5])):
  ap_sum = 0

  for iou_threshold in iou_thresholds:
    tp, fp, fn = evaluate_predictions(pred_boxes, gt_boxes, iou_threshold)
    precision, recall = calculate_precision_recall(tp, fp, fn)
    ap_sum += precision

  # mAP
  mAP = ap_sum / len(iou_thresholds)
  return mAP


def save_evaluation_to_json(image_id, pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
  ap_sum = 0
  pred_boxes = pred_boxes.numpy()
  pred_scores = pred_scores.numpy()
  gt_boxes = np.array(gt_boxes)
  img_result = {}

  tp, fp, fn = evaluate_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold)
  precision, recall = calculate_precision_recall(tp, fp, fn)

  img_result["iou_threshold"]= iou_threshold
  img_result["tp"] = tp
  img_result["fp"] = fp
  img_result["fn"] = fn
  img_result["precision"] = precision
  img_result["recall"] = recall
  
  return img_result

def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    else:
        return data