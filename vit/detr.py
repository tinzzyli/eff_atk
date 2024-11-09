from transformers import AutoImageProcessor
from transformers import VitDetConfig, VitDetModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import RTDetrConfig, RTDetrModel
import torchvision.transforms as transforms
import torch
from datasets import load_dataset
import dataset
import CONSTANTS
import util
from PIL import Image
import requests
import numpy as np
import json
import pdb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on : ", device)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model.eval()

to_tensor = transforms.ToTensor()

if __name__ == "__main__":
  eval_results = {}
  # set up MS COCO 2017
  coco_data = load_dataset("detection-datasets/coco", split="val")
  # pdb.set_trace()
  for index, example in enumerate(coco_data):
    pass
  for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__(), desc="Processing COCO data"):
    # data and ground truth :)
    image_id, image, width, height, bbox_id, category, gt_boxes, area = util.parse_example(example)
    assert len(bbox_id) == len(category) == len(gt_boxes) == len(area)
    
    if image.mode != "RGB":
      image = image.convert("RGB")
        
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
      outputs = model(**inputs)
    target_size = torch.tensor([image.size[::-1]])
    # util.print_function_params(image_processor.post_process_object_detection)
    results = image_processor.post_process_object_detection(outputs, 
                                                            threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                            target_sizes = target_size)[0]
    results = util.move_to_cpu(results)
    pred_scores, pred_labels, pred_boxes = util.parse_prediction(results)
    
    # for visualization
    # util.visualize_predictions(image, pred_boxes, pred_labels, gt_boxes, category, image_id)
        
    # for score, label, box in zip(pred_scores, pred_labels, pred_boxes):
    #   box = [round(i, 2) for i in box.tolist()]
    #   print(
    #       f"Detected {model.config.id2label[label.item()]}-{label.item()} with confidence "
    #       f"{round(score.item(), 3)} at location {box}"
    #   )
    # for c, b in zip(category, gt_boxes):
    #   print(f"Detected {CONSTANTS.COCO_CLASSES[c]}-{c} at location {b}")
    # print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    
    
    img_result = util.save_evaluation_to_json(image_id, 
                                              pred_boxes, 
                                              pred_scores,
                                              gt_boxes, 
                                              iou_threshold=0.5)
    
    eval_results[f"image_{image_id}"] = img_result
  
  output_path = "../prediction/detr_eval_result.json"
  with open(output_path, "w") as f:
    json.dump(eval_results, f, indent=4)

  print(f"Evaluation results saved to {output_path}")