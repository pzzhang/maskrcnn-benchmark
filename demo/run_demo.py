from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import to_image_list

import torch
import cv2
import pdb

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image1 = cv2.imread("/var/maskrcnn-benchmark/datasets/coco/val2017/000000000139.jpg")
image2 = cv2.imread('/var/maskrcnn-benchmark/datasets/coco/train2017/000000498666.jpg')

predictions = coco_demo.compute_prediction(image1)
print(predictions.fields())

image_list = [coco_demo.transforms(image) for image in [image1, image2]]
image_list = to_image_list(image_list, coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY)
image_list = image_list.to(coco_demo.device)
with torch.no_grad():
	predictions = coco_demo.model(image_list)

pdb.set_trace()