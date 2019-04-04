import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
cfg.merge_from_file('configs/vg_attribute.yaml')
data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)

output_folder="/var/maskrcnn-benchmark/models/detection_with_attribute_bs8/inference/vg_val/"
iou_types="bbox"
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
dataset = data_loaders_val[0].dataset
eval_attributes=False
box_only=True
predictions=torch.load("/var/maskrcnn-benchmark/models/detection_with_attribute_bs8/inference/vg_val/predictions.pth")

evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, box_only=box_only, eval_attributes=eval_attributes, iou_types=iou_types)