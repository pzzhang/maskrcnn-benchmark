# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList


# TODO check if want to return a single BoxList or a composite
# object
class AttributePostProcessor(nn.Module):
    """
    From the results of the CNN, post process the attributes
    by taking the attributes corresponding to the class with max
    probability (which are padded to fixed size) and return the 
    attributes in the mask field of the BoxList.
    """

    def __init__(self, cfg):
        super(AttributePostProcessor, self).__init__()
        self.max_num_attr = cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_NUM_ATTR_PER_OBJ
        # self.attr_thresh = cfg.MODEL.ROI_ATTRIBUTE_HEAD.POSTPROCESS_ATTRIBUTES_THRESHOLD

    def forward(self, x, boxes, features):
        """
        Arguments:
            x (Tensor): the attribute logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field attribute
        """
        attribute_prob = F.softmax(x, -1)
        # apply filter
        attribute_prob, attribute_inds = torch.sort(attribute_prob, descending=True)
        attribute_prob = attribute_prob[:, :self.max_num_attr]
        attribute_inds = attribute_inds[:, :self.max_num_attr]

        boxes_per_image = [len(box) for box in boxes]
        attribute_prob = attribute_prob.split(boxes_per_image, dim=0)
        attribute_inds = attribute_inds.split(boxes_per_image, dim=0)
        features = features.split(boxes_per_image, dim=0)

        results = []
        for attr_ind, box, feature in zip(attribute_inds, boxes, features):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("attribute", attr_ind)
            bbox.add_field("attr_feature", feature)
            results.append(bbox)

        return results


def make_roi_attribute_post_processor(cfg):
    attribute_post_processor = AttributePostProcessor(cfg)
    return attribute_post_processor
