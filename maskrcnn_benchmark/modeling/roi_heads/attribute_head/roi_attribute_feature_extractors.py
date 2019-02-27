# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ..box_head.roi_box_feature_extractors import *
from maskrcnn_benchmark.modeling import registry


registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)

registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "FPN2MLPFeatureExtractor", FPN2MLPFeatureExtractor
)

registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS.register(
    "FPNXconv1fcFeatureExtractor", FPNXconv1fcFeatureExtractor
)


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_ATTRIBUTE_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
