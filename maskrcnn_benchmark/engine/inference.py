# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    results_attribute_dict = {}
    flag_attribute = False
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output_dict = model(images, targets)
            output = output_dict['detection_output']
            output = [o.to(cpu_device) for o in output]
            if 'attribute_output' in output_dict:
                flag_attribute = True
                output_attribute = output_dict['attribute_output']
                output_attribute = [o.to(cpu_device) for o in output_attribute]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        if flag_attribute:
            results_attribute_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output_attribute)}
            )
    if flag_attribute:
        return dict(results_dict=results_dict, results_attribute_dict=results_attribute_dict)
    else:
        return dict(results_dict=results_dict)


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        has_attribute=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions_dict = compute_on_dataset(model, data_loader, device)
    predictions = predictions_dict['results_dict']
    flag_attribute = False
    if "results_attribute_dict" in predictions_dict:
        flag_attribute = True
        predictions_attribute = predictions_dict['results_attribute_dict']
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if flag_attribute:
        predictions_attribute = _accumulate_predictions_from_multiple_gpus(predictions_attribute)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        if flag_attribute:
            torch.save(predictions_attribute, os.path.join(output_folder, "predictions_attribute.pth"))

    if not dataset_name.startswith("vg"):
        extra_args = dict(
            box_only=box_only,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )

        return evaluate(dataset=dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)
    else:  ## vg dataset

        extra_args = dict(
            box_only=box_only,
            eval_attributes=False,
            iou_types=iou_types,
            expected_results=expected_results,
            expected_results_sigma_tol=expected_results_sigma_tol,
        )

        object_evaluation =  evaluate(dataset=dataset,
                                      predictions=predictions,
                                      output_folder=output_folder,
                                      **extra_args)

        if has_attribute:
            extra_args = dict(
                box_only=box_only,
                eval_attributes=True,
                iou_types=iou_types,
                expected_results=expected_results,
                expected_results_sigma_tol=expected_results_sigma_tol,
            )

            attribute_evaluation = evaluate(dataset=dataset,
                                            predictions=predictions_attribute,
                                            output_folder=output_folder,
                                            **extra_args)
            return dict(object_evaluation=object_evaluation, attribute_evaluation=attribute_evaluation)

        return object_evaluation