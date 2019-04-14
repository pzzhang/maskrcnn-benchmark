# single gpu training
python tools/train_net.py --gpu_ids 2 --config-file "configs/vg_attribute.yaml" SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1520000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 8

# multi gpu training
python -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py --gpu_ids 2,5 --config-file "configs/vg_attribute.yaml" --skip-test SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1520000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 4

# single gpu testing
python tools/test_net.py --gpu_ids 2 --config-file "configs/vg_attribute.yaml" TEST.IMS_PER_BATCH 8


python wrapper.py --data-dir '/var/maskrcnn-benchmark'

python wrapper.py --data-dir ./ --out-dir ./ --config-file "configs/vg/rpn_faster_rcnn_R_101_FPN.yaml" --extra-args 'SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1520000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 8'


python tools/test_net.py --gpu_ids 2 --config-file "configs/vg/attr_faster_rcnn_R_101_FPN_minitest.yaml" TEST.IMS_PER_BATCH 8