import os, argparse


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--data-dir",
    default=".",
    metavar="DIR",
    help="data dir for training",
    type=str,
)
parser.add_argument(
    "--out-dir",
    default=".",
    metavar="DIR",
    help="output dir for model",
    type=str,
)
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()
config_file = args.config_file
data_dir = args.data_dir
out_dir = args.out_dir

os.system('python setup.py clean --all')
os.system('python setup.py build develop')
os.system('python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py \
	--data-dir {0} --out-dir {1} --gpu_ids 0,1,2,3 --config-file {2} \
	--skip-test SOLVER.IMS_PER_BATCH 8 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 1520000 \
    SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 8'.format(data_dir, out_dir, config_file))