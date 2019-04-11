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
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()
config_file = args.config_file
data_dir = args.data_dir
out_dir = args.out_dir
opts = args.opts

os.system('python setup.py clean --all')
os.system('python setup.py build develop')
os.system('python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py \
	--data-dir {0} --out-dir {1} --gpu_ids 0,1,2,3 --config-file {2} \
	--skip-test {3}'.format(data_dir, out_dir, config_file, opts))