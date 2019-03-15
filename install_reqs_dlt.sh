#!/usr/bin/bash
#cd /tmp/
#git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
#cp -rf $AZ_BATCHAI_JOB_MOUNT_ROOT/data/DisentangledRelationDetection/maskrcnn-benchmark/maskrcnn_benchmark/* /tmp/maskrcnn-benchmark/maskrcnn_benchmark/.
#cd /tmp/maskrcnn-benchmark
#cd $/var/DisentangledRelationDetection/maskrcnn-benchmark  
pip install --user future
python setup.py clean --all 
python setup.py build develop
pip install -U pip
pip install --user opencv-python
#pip install --user tabulate
#pip install --user torch torchvision
#pip install --user easydict
pip install --user progressbar
pip install --user tqdm
# pip install --user ipdb
#pip install --user scikit-image
#pip install --user torchtext
#pip install --user -U spacy
#python -m spacy download en

